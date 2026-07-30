"""
Common Feature Engineering Component.

Converts the Master Dataset (single source of truth) into a model-agnostic
Feature Store consumable by every downstream ML pipeline (Forecasting,
Salary Prediction, NLP Analytics, Recommendation System).

Design principles:
    * Single Responsibility  -> one method per transformation.
    * Open/Closed            -> new transformations are added as new methods
                                 + new config flags, existing ones untouched.
    * Composition            -> `run()` composes independent steps; no
                                 inheritance hierarchy required.
    * Configuration-driven   -> zero hardcoded column names in this file.
    * Deterministic          -> list/category normalization is sorted;
                                 no randomness anywhere in this stage.

This stage intentionally does NOT: one-hot encode, TF-IDF, train models,
split train/test, create lag/rolling features, or leak target information.
Those all belong to task-specific pipelines that consume this Feature Store.
"""

from __future__ import annotations

from datetime import datetime, timezone
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from ..configs.common_feature_engineering_config import (
    CommonFeatureEngineeringConfig,
)
from ..entity.common_feature_engineering_entity import (
    FeatureEngineeringReport,
    FeatureEngineeringResult,
    FeatureEngineeringSummary,
    TransformationStepSummary,
)

from ..exception import CustomException
from ..logger import logging
from ..utils.common_feature_engineering_utils import (
    compute_schema_hash,
    ensure_dir,
    generate_dataset_id,
    infer_column_types,
    list_length,
    normalize_boolean,
    normalize_category,
    normalize_list_value,
    normalize_text,
    save_json,
    schema_dict,
    to_datetime_safe,
    validate_feature_store_schema,
)

SEP = "=" * 70


class CommonFeatureEngineering:
    """Model-agnostic Feature Store builder consumed by every downstream pipeline."""

    def __init__(self, config: CommonFeatureEngineeringConfig):
        self.config = config
        self.steps: List[TransformationStepSummary] = []
        self.dataset_id = generate_dataset_id(prefix="fs")
        
        # Precompute set of truthy values for boolean normalization
        self.truthy_values = {
            str(v).strip().lower() if isinstance(v, str) else v
            for v in self.config.boolean_truthy_values
        }
        
        self.dtype_conversions: Dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Orchestrator
    # ------------------------------------------------------------------ #
    def run(self) -> FeatureEngineeringResult:
        try:
            start_time = time.time()
            
            self.steps.clear()
            self.dtype_conversions.clear()
            
            logging.info(SEP)
            logging.info("COMMON FEATURE ENGINEERING STARTED")
            logging.info(SEP)

            df = self._load_master_dataset()
            rows_before, cols_before = df.shape
            null_before = int(df.isna().sum().sum())

            # Note: Column groups describe only the original master dataset schema.
            col_groups = infer_column_types(df, self.config)

            df = self._standardize_dtypes(df, col_groups)
            df, identifiers_dropped = self._drop_identifiers(df, col_groups)
            df = self._standardize_datetime(df, col_groups)
            df, missing_filled = self._handle_missing_values(df)
            df = self._standardize_booleans(df, col_groups)
            df = self._clean_categoricals(df, col_groups)
            df = self._clean_text(df, col_groups)
            df = self._standardize_list_columns(df, col_groups)
            df, columns_created = self._create_derived_features(df)
            df, duplicates_removed = self._remove_duplicates(df)

            self._final_validation(df)

            rows_after, cols_after = df.shape
            null_after = int(df.isna().sum().sum())
            schema_hash = compute_schema_hash(df)

            summary = FeatureEngineeringSummary(
                dataset_id=self.dataset_id,
                rows_before=rows_before,
                rows_after=rows_after,
                rows_removed=rows_before - rows_after,
                columns_before=cols_before,
                columns_after=cols_after,
                identifiers_dropped=identifiers_dropped,
                columns_removed=identifiers_dropped,
                columns_created=columns_created,
                dtype_conversions=self.dtype_conversions,
                missing_values_filled=missing_filled,
                duplicates_removed=duplicates_removed,
                null_values_before=null_before,
                null_values_after=null_after,
                schema_hash=schema_hash,
                execution_time_seconds=round(time.time() - start_time, 4),
                integrity_passed=True,
            )

            result = self._save_outputs(df, summary)

            logging.info(
                "Build summary: %s",
                {
                    "dataset_id": summary.dataset_id,
                    "rows": f"{summary.rows_before} -> {summary.rows_after}",
                    "columns": f"{summary.columns_before} -> {summary.columns_after}",
                    "columns_created": len(summary.columns_created),
                    "columns_removed": len(summary.columns_removed),
                    "nulls": f"{summary.null_values_before} -> {summary.null_values_after}",
                    "execution_time_seconds": summary.execution_time_seconds,
                    "schema_hash": summary.schema_hash[:16] + "...",
                },
            )
            logging.info(SEP)
            logging.info("COMMON FEATURE ENGINEERING COMPLETED")
            logging.info(SEP)
            return result

        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    # 0. Load
    # ------------------------------------------------------------------ #
    def _load_master_dataset(self) -> pd.DataFrame:
        try:
            path = Path(self.config.master_dataset_path)
            if not path.exists():
                raise FileNotFoundError(f"Master dataset not found at: {path}")
            df = pd.read_parquet(path)
            logging.info(
                "Loaded master dataset from '%s' -> %s rows, %s columns",
                path, df.shape[0], df.shape[1],
            )
            return df
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    # 1. Data type standardization
    # ------------------------------------------------------------------ #
    def _standardize_dtypes(self, df: pd.DataFrame, col_groups: Dict[str, List[str]]) -> pd.DataFrame:
        """Mutates dataframe in-place."""
        logging.info("[dtype_standardization] START")
        t0 = time.time()
        try:
            numeric_cols = col_groups.get("numeric", self.config.numeric_columns)
            
            for col in numeric_cols:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    old_dtype = str(df[col].dtype)
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    new_dtype = str(df[col].dtype)
                    self.dtype_conversions[col] = f"{old_dtype} -> {new_dtype}"

            logging.info(
                "[dtype_standardization] END - Converted %s column(s) to numeric.",
                len(self.dtype_conversions),
            )
            self._record_step(
                "dtype_standardization",
                columns=list(self.dtype_conversions.keys()),
                rows_before=len(df),
                rows_after=len(df),
                details={"dtype_conversions": self.dtype_conversions},
                t0=t0,
            )
            return df
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    # 2. Identifier handling
    # ------------------------------------------------------------------ #
    def _drop_identifiers(self, df: pd.DataFrame, col_groups: Dict[str, List[str]]) -> Tuple[pd.DataFrame, List[str]]:
        logging.info("[identifier_handling] START")
        t0 = time.time()
        try:
            identifiers = col_groups.get("identifier", self.config.identifier_columns)
            present = [c for c in identifiers if c in df.columns]
            missing = [c for c in identifiers if c not in df.columns]
            
            if missing:
                logging.warning(
                    "[identifier_handling] Configured identifier column(s) not found, skipped: %s",
                    missing,
                )
            df = df.drop(columns=present)
            logging.info("[identifier_handling] END - Dropped identifier column(s): %s", present)
            self._record_step(
                "identifier_handling",
                columns=present,
                rows_before=len(df),
                rows_after=len(df),
                details={"dropped": present, "missing_from_source": missing},
                t0=t0,
            )
            return df, present
        except Exception as e:
            raise CustomException(e, sys)
    # ------------------------------------------------------------------ #
    # 3. Datetime standardization (Memory & Speed Guard)
    # ------------------------------------------------------------------ #
    def _standardize_datetime(self, df: pd.DataFrame, col_groups: Dict[str, List[str]]) -> pd.DataFrame:
        """Mutates dataframe in-place."""
        logging.info("[datetime_standardization] START")
        t0 = time.time()
        try:
            converted = []
            datetime_cols = col_groups.get("datetime", self.config.datetime_columns)
            
            for col in datetime_cols:
                if col not in df.columns:
                    continue
                epoch_ms = col in self.config.epoch_ms_datetime_columns
                before_non_null = df[col].notna().sum()
                
                # NOTE: Ensure your utility `to_datetime_safe` utilizes pd.to_datetime(..., cache=True)
                # or specifies `format='mixed'` to prevent infinite hanging.
                df[col] = to_datetime_safe(df[col], epoch_ms=epoch_ms)
                
                after_non_null = df[col].notna().sum()
                converted.append(col)
                if after_non_null < before_non_null:
                    logging.warning(
                        "[datetime_standardization] '%s': %s value(s) could not be parsed -> set to NaT.",
                        col, before_non_null - after_non_null,
                    )
            logging.info("[datetime_standardization] END - Standardized column(s): %s", converted)
            self._record_step(
                "datetime_standardization",
                columns=converted,
                rows_before=len(df),
                rows_after=len(df),
                details={},
                t0=t0,
            )
            return df
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    # 4. Missing value strategy
    # ------------------------------------------------------------------ #
    def _handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Mutates dataframe in-place."""
        logging.info("[missing_value_handling] START")
        t0 = time.time()
        try:
            cfg = self.config
            total_filled = 0
            fill_log: Dict[str, Any] = {}

            for col in df.columns:
                # Determine strategy
                strategy = cfg.missing_value_strategy.get(col)
                n_missing = int(df[col].isna().sum())
                
                if n_missing == 0:
                    continue


                if strategy is None:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        strategy = cfg.default_numeric_strategy
                    elif pd.api.types.is_bool_dtype(df[col]):
                        strategy = "leave"
                    else:
                        strategy = cfg.default_categorical_strategy
                    logging.info("[missing_value_handling] Column='%s' Missing=%d Strategy=%s",col, n_missing, strategy, )

                if strategy == "leave":
                    continue

                fill_value = None
                
                if col in cfg.list_columns:
                    logging.info("[missing_value_handling] Skipping list column '%s'.", col)
                    continue
                
                # Process based on strategy
                if strategy == "median":
                    if pd.api.types.is_numeric_dtype(df[col]):
                        fill_value = df[col].median()
                    else:
                        logging.warning(
                            "[missing_value_handling] Cannot apply 'median' to non-numeric column '%s'. Skipping.", col
                        )
                        continue

                elif strategy == "mode":
                    # SAFEGUARD: Prevent pipeline freeze on high-cardinality text columns
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        unique_count = df[col].nunique()
                        # If a column has too many unique values, finding a mode will hang pandas
                        if unique_count > 100:
                            logging.warning(
                                "[missing_value_handling] Column '%s' has high cardinality (%d unique). "
                                "Skipping 'mode' strategy to prevent pipeline freeze.", 
                                col, unique_count
                            )
                            continue
                    
                    mode_vals = df[col].mode(dropna=True)
                    fill_value = mode_vals.iloc[0] if not mode_vals.empty else None

                elif strategy == "constant":
                    if col in cfg.constant_fill_values:
                        fill_value = cfg.constant_fill_values[col]
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        fill_value = cfg.default_constant_numeric_fill
                    else:
                        fill_value = cfg.default_constant_text_fill

                else:
                    logging.warning(
                        "[missing_value_handling] Unknown strategy '%s' for '%s', leaving as-is.",
                        strategy, col,
                    )
                    continue

                # Apply the fill
                if fill_value is not None:
                    df[col] = df[col].fillna(fill_value)
                    n_missing_after = int(df[col].isna().sum())
                    filled_count = n_missing - n_missing_after
                    total_filled += filled_count
                    fill_log[col] = {
                        "strategy": strategy,
                        "null_before": n_missing,
                        "null_after": n_missing_after,
                        "filled": filled_count
                    }

            logging.info(
                "[missing_value_handling] END - Filled %s null value(s) across %s column(s).",
                total_filled, len(fill_log),
            )
            self._record_step(
                "missing_value_handling",
                columns=list(fill_log.keys()),
                rows_before=len(df),
                rows_after=len(df),
                details=fill_log,
                t0=t0,
            )
            return df, total_filled
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    # 5. Boolean standardization
    # ------------------------------------------------------------------ #
    def _standardize_booleans(self, df: pd.DataFrame, col_groups: Dict[str, List[str]]) -> pd.DataFrame:
        """Mutates dataframe in-place."""
        logging.info("[boolean_standardization] START")
        t0 = time.time()
        try:
            converted = []
            boolean_cols = col_groups.get("boolean", self.config.boolean_columns)
            
            for col in boolean_cols:
                if col not in df.columns:
                    continue
                df[col] = df[col].apply(
                    lambda v: normalize_boolean(v, self.truthy_values)
                ).astype("boolean")  # nullable pandas BooleanDtype
                converted.append(col)
                
            logging.info("[boolean_standardization] END - Standardized column(s): %s", converted)
            self._record_step(
                "boolean_standardization",
                columns=converted,
                rows_before=len(df),
                rows_after=len(df),
                details={},
                t0=t0,
            )
            return df
        except Exception as e:
            raise CustomException(e, sys)

# ------------------------------------------------------------------ #
    # 6. Categorical cleanup (Optimized)
    # ------------------------------------------------------------------ #
    def _clean_categoricals(self, df: pd.DataFrame, col_groups: Dict[str, List[str]]) -> pd.DataFrame:
        """Mutates dataframe in-place. Uses vectorized string methods for speed."""
        logging.info("[categorical_cleanup] START")
        t0 = time.time()
        try:
            cleaned = []
            categorical_cols = col_groups.get("categorical", self.config.categorical_columns)
            
            for col in categorical_cols:
                if col not in df.columns:
                    continue
                
                # VECTORIZATION OVERRIDE: If the column is string/object, use fast C-level methods
                # rather than .apply() if possible. Assuming normalize_category does basic cleaning:
                if df[col].dtype == "object" or isinstance(df[col].dtype, pd.StringDtype):
                    # Fast track: lowercase, strip whitespace, handle basic nulls
                    df[col] = df[col].astype(str).str.lower().str.strip()
                    df[col] = df[col].replace({"nan": None, "none": None, "": None})
                else:
                    # Slow track fallback
                    df[col] = df[col].apply(normalize_category)
                    
                df[col] = df[col].astype("category")
                cleaned.append(col)
                
            logging.info("[categorical_cleanup] END - Normalized column(s): %s", cleaned)
            self._record_step(
                "categorical_cleanup",
                columns=cleaned,
                rows_before=len(df),
                rows_after=len(df),
                details={},
                t0=t0,
            )
            return df
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    # 7. Text cleanup (Safe Apply)
    # ------------------------------------------------------------------ #
    def _clean_text(self, df: pd.DataFrame, col_groups: Dict[str, List[str]]) -> pd.DataFrame:
        """Mutates dataframe in-place."""
        logging.info("[text_cleanup] START")
        t0 = time.time()
        try:
            cleaned = []
            text_cols = col_groups.get("text", self.config.text_columns)
            
            for col in text_cols:
                if col not in df.columns:
                    continue
                
                logging.info("[text_cleanup] Processing heavy text column: '%s'", col)
                # Safeguard: Ensure values are strings to prevent apply() crashing on floats
                # We use .map (slightly faster than apply for single-arg functions)
                df[col] = df[col].astype(str).map(normalize_text)
                cleaned.append(col)
                
            logging.info("[text_cleanup] END - Lightweight-cleaned column(s): %s", cleaned)
            self._record_step(
                "text_cleanup",
                columns=cleaned,
                rows_before=len(df),
                rows_after=len(df),
                details={},
                t0=t0,
            )
            return df
        except Exception as e:
            raise CustomException(e, sys)

# ------------------------------------------------------------------ #
    # 8. List column standardization
    # ------------------------------------------------------------------ #
    def _standardize_list_columns(self, df: pd.DataFrame, col_groups: Dict[str, List[str]]) -> pd.DataFrame:
        """Mutates dataframe in-place."""
        logging.info("[list_column_standardization] START")
        t0 = time.time()
        try:
            standardized = []
            list_cols = col_groups.get("list", self.config.list_columns)
            
            for col in list_cols:
                if col not in df.columns:
                    continue
                
                # SAFEGUARD: Convert numpy arrays to standard lists before passing to the utility.
                # This prevents "ambiguous truth value" exceptions inside normalize_list_value.
                df[col] = df[col].apply(
                    lambda v: normalize_list_value( v, delimiter=self.config.list_column_delimiter))
                
                standardized.append(col)
                
            logging.info("[list_column_standardization] END - Standardized column(s): %s", standardized)
            self._record_step(
                "list_column_standardization",
                columns=standardized,
                rows_before=len(df),
                rows_after=len(df),
                details={"delimiter": self.config.list_column_delimiter},
                t0=t0,
            )
            return df
        except Exception as e:
            raise CustomException(e, sys)
    # ------------------------------------------------------------------ #
    # 9. Derived generic features
    # ------------------------------------------------------------------ #
    def _add_feature(self, df: pd.DataFrame, name: str, series: pd.Series, created: List[str]) -> None:
        """Helper to assign derived features cast as pandas boolean or preserved dtype."""
        if pd.api.types.is_bool_dtype(series):
            df[name] = series.astype("boolean")
        else:
            df[name] = series
        created.append(name)

    def _create_derived_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        logging.info("[derived_features] START")
        t0 = time.time()
        try:
            flags = self.config.derived_feature_flags
            created: List[str] = []
            delimiter = self.config.list_column_delimiter

            if flags.get("salary_available") and any(
                c in df.columns for c in ("min_salary", "max_salary", "med_salary", "normalized_salary")
            ):
                salary_cols = [
                    c for c in ("min_salary", "max_salary", "med_salary", "normalized_salary")
                    if c in df.columns
                ]
                self._add_feature(df, "salary_available", df[salary_cols].notna().any(axis=1), created)

            if flags.get("has_company_description") and "company_description" in df.columns:
                self._add_feature(df, "has_company_description", df["company_description"].notna(), created)

            if flags.get("has_description") and "description" in df.columns:
                self._add_feature(df, "has_description", df["description"].notna(), created)

            if flags.get("listing_age_available") and {"listed_time", "expiry"} <= set(df.columns):
                self._add_feature(
                    df,
                    "listing_age_available",
                    df["listed_time"].notna() & df["expiry"].notna(),
                    created,
                )

            if flags.get("skill_count") and "skill_list" in df.columns:
                self._add_feature(df, "skill_count", df["skill_list"].apply(lambda v: list_length(v, delimiter)), created)

            if flags.get("benefit_count") and "benefit_list" in df.columns:
                self._add_feature(df, "benefit_count", df["benefit_list"].apply(lambda v: list_length(v, delimiter)), created)

            if flags.get("industry_count") and "industry_list" in df.columns:
                self._add_feature(df, "industry_count", df["industry_list"].apply(lambda v: list_length(v, delimiter)), created)

            if flags.get("speciality_count") and "speciality_list" in df.columns:
                self._add_feature(df, "speciality_count", df["speciality_list"].apply(lambda v: list_length(v, delimiter)), created)

            logging.info("[derived_features] END - Created column(s): %s", created)
            self._record_step(
                "derived_features",
                columns=created,
                rows_before=len(df),
                rows_after=len(df),
                details={},
                t0=t0,
            )
            return df, created
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    # 10. Duplicate removal
    # ------------------------------------------------------------------ #
    def _remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        logging.info("[duplicate_removal] START")
        t0 = time.time()
        try:
            if not self.config.drop_duplicates:
                return df, 0
            rows_before = len(df)
            df = df.drop_duplicates(subset=self.config.duplicate_subset).reset_index(drop=True)
            removed = rows_before - len(df)
            logging.info("[duplicate_removal] END - Removed %s duplicate row(s).", removed)
            self._record_step(
                "duplicate_removal",
                columns=self.config.duplicate_subset or [],
                rows_before=rows_before,
                rows_after=len(df),
                details={"removed": removed},
                t0=t0,
            )
            return df, removed
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    # 11. Final validation
    # ------------------------------------------------------------------ #
    def _final_validation(self, df: pd.DataFrame) -> None:
        try:
            validate_feature_store_schema(df, self.config)
            logging.info(
                "[final_validation] Passed -> %s rows, %s columns.",
                df.shape[0], df.shape[1],
            )
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    # Save outputs (latest + optional archive), mirrors Master Dataset Builder
    # ------------------------------------------------------------------ #
    def _save_outputs(
        self, df: pd.DataFrame, summary: FeatureEngineeringSummary
    ) -> FeatureEngineeringResult:
        try:
            cfg = self.config
            out_dir = ensure_dir(cfg.latest_output_dir)

            feature_store_path = out_dir / "feature_store.parquet"
            df.to_parquet(feature_store_path, index=False)
            logging.info("Feature Store successfully written to %s", feature_store_path)
            
            result = FeatureEngineeringResult(
                feature_store=df,
                feature_store_path=str(feature_store_path),
                summary=summary,
            )

            # --- Save Metadata ---
            if getattr(cfg, "save_metadata", True):
                metadata_path = out_dir / "feature_store_metadata.json"
                metadata = {
                    "dataset_id": summary.dataset_id,
                    "rows": summary.rows_after,
                    "columns": summary.columns_after,
                    "schema_hash": summary.schema_hash,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source_master_dataset": str(cfg.master_dataset_path),
                }
                save_json(metadata, metadata_path)
                result.metadata_path = str(metadata_path)

            # --- Save Report ---
            if getattr(cfg, "save_report", True):
                report_path = out_dir / "feature_store_report.json"
                report = FeatureEngineeringReport(
                    summary=summary,
                    steps=self.steps,
                    final_schema=schema_dict(df),
                )
                save_json(report.to_dict(), report_path)
                result.report_path = str(report_path)

            # --- Save Schema Fingerprint ---
            if getattr(cfg, "save_schema", True):
                schema_fingerprint_path = out_dir / "schema_fingerprint.json"
                save_json(
                    {"schema_hash": summary.schema_hash, "schema": schema_dict(df)},
                    schema_fingerprint_path,
                )
                result.schema_fingerprint_path = str(schema_fingerprint_path)

            logging.info(
                "Artifacts saved:\n- Feature Store\n- Metadata\n- Report\n- Schema"
            )

            # --- Save Archive Copy ---
            if cfg.save_archive_copy:
                archive_dir = ensure_dir(Path(cfg.archive_root_dir) / summary.dataset_id)

                dst_fs = archive_dir / "feature_store.parquet"
                shutil.copy2(feature_store_path, dst_fs)
                result.archive_feature_store_path = str(dst_fs)

                if getattr(cfg, "save_metadata", True) and result.metadata_path:
                    dst_md = archive_dir / "feature_store_metadata.json"
                    shutil.copy2(result.metadata_path, dst_md)
                    result.archive_metadata_path = str(dst_md)

                if getattr(cfg, "save_report", True) and result.report_path:
                    dst_rp = archive_dir / "feature_store_report.json"
                    shutil.copy2(result.report_path, dst_rp)
                    result.archive_report_path = str(dst_rp)

                if getattr(cfg, "save_schema", True) and result.schema_fingerprint_path:
                    dst_sc = archive_dir / "schema_fingerprint.json"
                    shutil.copy2(result.schema_fingerprint_path, dst_sc)
                    result.archive_schema_fingerprint_path = str(dst_sc)

                logging.info("Archive created:\n%s", archive_dir)
                
            else:
                logging.info("Archive copy disabled.")

            return result
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    # Internal bookkeeping
    # ------------------------------------------------------------------ #
    def _record_step(
        self,
        step_name: str,
        columns: List[str],
        rows_before: int,
        rows_after: int,
        details: Dict[str, Any],
        t0: float,
    ) -> None:
        self.steps.append(
            TransformationStepSummary(
                step_name=step_name,
                columns_affected=columns,
                rows_before=rows_before,
                rows_after=rows_after,
                details=details,
                execution_time_seconds=round(time.time() - t0, 4),
            )
        )
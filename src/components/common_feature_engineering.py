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
    """
    Model-agnostic Feature Store builder.

    Pipeline:

        master_dataset.parquet
                ↓
        CommonFeatureEngineering
                ↓
        feature_store.csv

    This stage performs generic cleaning and normalization.

    Model-specific transformations such as advanced salary feature
    construction and description-based skill extraction should happen
    downstream.
    """

    # ================================================================
    # INITIALIZATION
    # ================================================================

    def __init__(
        self,
        config: CommonFeatureEngineeringConfig,
    ):

        self.config = config

        self.steps: List[TransformationStepSummary] = []

        self.dataset_id = generate_dataset_id(prefix="fs")

        self.truthy_values = {
            str(v).strip().lower() if isinstance(v, str) else v
            for v in self.config.boolean_truthy_values
        }

        self.dtype_conversions: Dict[
            str,
            str,
        ] = {}

    # ================================================================
    # MAIN PIPELINE
    # ================================================================

    def run(
        self,
    ) -> FeatureEngineeringResult:

        try:

            start_time = time.time()

            self.steps.clear()
            self.dtype_conversions.clear()

            logging.info(SEP)
            logging.info("COMMON FEATURE ENGINEERING STARTED")
            logging.info(SEP)

            # --------------------------------------------------------
            # LOAD
            # --------------------------------------------------------

            df = self._load_master_dataset()

            rows_before, cols_before = df.shape

            null_before = int(df.isna().sum().sum())

            # --------------------------------------------------------
            # DETERMINE COLUMN GROUPS
            # --------------------------------------------------------

            col_groups = infer_column_types(
                df,
                self.config,
            )

            # --------------------------------------------------------
            # TRANSFORM
            # --------------------------------------------------------

            df = self._standardize_dtypes(
                df,
                col_groups,
            )

            df, identifiers_dropped = self._drop_identifiers(
                df,
                col_groups,
            )

            df = self._standardize_datetime(
                df,
                col_groups,
            )

            df, missing_filled = self._handle_missing_values(df)

            df = self._standardize_booleans(
                df,
                col_groups,
            )

            df = self._clean_categoricals(
                df,
                col_groups,
            )

            df = self._clean_text(
                df,
                col_groups,
            )

            df = self._standardize_list_columns(
                df,
                col_groups,
            )

            df, columns_created = self._create_derived_features(df)

            df, duplicates_removed = self._remove_duplicates(df)

            # --------------------------------------------------------
            # FINAL VALIDATION
            # --------------------------------------------------------

            self._final_validation(df)

            rows_after, cols_after = df.shape

            null_after = int(df.isna().sum().sum())

            schema_hash = compute_schema_hash(df)

            # --------------------------------------------------------
            # SUMMARY
            # --------------------------------------------------------

            summary = FeatureEngineeringSummary(
                dataset_id=self.dataset_id,
                rows_before=rows_before,
                rows_after=rows_after,
                rows_removed=(rows_before - rows_after),
                columns_before=cols_before,
                columns_after=cols_after,
                identifiers_dropped=(identifiers_dropped),
                columns_removed=(identifiers_dropped),
                columns_created=(columns_created),
                dtype_conversions=(self.dtype_conversions),
                missing_values_filled=(missing_filled),
                duplicates_removed=(duplicates_removed),
                null_values_before=(null_before),
                null_values_after=(null_after),
                schema_hash=schema_hash,
                execution_time_seconds=round(
                    time.time() - start_time,
                    4,
                ),
                integrity_passed=True,
            )

            # --------------------------------------------------------
            # SAVE
            # --------------------------------------------------------

            result = self._save_outputs(
                df,
                summary,
            )

            logging.info(
                "Build summary: %s",
                {
                    "dataset_id": summary.dataset_id,
                    "rows": f"{summary.rows_before} " f"-> {summary.rows_after}",
                    "columns": f"{summary.columns_before} "
                    f"-> {summary.columns_after}",
                    "columns_created": len(summary.columns_created),
                    "columns_removed": len(summary.columns_removed),
                    "nulls": f"{summary.null_values_before} "
                    f"-> {summary.null_values_after}",
                    "execution_time_seconds": summary.execution_time_seconds,
                    "schema_hash": summary.schema_hash[:16] + "...",
                },
            )

            logging.info(SEP)

            logging.info("COMMON FEATURE ENGINEERING COMPLETED")

            logging.info(SEP)

            return result

        except Exception as e:

            logging.exception("Common Feature Engineering failed.")

            raise CustomException(
                e,
                sys,
            )

    # ================================================================
    # LOAD MASTER DATASET
    # ================================================================

    def _load_master_dataset(
        self,
    ) -> pd.DataFrame:

        try:

            path = Path(self.config.master_dataset_path)

            if not path.exists():

                raise FileNotFoundError(f"Master dataset not found at: {path}")

            if path.suffix.lower() == ".parquet":

                df = pd.read_parquet(path)

            elif path.suffix.lower() == ".csv":

                df = pd.read_csv(path)

            else:

                raise ValueError(
                    f"Unsupported master dataset format: "
                    f"{path.suffix}. "
                    f"Expected .parquet or .csv."
                )

            if df.empty:

                if self.config.fail_on_empty_output:

                    raise ValueError("Master dataset contains 0 rows.")

            if df.columns.duplicated().any():

                duplicates = df.columns[df.columns.duplicated()].tolist()

                if self.config.fail_on_duplicate_column_names:

                    raise ValueError(
                        "Duplicate column names found "
                        f"in master dataset: {duplicates}"
                    )

                logging.warning(
                    "Duplicate column names detected: %s",
                    duplicates,
                )

            logging.info(
                "Loaded master dataset from '%s' " "-> %s rows, %s columns",
                path,
                df.shape[0],
                df.shape[1],
            )

            return df

        except Exception as e:

            raise CustomException(
                e,
                sys,
            )

    # ================================================================
    # DATA TYPE STANDARDIZATION
    # ================================================================

    def _standardize_dtypes(
        self,
        df: pd.DataFrame,
        col_groups: Dict[
            str,
            List[str],
        ],
    ) -> pd.DataFrame:

        logging.info("[dtype_standardization] START")

        t0 = time.time()

        try:

            numeric_cols = col_groups.get(
                "numeric",
                self.config.numeric_columns,
            )

            for col in numeric_cols:

                if col not in df.columns:
                    continue

                if not pd.api.types.is_numeric_dtype(df[col]):

                    old_dtype = str(df[col].dtype)

                    df[col] = pd.to_numeric(
                        df[col],
                        errors="coerce",
                    )

                    new_dtype = str(df[col].dtype)

                    self.dtype_conversions[col] = f"{old_dtype} -> " f"{new_dtype}"

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

            raise CustomException(
                e,
                sys,
            )

    # ================================================================
    # IDENTIFIER HANDLING
    # ================================================================

    def _drop_identifiers(
        self,
        df: pd.DataFrame,
        col_groups: Dict[
            str,
            List[str],
        ],
    ) -> Tuple[
        pd.DataFrame,
        List[str],
    ]:

        logging.info("[identifier_handling] START")

        t0 = time.time()

        try:

            identifiers = col_groups.get(
                "identifier",
                self.config.identifier_columns,
            )

            present = [c for c in identifiers if c in df.columns]

            missing = [c for c in identifiers if c not in df.columns]

            if missing:

                logging.warning(
                    "Configured identifier columns " "not found: %s",
                    missing,
                )

            df = df.drop(
                columns=present,
                errors="ignore",
            )

            self._record_step(
                "identifier_handling",
                columns=present,
                rows_before=len(df),
                rows_after=len(df),
                details={
                    "dropped": present,
                    "missing_from_source": missing,
                },
                t0=t0,
            )

            return df, present

        except Exception as e:

            raise CustomException(
                e,
                sys,
            )

    # ================================================================
    # DATETIME
    # ================================================================

    def _standardize_datetime(
        self,
        df: pd.DataFrame,
        col_groups: Dict[
            str,
            List[str],
        ],
    ) -> pd.DataFrame:

        logging.info("[datetime_standardization] START")

        t0 = time.time()

        try:

            converted = []

            datetime_cols = col_groups.get(
                "datetime",
                self.config.datetime_columns,
            )

            for col in datetime_cols:

                if col not in df.columns:
                    continue

                epoch_ms = col in self.config.epoch_ms_datetime_columns

                before_non_null = int(df[col].notna().sum())

                df[col] = to_datetime_safe(
                    df[col],
                    epoch_ms=epoch_ms,
                )

                after_non_null = int(df[col].notna().sum())

                converted.append(col)

                if after_non_null < before_non_null:

                    logging.warning(
                        "Datetime column '%s': " "%d values could not be parsed.",
                        col,
                        before_non_null - after_non_null,
                    )

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

            raise CustomException(
                e,
                sys,
            )

    # ================================================================
    # MISSING VALUES
    # ================================================================

    def _handle_missing_values(
        self,
        df: pd.DataFrame,
    ) -> Tuple[
        pd.DataFrame,
        int,
    ]:

        logging.info("[missing_value_handling] START")

        t0 = time.time()

        try:

            cfg = self.config

            total_filled = 0

            fill_log: Dict[
                str,
                Any,
            ] = {}

            for col in df.columns:

                n_missing = int(df[col].isna().sum())

                if n_missing == 0:
                    continue

                # ----------------------------------------------------
                # Lists are handled separately.
                # ----------------------------------------------------

                if col in cfg.list_columns:

                    logging.info(
                        "Skipping missing-value fill " "for list column '%s'.",
                        col,
                    )

                    continue

                strategy = cfg.missing_value_strategy.get(col)

                if strategy is None:

                    if pd.api.types.is_numeric_dtype(df[col]):

                        strategy = cfg.default_numeric_strategy

                    elif pd.api.types.is_bool_dtype(df[col]):

                        strategy = "leave"

                    elif col in cfg.text_columns:

                        strategy = cfg.default_text_strategy

                    else:

                        strategy = cfg.default_categorical_strategy

                if strategy == "leave":
                    continue

                fill_value = None

                # ----------------------------------------------------
                # MEDIAN
                # ----------------------------------------------------

                if strategy == "median":

                    if pd.api.types.is_numeric_dtype(df[col]):

                        fill_value = df[col].median()

                    else:

                        logging.warning(
                            "Cannot use median on " "non-numeric column '%s'.",
                            col,
                        )

                        continue

                # ----------------------------------------------------
                # MODE
                # ----------------------------------------------------

                elif strategy == "mode":

                    # Don't perform expensive mode calculations
                    # on high-cardinality text columns.

                    if col in cfg.text_columns and not pd.api.types.is_numeric_dtype(
                        df[col]
                    ):

                        logging.info(
                            "Skipping mode fill for " "text column '%s'.",
                            col,
                        )

                        continue

                    unique_count = df[col].nunique(dropna=True)

                    if unique_count > 100:

                        logging.warning(
                            "Skipping mode for "
                            "high-cardinality column "
                            "'%s' (%d unique values).",
                            col,
                            unique_count,
                        )

                        continue

                    mode_values = df[col].mode(dropna=True)

                    if not mode_values.empty:

                        fill_value = mode_values.iloc[0]

                # ----------------------------------------------------
                # CONSTANT
                # ----------------------------------------------------

                elif strategy == "constant":

                    if col in cfg.constant_fill_values:

                        fill_value = cfg.constant_fill_values[col]

                    elif pd.api.types.is_numeric_dtype(df[col]):

                        fill_value = cfg.default_constant_numeric_fill

                    else:

                        fill_value = cfg.default_constant_text_fill

                else:

                    logging.warning(
                        "Unknown missing-value " "strategy '%s' for '%s'.",
                        strategy,
                        col,
                    )

                    continue

                if fill_value is None:
                    continue

                df[col] = df[col].fillna(fill_value)

                n_missing_after = int(df[col].isna().sum())

                filled_count = n_missing - n_missing_after

                total_filled += filled_count

                fill_log[col] = {
                    "strategy": strategy,
                    "null_before": n_missing,
                    "null_after": n_missing_after,
                    "filled": filled_count,
                }

            self._record_step(
                "missing_value_handling",
                columns=list(fill_log.keys()),
                rows_before=len(df),
                rows_after=len(df),
                details=fill_log,
                t0=t0,
            )

            return (
                df,
                total_filled,
            )

        except Exception as e:

            raise CustomException(
                e,
                sys,
            )

    # ================================================================
    # BOOLEAN
    # ================================================================

    def _standardize_booleans(
        self,
        df: pd.DataFrame,
        col_groups: Dict[
            str,
            List[str],
        ],
    ) -> pd.DataFrame:

        logging.info("[boolean_standardization] START")

        t0 = time.time()

        try:

            converted = []

            boolean_cols = col_groups.get(
                "boolean",
                self.config.boolean_columns,
            )

            for col in boolean_cols:

                if col not in df.columns:
                    continue

                df[col] = (
                    df[col]
                    .apply(
                        lambda v: normalize_boolean(
                            v,
                            self.truthy_values,
                        )
                    )
                    .astype("boolean")
                )

                converted.append(col)

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

            raise CustomException(
                e,
                sys,
            )

    # ================================================================
    # CATEGORICAL CLEANUP
    # ================================================================

    def _clean_categoricals(
        self,
        df: pd.DataFrame,
        col_groups: Dict[
            str,
            List[str],
        ],
    ) -> pd.DataFrame:

        logging.info("[categorical_cleanup] START")

        t0 = time.time()

        try:

            cleaned = []

            categorical_cols = col_groups.get(
                "categorical",
                self.config.categorical_columns,
            )

            for col in categorical_cols:

                if col not in df.columns:
                    continue

                # Preserve NaN instead of converting it
                # into the string "nan".

                if pd.api.types.is_object_dtype(
                    df[col]
                ) or pd.api.types.is_string_dtype(df[col]):

                    df[col] = df[col].astype("string").str.strip().str.lower()

                    df[col] = df[col].replace(
                        {
                            "": pd.NA,
                            "nan": pd.NA,
                            "none": pd.NA,
                            "null": pd.NA,
                        }
                    )

                else:

                    df[col] = df[col].apply(normalize_category)

                df[col] = df[col].astype("category")

                cleaned.append(col)

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

            raise CustomException(
                e,
                sys,
            )

    # ================================================================
    # TEXT CLEANUP
    # ================================================================

    def _clean_text(
        self,
        df: pd.DataFrame,
        col_groups: Dict[
            str,
            List[str],
        ],
    ) -> pd.DataFrame:

        logging.info("[text_cleanup] START")

        t0 = time.time()

        try:

            cleaned = []

            text_cols = col_groups.get(
                "text",
                self.config.text_columns,
            )

            for col in text_cols:

                if col not in df.columns:
                    continue

                logging.info(
                    "Cleaning text column '%s'",
                    col,
                )

                # IMPORTANT:
                # Do not use astype(str) directly because
                # NaN becomes the string "nan".

                df[col] = df[col].fillna("").astype("string").map(normalize_text)

                cleaned.append(col)

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

            raise CustomException(
                e,
                sys,
            )

    # ================================================================
    # LIST STANDARDIZATION
    # ================================================================

    def _standardize_list_columns(
        self,
        df: pd.DataFrame,
        col_groups: Dict[
            str,
            List[str],
        ],
    ) -> pd.DataFrame:

        logging.info("[list_column_standardization] START")

        t0 = time.time()

        try:

            standardized = []

            list_cols = col_groups.get(
                "list",
                self.config.list_columns,
            )

            for col in list_cols:

                if col not in df.columns:
                    continue

                df[col] = df[col].apply(
                    lambda value: normalize_list_value(
                        value,
                        delimiter=(self.config.list_column_delimiter),
                    )
                )

                standardized.append(col)

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

            raise CustomException(
                e,
                sys,
            )

    # ================================================================
    # DERIVED FEATURES
    # ================================================================

    def _add_feature(
        self,
        df: pd.DataFrame,
        name: str,
        series: pd.Series,
        created: List[str],
    ) -> None:

        if pd.api.types.is_bool_dtype(series):

            df[name] = series.astype("boolean")

        else:

            df[name] = series

        created.append(name)

    def _create_derived_features(
        self,
        df: pd.DataFrame,
    ) -> Tuple[
        pd.DataFrame,
        List[str],
    ]:

        logging.info("[derived_features] START")

        t0 = time.time()

        try:

            flags = self.config.derived_feature_flags

            created: List[str] = []

            delimiter = self.config.list_column_delimiter

            # --------------------------------------------------------
            # Salary availability
            # --------------------------------------------------------

            if flags.get("salary_available") and any(
                c in df.columns
                for c in (
                    "min_salary",
                    "max_salary",
                    "med_salary",
                    "normalized_salary",
                )
            ):

                salary_cols = [
                    c
                    for c in (
                        "min_salary",
                        "max_salary",
                        "med_salary",
                        "normalized_salary",
                    )
                    if c in df.columns
                ]

                self._add_feature(
                    df,
                    "salary_available",
                    df[salary_cols].notna().any(axis=1),
                    created,
                )

            # --------------------------------------------------------
            # Description availability
            # --------------------------------------------------------

            if (
                flags.get("has_company_description")
                and "company_description" in df.columns
            ):

                self._add_feature(
                    df,
                    "has_company_description",
                    df["company_description"].fillna("").str.len() > 0,
                    created,
                )

            if flags.get("has_description") and "description" in df.columns:

                self._add_feature(
                    df,
                    "has_description",
                    df["description"].fillna("").str.len() > 0,
                    created,
                )

            # --------------------------------------------------------
            # Listing age availability
            # --------------------------------------------------------

            if flags.get("listing_age_available") and {
                "listed_time",
                "expiry",
            }.issubset(df.columns):

                self._add_feature(
                    df,
                    "listing_age_available",
                    (df["listed_time"].notna() & df["expiry"].notna()),
                    created,
                )

            # --------------------------------------------------------
            # Skill count
            # --------------------------------------------------------

            if flags.get("skill_count") and "skill_list" in df.columns:

                self._add_feature(
                    df,
                    "skill_count",
                    df["skill_list"].apply(
                        lambda value: list_length(
                            value,
                            delimiter,
                        )
                    ),
                    created,
                )

            # --------------------------------------------------------
            # Benefit count
            # --------------------------------------------------------

            if flags.get("benefit_count") and "benefit_list" in df.columns:

                self._add_feature(
                    df,
                    "benefit_count",
                    df["benefit_list"].apply(
                        lambda value: list_length(
                            value,
                            delimiter,
                        )
                    ),
                    created,
                )

            # --------------------------------------------------------
            # Industry count
            # --------------------------------------------------------

            if flags.get("industry_count") and "industry_list" in df.columns:

                self._add_feature(
                    df,
                    "industry_count",
                    df["industry_list"].apply(
                        lambda value: list_length(
                            value,
                            delimiter,
                        )
                    ),
                    created,
                )

            # --------------------------------------------------------
            # Specialty count
            # --------------------------------------------------------

            if flags.get("speciality_count") and "speciality_list" in df.columns:

                self._add_feature(
                    df,
                    "speciality_count",
                    df["speciality_list"].apply(
                        lambda value: list_length(
                            value,
                            delimiter,
                        )
                    ),
                    created,
                )

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

            raise CustomException(
                e,
                sys,
            )

    # ================================================================
    # DUPLICATES
    # ================================================================

    def _remove_duplicates(
        self,
        df: pd.DataFrame,
    ) -> Tuple[
        pd.DataFrame,
        int,
    ]:

        logging.info("[duplicate_removal] START")

        t0 = time.time()

        try:

            if not self.config.drop_duplicates:

                return df, 0

            rows_before = len(df)

            if self.config.duplicate_subset:

                df = df.drop_duplicates(subset=(self.config.duplicate_subset))

            else:

                df = df.drop_duplicates()

            df = df.reset_index(drop=True)

            removed = rows_before - len(df)

            self._record_step(
                "duplicate_removal",
                columns=(self.config.duplicate_subset or []),
                rows_before=rows_before,
                rows_after=len(df),
                details={"removed": removed},
                t0=t0,
            )

            return df, removed

        except Exception as e:

            raise CustomException(
                e,
                sys,
            )

    # ================================================================
    # FINAL VALIDATION
    # ================================================================

    def _final_validation(
        self,
        df: pd.DataFrame,
    ) -> None:

        try:

            if self.config.fail_on_empty_output and df.empty:

                raise ValueError(
                    "Common Feature Engineering " "produced an empty feature store."
                )

            duplicate_columns = df.columns[df.columns.duplicated()].tolist()

            if duplicate_columns:

                if self.config.fail_on_duplicate_column_names:

                    raise ValueError(
                        "Duplicate columns found "
                        f"after feature engineering: "
                        f"{duplicate_columns}"
                    )

            validate_feature_store_schema(
                df,
                self.config,
            )

            logging.info(
                "[final_validation] Passed -> " "%s rows, %s columns.",
                df.shape[0],
                df.shape[1],
            )

        except Exception as e:

            raise CustomException(
                e,
                sys,
            )

    # ================================================================
    # SAVE OUTPUTS
    # ================================================================

    def _save_outputs(
        self,
        df: pd.DataFrame,
        summary: FeatureEngineeringSummary,
    ) -> FeatureEngineeringResult:

        try:

            cfg = self.config

            out_dir = ensure_dir(cfg.latest_output_dir)

            # IMPORTANT:
            # This now comes from config.
            feature_store_path = out_dir / cfg.feature_store_filename

            # --------------------------------------------------------
            # CSV
            # --------------------------------------------------------

            df.to_csv(
                feature_store_path,
                index=False,
            )

            logging.info(
                "Feature Store written to %s",
                feature_store_path,
            )

            result = FeatureEngineeringResult(
                feature_store=df,
                feature_store_path=str(feature_store_path),
                summary=summary,
            )

            # --------------------------------------------------------
            # Metadata
            # --------------------------------------------------------

            if cfg.save_metadata:

                metadata_path = out_dir / "feature_store_metadata.json"

                metadata = {
                    "dataset_id": summary.dataset_id,
                    "rows": summary.rows_after,
                    "columns": summary.columns_after,
                    "schema_hash": summary.schema_hash,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source_master_dataset": str(cfg.master_dataset_path),
                    "feature_store_format": feature_store_path.suffix.lstrip("."),
                }

                save_json(
                    metadata,
                    metadata_path,
                )

                result.metadata_path = str(metadata_path)

            # --------------------------------------------------------
            # Report
            # --------------------------------------------------------

            if cfg.save_report:

                report_path = out_dir / "feature_store_report.json"

                report = FeatureEngineeringReport(
                    summary=summary,
                    steps=self.steps,
                    final_schema=schema_dict(df),
                )

                save_json(
                    report.to_dict(),
                    report_path,
                )

                result.report_path = str(report_path)

            # --------------------------------------------------------
            # Schema
            # --------------------------------------------------------

            if cfg.save_schema:

                schema_path = out_dir / "schema_fingerprint.json"

                save_json(
                    {
                        "schema_hash": summary.schema_hash,
                        "schema": schema_dict(df),
                    },
                    schema_path,
                )

                result.schema_fingerprint_path = str(schema_path)

            # --------------------------------------------------------
            # ARCHIVE
            # --------------------------------------------------------

            if cfg.save_archive_copy:

                archive_dir = ensure_dir(
                    Path(cfg.archive_root_dir) / summary.dataset_id
                )

                archive_feature_store = archive_dir / cfg.feature_store_filename

                shutil.copy2(
                    feature_store_path,
                    archive_feature_store,
                )

                result.archive_feature_store_path = str(archive_feature_store)

                if cfg.save_metadata and result.metadata_path:

                    destination = archive_dir / "feature_store_metadata.json"

                    shutil.copy2(
                        result.metadata_path,
                        destination,
                    )

                    result.archive_metadata_path = str(destination)

                if cfg.save_report and result.report_path:

                    destination = archive_dir / "feature_store_report.json"

                    shutil.copy2(
                        result.report_path,
                        destination,
                    )

                    result.archive_report_path = str(destination)

                if cfg.save_schema and result.schema_fingerprint_path:

                    destination = archive_dir / "schema_fingerprint.json"

                    shutil.copy2(
                        result.schema_fingerprint_path,
                        destination,
                    )

                    result.archive_schema_fingerprint_path = str(destination)

            logging.info("Feature Store artifacts successfully saved.")

            return result

        except Exception as e:

            raise CustomException(
                e,
                sys,
            )

    # ================================================================
    # STEP BOOKKEEPING
    # ================================================================

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
                execution_time_seconds=round(
                    time.time() - t0,
                    4,
                ),
            )
        )

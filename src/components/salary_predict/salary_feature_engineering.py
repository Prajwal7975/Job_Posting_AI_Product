from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from src.configs.salary_predict.salary_feature_engineering_config import (
    SalaryFeatureEngineeringConfig,
)

from src.entity.salary_predict.salary_feature_engineering_entity import (
    SalaryFeatureEngineeringSummary,
    SalaryFeatureEngineeringReport,
    SalaryFeatureEngineeringResult,
)


class SalaryFeatureEngineering:

    def __init__(
        self,
        config: Optional[SalaryFeatureEngineeringConfig] = None,
    ):
        self.config = config or SalaryFeatureEngineeringConfig()

    # ================================================================
    # PUBLIC ENTRY POINT
    # ================================================================

    def initiate_salary_feature_engineering(
        self,
    ) -> SalaryFeatureEngineeringResult:

        logging.info("=== Salary Feature Engineering started ===")

        start_time = time.time()

        try:

            cfg = self.config

            # --------------------------------------------------------
            # Unique dataset/run identifier
            # --------------------------------------------------------

            dataset_id = (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + "_"
                + uuid.uuid4().hex[:8]
            )

            # --------------------------------------------------------
            # Load common feature store
            # --------------------------------------------------------

            df = self._load_feature_store(cfg.feature_store_path)

            input_row_count = len(df)

            # --------------------------------------------------------
            # Validate required target columns
            # --------------------------------------------------------

            self._validate_required_columns(
                df,
                cfg.required_target_construction_columns,
            )

            # --------------------------------------------------------
            # Determine available predictors
            # --------------------------------------------------------

            (
                present_predictors,
                missing_predictors,
            ) = self._resolve_candidate_predictors(df)

            if missing_predictors:

                logging.warning(
                    "Candidate predictor columns missing from "
                    "feature store and will be skipped: %s",
                    missing_predictors,
                )

            # --------------------------------------------------------
            # Explicit predictor leakage guard
            # --------------------------------------------------------

            self._validate_predictor_configuration(present_predictors)

            # --------------------------------------------------------
            # Construct salary targets
            # --------------------------------------------------------

            (
                target_annual,
                target_log,
                target_source,
                funnel_counts,
            ) = self._construct_target(df)

            keep_mask = target_annual.notna()

            final_row_count = int(keep_mask.sum())

            if final_row_count == 0:

                raise ValueError(
                    "Salary feature engineering produced " "0 valid salary rows."
                )

            # --------------------------------------------------------
            # Deterministic numeric transformations
            # --------------------------------------------------------

            (
                log_employee,
                log_follower,
            ) = self._build_numeric_features(df)

            # --------------------------------------------------------
            # Group identifier for leakage-safe splitting
            # --------------------------------------------------------

            posting_group_id = self._build_posting_group_id(df)

            # --------------------------------------------------------
            # Construct clean salary modeling dataset
            # --------------------------------------------------------

            out = self._assemble_output_frame(
                df=df,
                keep_mask=keep_mask,
                target_annual=target_annual,
                target_log=target_log,
                target_source=target_source,
                present_predictors=present_predictors,
                log_employee=log_employee,
                log_follower=log_follower,
                posting_group_id=posting_group_id,
            )

            # --------------------------------------------------------
            # Post-condition validation
            # --------------------------------------------------------

            self._run_integrity_checks(
                out,
                present_predictors,
            )

            # --------------------------------------------------------
            # Dataset statistics
            # --------------------------------------------------------

            annual_stats = self._describe(out[cfg.target_annual_col])

            log_stats = self._describe(out[cfg.target_log_col])

            source_counts = (
                out[cfg.target_source_col].value_counts(dropna=False).to_dict()
            )

            # --------------------------------------------------------
            # Schema fingerprint
            # --------------------------------------------------------

            schema_hash = self._schema_fingerprint(out)

            execution_time = time.time() - start_time

            # --------------------------------------------------------
            # Final feature list
            # --------------------------------------------------------

            engineered_features = [
                cfg.log_company_employee_count_col,
                cfg.log_company_follower_count_col,
            ]

            feature_columns = list(
                dict.fromkeys(present_predictors + engineered_features)
            )

            actual_metadata_columns = [
                col for col in cfg.metadata_columns if col in out.columns
            ]

            metadata_columns = actual_metadata_columns + [
                cfg.posting_group_id_col,
                cfg.target_source_col,
            ]

            # --------------------------------------------------------
            # Summary entity
            # --------------------------------------------------------

            summary = SalaryFeatureEngineeringSummary(
                dataset_id=dataset_id,
                input_row_count=input_row_count,
                salary_candidate_count=(funnel_counts["salary_candidate_count"]),
                valid_usd_salary_count=(funnel_counts["valid_usd_salary_count"]),
                rows_removed_missing_salary=(
                    funnel_counts["rows_removed_missing_salary"]
                ),
                rows_removed_currency=(funnel_counts["rows_removed_currency"]),
                rows_removed_unsupported_pay_period=(
                    funnel_counts["rows_removed_unsupported_pay_period"]
                ),
                rows_removed_out_of_bounds=(
                    funnel_counts["rows_removed_out_of_bounds"]
                ),
                final_row_count=final_row_count,
                target_coverage_pct=round(
                    (final_row_count / input_row_count) * 100,
                    4,
                ),
                target_source_counts={str(k): int(v) for k, v in source_counts.items()},
                annual_salary_stats=annual_stats,
                log_salary_stats=log_stats,
                feature_columns=feature_columns,
                metadata_columns=metadata_columns,
                leakage_columns_removed=(cfg.leakage_columns.copy()),
                schema_hash=schema_hash,
                execution_time_seconds=round(
                    execution_time,
                    3,
                ),
                integrity_passed=True,
            )

            # --------------------------------------------------------
            # Persist artifacts
            # --------------------------------------------------------

            result = self._write_artifacts(
                out=out,
                summary=summary,
                dataset_id=dataset_id,
            )

            logging.info(
                "=== Salary Feature Engineering completed: "
                "%d/%d rows retained (%.2f%% coverage) "
                "in %.2fs ===",
                final_row_count,
                input_row_count,
                summary.target_coverage_pct,
                execution_time,
            )

            return result

        except Exception as e:

            logging.exception("Salary Feature Engineering failed.")

            raise CustomException(
                e,
                sys,
            )

    # ================================================================
    # DATA LOADING
    # ================================================================

    def _load_feature_store(
        self,
        path: Path,
    ) -> pd.DataFrame:

        if not path.exists():

            raise FileNotFoundError(f"Common feature store not found: {path}")

        logging.info(
            "Loading feature store from %s",
            path,
        )

        # Common Feature Engineering currently writes CSV.
        # Keep this loader defensive so a future parquet Feature Store
        # can also be consumed without changing this component again.
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        else:
            raise ValueError(
                f"Unsupported feature store format: {path.suffix}. "
                "Expected .csv or .parquet."
            )

        logging.info(
            "Feature store loaded -> shape=%s",
            df.shape,
        )

        if df.empty:

            raise ValueError("Feature store contains 0 rows.")

        return df

    # ================================================================
    # REQUIRED COLUMN VALIDATION
    # ================================================================

    @staticmethod
    def _validate_required_columns(
        df: pd.DataFrame,
        required: List[str],
    ) -> None:

        missing = [col for col in required if col not in df.columns]

        if missing:

            raise ValueError(
                "Required target-construction columns " f"missing: {missing}"
            )

    # ================================================================
    # PREDICTOR RESOLUTION
    # ================================================================

    def _resolve_candidate_predictors(
        self,
        df: pd.DataFrame,
    ) -> Tuple[List[str], List[str]]:

        configured = self.config.candidate_predictor_columns

        present = [col for col in configured if col in df.columns]

        missing = [col for col in configured if col not in df.columns]

        return present, missing

    # ================================================================
    # CONFIGURATION LEAKAGE CHECK
    # ================================================================

    def _validate_predictor_configuration(
        self,
        predictors: List[str],
    ) -> None:

        leakage = set(predictors) & set(self.config.leakage_columns)

        if leakage:

            raise ValueError(
                "Invalid SalaryFeatureEngineeringConfig: "
                "candidate predictors contain leakage columns: "
                f"{sorted(leakage)}"
            )

    # ================================================================
    # TARGET CONSTRUCTION
    # ================================================================

    def _construct_target(
        self,
        df: pd.DataFrame,
    ) -> Tuple[
        pd.Series,
        pd.Series,
        pd.Series,
        dict,
    ]:

        cfg = self.config

        # ================================================================
        # 1. CONVERT SALARY COLUMNS TO NUMERIC
        # ================================================================

        min_salary = pd.to_numeric(
            df[cfg.min_salary_col],
            errors="coerce",
        )

        med_salary = pd.to_numeric(
            df[cfg.med_salary_col],
            errors="coerce",
        )

        max_salary = pd.to_numeric(
            df[cfg.max_salary_col],
            errors="coerce",
        )

        # ================================================================
        # 2. NORMALIZE PAY PERIOD
        #
        # IMPORTANT:
        # This MUST happen before the hourly validation.
        # ================================================================

        pay_period = df[cfg.pay_period_col].astype("string").str.strip().str.lower()

        # ================================================================
        # 3. NORMALIZE CURRENCY
        # ================================================================

        currency = df[cfg.currency_col].astype("string").str.strip().str.lower()

        # ================================================================
        # 4. DETERMINE AVAILABLE SALARY INFORMATION
        # ================================================================

        min_present = min_salary.notna()

        med_present = med_salary.notna()

        max_present = max_salary.notna()

        any_salary_present = min_present | med_present | max_present

        # ================================================================
        # 5. RANGE MIDPOINT
        #
        # Prefer min/max when both are valid.
        # ================================================================

        has_range = (
            min_present
            & max_present
            & (min_salary > 0)
            & (max_salary > 0)
            & (min_salary <= max_salary)
        )

        # ================================================================
        # 6. MEDIAN FALLBACK
        # ================================================================

        has_median = med_present & (med_salary > 0) & ~has_range

        # ================================================================
        # 7. RAW SALARY
        # ================================================================

        raw_salary = pd.Series(
            np.nan,
            index=df.index,
            dtype="float64",
        )

        # Range midpoint

        raw_salary.loc[has_range] = (
            min_salary.loc[has_range] + max_salary.loc[has_range]
        ) / 2.0

        # Median fallback

        raw_salary.loc[has_median] = med_salary.loc[has_median]

        # ================================================================
        # 8. TARGET SOURCE
        # ================================================================

        target_source = pd.Series(
            pd.NA,
            index=df.index,
            dtype="string",
        )

        target_source.loc[has_range] = cfg.source_range_midpoint_label

        target_source.loc[has_median] = cfg.source_median_salary_label

        # ================================================================
        # 9. SALARY CANDIDATES
        # ================================================================

        salary_candidate_mask = raw_salary.notna()

        salary_candidate_count = int(salary_candidate_mask.sum())

        rows_removed_missing_salary = int((~any_salary_present).sum())

        # ================================================================
        # 10. HOURLY SANITY CHECK
        #
        # IMPORTANT:
        #
        # hourly multiplier = 2080
        #
        # We are NOT changing it.
        #
        # We only reject obviously corrupted hourly values.
        # ================================================================

        suspicious_hourly_mask = pay_period.eq("hourly") & raw_salary.gt(
            float(cfg.hourly_max_rate)
        )

        suspicious_hourly_count = int(suspicious_hourly_mask.sum())

        if suspicious_hourly_count:

            logging.warning(
                "Rejecting %d suspicious hourly salary " "records above $%.2f/hour.",
                suspicious_hourly_count,
                float(cfg.hourly_max_rate),
            )

            raw_salary.loc[suspicious_hourly_mask] = np.nan

            target_source.loc[suspicious_hourly_mask] = pd.NA

        # Recalculate after removing bad hourly records.

        salary_candidate_mask = raw_salary.notna()

        # ================================================================
        # 11. CURRENCY VALIDATION
        # ================================================================

        is_allowed_currency = currency.eq(cfg.allowed_currency.lower()).fillna(False)

        valid_currency_mask = salary_candidate_mask & is_allowed_currency

        valid_usd_salary_count = int(valid_currency_mask.sum())

        rows_removed_currency = int(
            (salary_candidate_mask & ~is_allowed_currency).sum()
        )

        # ================================================================
        # 12. PAY PERIOD MULTIPLIER
        # ================================================================

        multiplier = pay_period.map(cfg.pay_period_multipliers).astype("float64")

        valid_period_mask = valid_currency_mask & multiplier.notna()

        rows_removed_unsupported_pay_period = int(
            (valid_currency_mask & multiplier.isna()).sum()
        )

        # ================================================================
        # 13. ANNUALIZE
        # ================================================================

        target_annual = pd.Series(
            np.nan,
            index=df.index,
            dtype="float64",
        )

        target_annual.loc[valid_period_mask] = (
            raw_salary.loc[valid_period_mask] * multiplier.loc[valid_period_mask]
        )

        # ================================================================
        # 14. ANNUAL SALARY BOUNDS
        # ================================================================

        in_bounds = target_annual.ge(cfg.min_annual_salary) & target_annual.le(
            cfg.max_annual_salary
        )

        rows_removed_out_of_bounds = int((valid_period_mask & ~in_bounds).sum())

        final_mask = valid_period_mask & in_bounds

        # Remove invalid targets.

        target_annual = target_annual.where(final_mask)

        target_source = target_source.where(final_mask)

        # ================================================================
        # 15. LOG TARGET
        # ================================================================

        target_log = np.log1p(target_annual)

        # ================================================================
        # 16. FUNNEL
        # ================================================================

        funnel_counts = {
            "salary_candidate_count": salary_candidate_count,
            "valid_usd_salary_count": valid_usd_salary_count,
            "rows_removed_missing_salary": rows_removed_missing_salary,
            "rows_removed_currency": rows_removed_currency,
            "rows_removed_unsupported_pay_period": rows_removed_unsupported_pay_period,
            "rows_removed_out_of_bounds": rows_removed_out_of_bounds,
            "suspicious_hourly_count": suspicious_hourly_count,
        }

        return (
            target_annual,
            target_log,
            target_source,
            funnel_counts,
        )

        # ================================================================

    # NUMERIC FEATURES
    # ================================================================

    def _build_numeric_features(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series]:

        cfg = self.config

        def safe_log1p(
            column: str,
        ) -> pd.Series:

            if column not in df.columns:

                logging.warning(
                    "%s not found. Generated log feature " "will contain NaN.",
                    column,
                )

                return pd.Series(
                    np.nan,
                    index=df.index,
                    dtype="float64",
                )

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            negative_mask = values < 0

            negative_count = int(negative_mask.sum())

            if negative_count:

                logging.warning(
                    "%s contains %d negative values; " "masking them as NaN.",
                    column,
                    negative_count,
                )

            values = values.mask(
                negative_mask,
                np.nan,
            )

            return np.log1p(values)

        log_employee = safe_log1p(cfg.company_employee_count_col)

        log_follower = safe_log1p(cfg.company_follower_count_col)

        return (log_employee, log_follower)

    # ================================================================
    # POSTING GROUP ID
    # ================================================================

    def _build_posting_group_id(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        cfg = self.config

        def normalize(
            series: pd.Series,
        ) -> pd.Series:

            return (
                series.astype("string")
                .fillna("unknown")
                .str.lower()
                .str.strip()
                .str.replace(
                    r"\s+",
                    " ",
                    regex=True,
                )
            )

        parts = []

        for column in cfg.posting_group_source_columns:

            if column in df.columns:

                parts.append(normalize(df[column]))

            else:

                logging.warning(
                    "posting_group_id source column " "%s missing; using 'unknown'.",
                    column,
                )

                parts.append(
                    pd.Series(
                        "unknown",
                        index=df.index,
                        dtype="string",
                    )
                )

        if not parts:

            raise ValueError("No posting-group source columns configured.")

        combined = parts[0]

        for part in parts[1:]:

            combined = combined + "||" + part

        return combined.map(
            lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
        )

    # ================================================================
    # OUTPUT DATASET CONSTRUCTION
    # ================================================================

    def _assemble_output_frame(
        self,
        df: pd.DataFrame,
        keep_mask: pd.Series,
        target_annual: pd.Series,
        target_log: pd.Series,
        target_source: pd.Series,
        present_predictors: List[str],
        log_employee: pd.Series,
        log_follower: pd.Series,
        posting_group_id: pd.Series,
    ) -> pd.DataFrame:

        cfg = self.config

        selected_index = df.index[keep_mask]

        out = pd.DataFrame(index=selected_index)

        # ------------------------------------------------------------
        # Targets
        # ------------------------------------------------------------

        out[cfg.target_annual_col] = target_annual.loc[selected_index]

        out[cfg.target_log_col] = target_log.loc[selected_index]

        out[cfg.target_source_col] = target_source.loc[selected_index]

        # ------------------------------------------------------------
        # Approved predictors only
        # ------------------------------------------------------------

        for column in present_predictors:

            out[column] = df.loc[
                selected_index,
                column,
            ]

        # ------------------------------------------------------------
        # Engineered numeric predictors
        # ------------------------------------------------------------

        out[cfg.log_company_employee_count_col] = log_employee.loc[selected_index]

        out[cfg.log_company_follower_count_col] = log_follower.loc[selected_index]

        # ------------------------------------------------------------
        # Split metadata
        # ------------------------------------------------------------

        out[cfg.posting_group_id_col] = posting_group_id.loc[selected_index]

        # ------------------------------------------------------------
        # Metadata
        # ------------------------------------------------------------

        for column in cfg.metadata_columns:

            if column in df.columns:

                # Prevent duplicate insertion if a column was
                # accidentally configured in both lists.
                if column in out.columns:

                    logging.warning(
                        "Metadata column %s already exists "
                        "in output; skipping duplicate insertion.",
                        column,
                    )

                    continue

                out[column] = df.loc[
                    selected_index,
                    column,
                ]

            else:

                logging.warning(
                    "Metadata column %s missing; skipping.",
                    column,
                )

        return out.reset_index(drop=True)

    # ================================================================
    # INTEGRITY CHECKS
    # ================================================================

    def _run_integrity_checks(
        self,
        out: pd.DataFrame,
        present_predictors: List[str],
    ) -> None:

        cfg = self.config

        if out.empty:

            raise ValueError(
                "Integrity check failed: " "output dataset contains 0 rows."
            )

        # ------------------------------------------------------------
        # Target null checks
        # ------------------------------------------------------------

        if out[cfg.target_annual_col].isna().any():

            raise ValueError(
                "Integrity check failed: " "annual salary target contains NaN."
            )

        if out[cfg.target_log_col].isna().any():

            raise ValueError(
                "Integrity check failed: " "log salary target contains NaN."
            )

        # ------------------------------------------------------------
        # Salary bounds
        # ------------------------------------------------------------

        valid_bounds = out[cfg.target_annual_col].between(
            cfg.min_annual_salary,
            cfg.max_annual_salary,
            inclusive="both",
        )

        if not valid_bounds.all():

            raise ValueError(
                "Integrity check failed: " "salary targets outside configured bounds."
            )

        # ------------------------------------------------------------
        # Duplicate columns
        # ------------------------------------------------------------

        duplicate_columns = out.columns[out.columns.duplicated()].tolist()

        if duplicate_columns:

            raise ValueError(
                "Integrity check failed: "
                "duplicate columns detected: "
                f"{duplicate_columns}"
            )

        # ------------------------------------------------------------
        # Infinity detection
        # ------------------------------------------------------------

        numeric_columns = out.select_dtypes(include=[np.number]).columns

        if len(numeric_columns):

            numeric_values = out[numeric_columns].to_numpy(dtype="float64")

            if np.isinf(numeric_values).any():

                raise ValueError(
                    "Integrity check failed: " "numeric features contain infinity."
                )

        # ------------------------------------------------------------
        # Predictor leakage
        # ------------------------------------------------------------

        model_predictors = set(present_predictors) | {
            cfg.log_company_employee_count_col,
            cfg.log_company_follower_count_col,
        }

        leaked = model_predictors & set(cfg.leakage_columns)

        if leaked:

            raise ValueError(
                "Integrity check failed: "
                "leakage columns found in predictors: "
                f"{sorted(leaked)}"
            )

        # ------------------------------------------------------------
        # Posting group ID
        # ------------------------------------------------------------

        if out[cfg.posting_group_id_col].isna().any():

            raise ValueError(
                "Integrity check failed: " "posting_group_id contains nulls."
            )

        # ------------------------------------------------------------
        # Log-target mathematical consistency
        # ------------------------------------------------------------

        expected_log = np.log1p(out[cfg.target_annual_col])

        if not np.allclose(
            out[cfg.target_log_col].to_numpy(dtype="float64"),
            expected_log.to_numpy(dtype="float64"),
            rtol=1e-9,
            atol=1e-9,
        ):

            raise ValueError(
                "Integrity check failed: "
                "target_log_salary != "
                "log1p(target_annual_salary)."
            )

        # ------------------------------------------------------------
        # Accidental index column
        # ------------------------------------------------------------

        if "index" in out.columns:

            raise ValueError(
                "Integrity check failed: " "accidental index column persisted."
            )

        logging.info("Salary feature engineering " "integrity checks passed.")

    # ================================================================
    # STATISTICS
    # ================================================================

    @staticmethod
    def _describe(
        series: pd.Series,
    ) -> dict:

        return {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
        }

    # ================================================================
    # SCHEMA FINGERPRINT
    # ================================================================

    @staticmethod
    def _schema_fingerprint(
        df: pd.DataFrame,
    ) -> str:

        schema = {column: str(dtype) for column, dtype in df.dtypes.items()}

        schema_repr = json.dumps(
            schema,
            sort_keys=True,
        )

        return hashlib.sha256(schema_repr.encode("utf-8")).hexdigest()

    # ================================================================
    # ARTIFACT WRITING
    # ================================================================

    def _write_artifacts(
        self,
        out: pd.DataFrame,
        summary: SalaryFeatureEngineeringSummary,
        dataset_id: str,
    ) -> SalaryFeatureEngineeringResult:

        cfg = self.config

        created_at = datetime.now(timezone.utc).isoformat()

        # ------------------------------------------------------------
        # Latest directory
        # ------------------------------------------------------------

        cfg.latest_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataset_path = cfg.latest_dir / "salary_modeling_dataset.parquet"

        metadata_path = cfg.latest_dir / "salary_feature_metadata.json"

        report_path = cfg.latest_dir / "salary_feature_report.json"

        schema_fp_path = cfg.latest_dir / "schema_fingerprint.json"

        # ------------------------------------------------------------
        # Atomic-ish parquet replacement
        # ------------------------------------------------------------

        temp_dataset_path = cfg.latest_dir / "salary_modeling_dataset.tmp.parquet"

        out.to_parquet(
            temp_dataset_path,
            index=False,
        )

        temp_dataset_path.replace(dataset_path)

        # ------------------------------------------------------------
        # Report
        # ------------------------------------------------------------

        report = SalaryFeatureEngineeringReport(
            dataset_id=dataset_id,
            created_at=created_at,
            input_feature_store_path=str(cfg.feature_store_path),
            output_dataset_path=str(dataset_path),
            summary=summary,
        )

        # ------------------------------------------------------------
        # Metadata
        # ------------------------------------------------------------

        metadata = {
            "dataset_id": dataset_id,
            "created_at": created_at,
            "row_count": len(out),
            "columns": list(out.columns),
            "feature_columns": summary.feature_columns,
            "metadata_columns": summary.metadata_columns,
            "target_columns": [
                cfg.target_annual_col,
                cfg.target_log_col,
            ],
        }

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        report_path.write_text(
            report.to_json(),
            encoding="utf-8",
        )

        schema_fp_path.write_text(
            json.dumps(
                {
                    "schema_hash": summary.schema_hash,
                    "dataset_id": dataset_id,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # ------------------------------------------------------------
        # Optional archive
        # ------------------------------------------------------------

        archive_dataset_path = None
        archive_metadata_path = None
        archive_report_path = None
        archive_schema_fp_path = None

        if cfg.keep_archive_snapshots:

            archive_run_dir = cfg.archive_dir / dataset_id

            archive_run_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            archive_dataset_path = archive_run_dir / dataset_path.name

            archive_metadata_path = archive_run_dir / metadata_path.name

            archive_report_path = archive_run_dir / report_path.name

            archive_schema_fp_path = archive_run_dir / schema_fp_path.name

            shutil.copy2(
                dataset_path,
                archive_dataset_path,
            )

            shutil.copy2(
                metadata_path,
                archive_metadata_path,
            )

            shutil.copy2(
                report_path,
                archive_report_path,
            )

            shutil.copy2(
                schema_fp_path,
                archive_schema_fp_path,
            )

        logging.info(
            "Salary modeling dataset written -> %s (%d rows)",
            dataset_path,
            len(out),
        )

        logging.info(
            "Salary Feature Engineering summary: %s",
            summary.as_log_dict(),
        )

        # ------------------------------------------------------------
        # Result entity
        # ------------------------------------------------------------

        return SalaryFeatureEngineeringResult(
            salary_modeling_dataset_path=(dataset_path),
            metadata_path=(metadata_path),
            report_path=(report_path),
            schema_fingerprint_path=(schema_fp_path),
            summary=summary,
            archive_dataset_path=(archive_dataset_path),
            archive_metadata_path=(archive_metadata_path),
            archive_report_path=(archive_report_path),
            archive_schema_fingerprint_path=(archive_schema_fp_path),
            dataframe=out,
        )

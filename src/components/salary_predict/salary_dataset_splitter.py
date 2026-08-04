"""
src/components/salary_dataset_splitter.py

Salary Dataset Splitting stage.

Input:  artifacts/salary_feature_store/latest/salary_modeling_dataset.parquet
Output: artifacts/salary_dataset_splits/latest/{train,validation,test}.parquet
        + split_report.json, split_metadata.json, schema_fingerprint.json,
          salary_split_state.json

Responsibility: ONLY produce leakage-safe, reproducible train/validation/test
partitions using group-aware splitting on `posting_group_id`. This component
does not fit any encoder/scaler/vectorizer and does not train any model —
see salary_dataset_splitter_config.py's spec docstring for the full list of
operations that are explicitly out of scope here.

Idempotency: if the input dataset content, the split configuration, AND the
persisted train/validation/test artifacts are all unchanged/intact since the
last run, this component REUSES the existing artifacts instead of
recomputing them. Any change to input, config, or a manual edit/corruption
of a persisted split file invalidates the cache and triggers a fresh
EXECUTED split.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.logger import logging
from src.exception import CustomException
from src.configs.salary_predict.salary_dataset_splitter_config import (
    SalaryDatasetSplitterConfig,
)
from src.entity.salary_predict.salary_dataset_splitter_entity import (
    SalaryDatasetSplitSummary,
    SalaryDatasetSplitReport,
    SalaryDatasetSplitResult,
)

_ROW_ID_COL = "_original_row_id"


class SalaryDatasetSplitter:
    def __init__(self, config: Optional[SalaryDatasetSplitterConfig] = None):
        self.config = config or SalaryDatasetSplitterConfig()

    # ==================================================================
    # Public orchestration
    # ==================================================================
    def initiate_dataset_splitting(
        self, force_rebuild: bool = False
    ) -> SalaryDatasetSplitResult:
        """
        Run (or reuse) the group-aware train/validation/test split.

        Parameters
        ----------
        force_rebuild:
            When True, bypasses the reuse check entirely and always produces
            a fresh split, regardless of whether the input dataset, split
            configuration, and persisted artifact fingerprints are all
            unchanged. Callers should always go through this parameter to
            invalidate the cache — never by touching
            `config.split_state_path` or any other artifact directly.
        """
        logging.info("=" * 70)
        logging.info("SALARY DATASET SPLITTING STARTED")
        logging.info("=" * 70)
        start_time = time.time()

        try:
            cfg = self.config

            input_fingerprint = self._calculate_input_fingerprint(
                cfg.input_dataset_path
            )
            dataset_id = self._resolve_dataset_id(input_fingerprint)
            split_signature = self._calculate_split_signature(input_fingerprint)

            if force_rebuild:
                logging.info(
                    "Force rebuild enabled. Salary Dataset Splitting will run."
                )
                reusable, existing_report = False, None
            else:
                reusable, existing_report = self._can_reuse_existing_split(
                    split_signature
                )

            if reusable and existing_report is not None:
                logging.info("-" * 70)
                logging.info("Input dataset unchanged.")
                logging.info("Split configuration unchanged.")
                logging.info("Existing artifacts valid.")
                logging.info("")
                logging.info("Salary Dataset Splitting : REUSED")
                logging.info("-" * 70)
                return self._build_result_from_existing(existing_report)

            df = self._load_dataset(cfg.input_dataset_path)
            logging.info(f"Loading: {cfg.input_dataset_path}")
            logging.info(f"Input rows: {len(df):,}")

            self._validate_input(df)

            df = df.reset_index(drop=True)
            df[_ROW_ID_COL] = np.arange(len(df))

            unique_groups = df[cfg.group_col].nunique()
            logging.info(f"Unique posting groups: {unique_groups:,}")
            logging.info(f"Split strategy: {cfg.split_strategy}")
            logging.info(
                f"Requested: Train {cfg.train_size*100:.0f}%  "
                f"Validation {cfg.validation_size*100:.0f}%  Test {cfg.test_size*100:.0f}%"
            )

            train_df, validation_df, test_df = self._create_group_split(df)

            self._validate_row_integrity(df, train_df, validation_df, test_df)
            overlap_counts = self._validate_group_integrity(
                train_df, validation_df, test_df
            )
            self._validate_row_id_integrity(train_df, validation_df, test_df)
            schema_hash = self._validate_schema_consistency(
                train_df, validation_df, test_df
            )

            # drop internal row id — never persisted, never a predictor
            train_df = train_df.drop(columns=[_ROW_ID_COL])
            validation_df = validation_df.drop(columns=[_ROW_ID_COL])
            test_df = test_df.drop(columns=[_ROW_ID_COL])

            actual_ratios = self._calculate_ratio_audit(
                df, train_df, validation_df, test_df
            )

            logging.info(
                f"Actual: Train {actual_ratios['train']*100:.2f}%  "
                f"Validation {actual_ratios['validation']*100:.2f}%  "
                f"Test {actual_ratios['test']*100:.2f}%"
            )
            logging.info(
                f"Group leakage: Train\u2194Validation = {overlap_counts['train_validation']}  "
                f"Train\u2194Test = {overlap_counts['train_test']}  "
                f"Validation\u2194Test = {overlap_counts['validation_test']}"
            )

            execution_time = time.time() - start_time

            summary = self._build_summary(
                dataset_id=dataset_id,
                df=df,
                train_df=train_df,
                validation_df=validation_df,
                test_df=test_df,
                actual_ratios=actual_ratios,
                overlap_counts=overlap_counts,
                schema_hash=schema_hash,
                execution_time=execution_time,
            )

            logging.info("Integrity checks: PASSED")

            self._write_split_artifacts(train_df, validation_df, test_df)
            report = self._write_report(
                dataset_id=dataset_id, summary=summary, status="EXECUTED"
            )
            self._write_metadata(
                dataset_id=dataset_id,
                summary=summary,
                input_fingerprint=input_fingerprint,
                split_signature=split_signature,
                status="EXECUTED",
            )
            self._write_split_state(dataset_id, input_fingerprint, split_signature)

            archive_paths = {}
            if cfg.keep_archive_snapshots:
                archive_paths = self._archive_artifacts(dataset_id)

            logging.info("Artifacts written...")
            logging.info("=" * 70)
            logging.info("SALARY DATASET SPLITTING COMPLETED")
            logging.info("=" * 70)

            return SalaryDatasetSplitResult(
                train_dataset_path=cfg.train_output_path,
                validation_dataset_path=cfg.validation_output_path,
                test_dataset_path=cfg.test_output_path,
                report_path=cfg.split_report_path,
                metadata_path=cfg.split_metadata_path,
                schema_fingerprint_path=cfg.schema_fingerprint_path,
                summary=summary,
                archive_train_path=archive_paths.get("train"),
                archive_validation_path=archive_paths.get("validation"),
                archive_test_path=archive_paths.get("test"),
                archive_report_path=archive_paths.get("report"),
                archive_metadata_path=archive_paths.get("metadata"),
                archive_schema_fingerprint_path=archive_paths.get("schema_fingerprint"),
                status="EXECUTED",
            )

        except Exception as e:
            logging.exception(f"Salary Dataset Splitting failed: {e}")
            raise CustomException(e, sys)

    # ==================================================================
    # Loading / fingerprinting
    # ==================================================================
    def _load_dataset(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Salary modeling dataset not found at: {path}")
        df = pd.read_parquet(path)
        if df.empty:
            raise ValueError(f"Salary modeling dataset at {path} is empty (0 rows).")
        return df

    def _calculate_input_fingerprint(self, path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"Salary modeling dataset not found at: {path}")
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _resolve_dataset_id(self, input_fingerprint: str) -> str:
        """
        Prefer propagating the upstream salary feature engineering
        dataset_id (lineage: feature store -> salary features -> split).
        Falls back to a deterministic id derived from the input content
        fingerprint if the upstream metadata file is unavailable.
        """
        cfg = self.config
        meta_path = cfg.salary_feature_metadata_path
        if meta_path.exists():
            try:
                upstream_meta = json.loads(meta_path.read_text())
                upstream_id = upstream_meta.get("dataset_id")
                if upstream_id:
                    return upstream_id
                logging.warning(
                    f"{meta_path} exists but has no 'dataset_id' key; using fallback id."
                )
            except (json.JSONDecodeError, OSError) as e:
                logging.warning(f"Could not read {meta_path} ({e}); using fallback id.")
        else:
            logging.warning(
                f"Upstream metadata not found at {meta_path}; using fallback id."
            )

        return f"salsplit_{input_fingerprint[:16]}"

    def _calculate_split_signature(self, input_fingerprint: str) -> str:
        cfg = self.config

        config_snapshot = {
            "input_fingerprint": input_fingerprint,
            "split_strategy": cfg.split_strategy,
            "group_col": cfg.group_col,
            "target_annual_col": cfg.target_annual_col,
            "target_log_col": cfg.target_log_col,
            "train_size": cfg.train_size,
            "validation_size": cfg.validation_size,
            "test_size": cfg.test_size,
            "random_state": cfg.random_state,
            "max_ratio_deviation": cfg.max_ratio_deviation,
            "min_unique_groups": cfg.min_unique_groups,
        }

        payload = json.dumps(
            config_snapshot, sort_keys=True, separators=(",", ":"), default=str
        )

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ==================================================================
    # Reuse / idempotency
    # ==================================================================

    def _can_reuse_existing_split(
        self, split_signature: str
    ) -> Tuple[bool, Optional[SalaryDatasetSplitReport]]:
        """
        Reuse existing split artifacts only when:

        1. Previous split state exists.
        2. Input + configuration split signature is unchanged.
        3. All expected artifacts exist.
        4. Train/validation/test fingerprints match the fingerprints
           recorded when those artifacts were originally created.
        5. Existing split report can be loaded successfully.

        This protects against accidental modification or corruption of
        previously generated split artifacts.
        """
        cfg = self.config

        # --------------------------------------------------------------
        # 1. Previous state must exist
        # --------------------------------------------------------------
        if not cfg.split_state_path.exists():
            logging.info("No previous split state found. Fresh split required.")
            return False, None

        try:
            state = json.loads(cfg.split_state_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logging.warning(
                f"Could not read previous split state ({e}). Fresh split required."
            )
            return False, None

        # --------------------------------------------------------------
        # 2. Input/configuration signature must match
        # --------------------------------------------------------------
        if state.get("split_signature") != split_signature:
            logging.info(
                "Input dataset or split configuration changed. " "Fresh split required."
            )
            return False, None

        # --------------------------------------------------------------
        # 3. Required artifacts must exist
        # --------------------------------------------------------------
        required_paths = [
            cfg.train_output_path,
            cfg.validation_output_path,
            cfg.test_output_path,
            cfg.split_report_path,
            cfg.split_metadata_path,
            cfg.schema_fingerprint_path,
        ]

        missing_paths = [str(path) for path in required_paths if not path.exists()]

        if missing_paths:
            logging.warning(
                f"Existing split is incomplete. Missing artifacts: {missing_paths}"
            )
            return False, None

        # --------------------------------------------------------------
        # 4. Stored output fingerprints must exist
        # --------------------------------------------------------------
        stored_fingerprints = state.get("output_fingerprints")

        if not isinstance(stored_fingerprints, dict):
            logging.info(
                "Previous split state does not contain output fingerprints. "
                "Fresh split required."
            )
            return False, None

        required_fingerprint_keys = {"train", "validation", "test"}

        if not required_fingerprint_keys.issubset(stored_fingerprints):
            logging.warning(
                "Previous split state contains incomplete output fingerprints. "
                "Fresh split required."
            )
            return False, None

        # --------------------------------------------------------------
        # 5. Recalculate fingerprints of persisted artifacts
        # --------------------------------------------------------------
        try:
            current_fingerprints = {
                "train": self._calculate_input_fingerprint(cfg.train_output_path),
                "validation": self._calculate_input_fingerprint(
                    cfg.validation_output_path
                ),
                "test": self._calculate_input_fingerprint(cfg.test_output_path),
            }

        except (OSError, FileNotFoundError) as e:
            logging.warning(
                f"Could not fingerprint existing split artifacts ({e}). "
                "Fresh split required."
            )
            return False, None

        # --------------------------------------------------------------
        # 6. Compare stored vs current fingerprints
        # --------------------------------------------------------------
        for split_name in ("train", "validation", "test"):

            stored_hash = stored_fingerprints.get(split_name)
            current_hash = current_fingerprints[split_name]

            if stored_hash != current_hash:
                logging.warning(
                    f"{split_name.capitalize()} artifact fingerprint mismatch. "
                    "The artifact may have been modified or corrupted. "
                    "Fresh split required."
                )
                return False, None

        # --------------------------------------------------------------
        # 7. Existing report must still be readable
        # --------------------------------------------------------------
        try:
            report = SalaryDatasetSplitReport.from_json(
                cfg.split_report_path.read_text()
            )

        except (json.JSONDecodeError, OSError, TypeError, KeyError) as e:
            logging.warning(
                f"Existing split report is invalid ({e}). " "Fresh split required."
            )
            return False, None

        logging.info("Existing train/validation/test artifact fingerprints verified.")

        return True, report

    def _build_result_from_existing(
        self, report: SalaryDatasetSplitReport
    ) -> SalaryDatasetSplitResult:
        cfg = self.config
        return SalaryDatasetSplitResult(
            train_dataset_path=cfg.train_output_path,
            validation_dataset_path=cfg.validation_output_path,
            test_dataset_path=cfg.test_output_path,
            report_path=cfg.split_report_path,
            metadata_path=cfg.split_metadata_path,
            schema_fingerprint_path=cfg.schema_fingerprint_path,
            summary=report.summary,
            status="REUSED",
        )

    # ==================================================================
    # Input validation
    # ==================================================================
    def _validate_input(self, df: pd.DataFrame) -> None:
        cfg = self.config

        missing = [c for c in cfg.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Required columns missing from input dataset: {missing}")

        if df[cfg.group_col].isna().any():
            raise ValueError(
                f"{cfg.group_col} contains null values; cannot split safely."
            )

        for target_col in (cfg.target_annual_col, cfg.target_log_col):
            if df[target_col].isna().any():
                raise ValueError(f"{target_col} contains null values.")
            values = df[target_col].to_numpy(dtype="float64")
            if not np.isfinite(values).all():
                raise ValueError(
                    f"{target_col} contains non-finite values (inf/-inf/NaN)."
                )

        if (df[cfg.target_annual_col] <= 0).any():
            raise ValueError(f"{cfg.target_annual_col} contains values <= 0.")

        unique_groups = df[cfg.group_col].nunique()
        if unique_groups < cfg.min_unique_groups:
            raise ValueError(
                f"Only {unique_groups} unique {cfg.group_col} values found; "
                f"need at least {cfg.min_unique_groups} for a meaningful 3-way split."
            )

    # ==================================================================
    # Splitting
    # ==================================================================
    def _create_group_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cfg = self.config

        # Split 1: carve out TEST from everything else.
        gss_test = GroupShuffleSplit(
            n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state
        )
        train_val_idx, test_idx = next(gss_test.split(df, groups=df[cfg.group_col]))

        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # Split 2: carve VALIDATION out of the remaining train+validation pool,
        # using the ratio relative to that pool (not the original dataset).
        gss_val = GroupShuffleSplit(
            n_splits=1,
            test_size=cfg.relative_validation_size,
            random_state=cfg.random_state,
        )
        train_idx, val_idx = next(
            gss_val.split(train_val_df, groups=train_val_df[cfg.group_col])
        )

        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        validation_df = train_val_df.iloc[val_idx].reset_index(drop=True)

        return train_df, validation_df, test_df

    # ==================================================================
    # Integrity validation
    # ==================================================================
    def _validate_row_integrity(
        self,
        df: pd.DataFrame,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        total = len(train_df) + len(validation_df) + len(test_df)
        if total != len(df):
            raise ValueError(
                f"Row conservation check failed: train+validation+test={total} "
                f"but input has {len(df)} rows."
            )
        if len(train_df) == 0:
            raise ValueError("Train split is empty.")
        if len(validation_df) == 0:
            raise ValueError("Validation split is empty.")
        if len(test_df) == 0:
            raise ValueError("Test split is empty.")

    def _validate_group_integrity(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict[str, int]:
        cfg = self.config
        train_groups = set(train_df[cfg.group_col])
        val_groups = set(validation_df[cfg.group_col])
        test_groups = set(test_df[cfg.group_col])

        train_val_overlap = len(train_groups & val_groups)
        train_test_overlap = len(train_groups & test_groups)
        val_test_overlap = len(val_groups & test_groups)

        if train_val_overlap:
            raise ValueError(
                f"Group leakage detected: {train_val_overlap} posting_group_id values "
                "appear in BOTH train and validation."
            )
        if train_test_overlap:
            raise ValueError(
                f"Group leakage detected: {train_test_overlap} posting_group_id values "
                "appear in BOTH train and test."
            )
        if val_test_overlap:
            raise ValueError(
                f"Group leakage detected: {val_test_overlap} posting_group_id values "
                "appear in BOTH validation and test."
            )

        return {
            "train_validation": train_val_overlap,
            "train_test": train_test_overlap,
            "validation_test": val_test_overlap,
        }

    def _validate_row_id_integrity(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        train_ids = set(train_df[_ROW_ID_COL])
        val_ids = set(validation_df[_ROW_ID_COL])
        test_ids = set(test_df[_ROW_ID_COL])

        if train_ids & val_ids:
            raise ValueError(
                "Duplicate row assignment detected between train and validation."
            )
        if train_ids & test_ids:
            raise ValueError(
                "Duplicate row assignment detected between train and test."
            )
        if val_ids & test_ids:
            raise ValueError(
                "Duplicate row assignment detected between validation and test."
            )

    def _validate_schema_consistency(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> str:
        train_cols = train_df.columns.tolist()
        if (
            validation_df.columns.tolist() != train_cols
            or test_df.columns.tolist() != train_cols
        ):
            raise ValueError(
                "Schema mismatch: train, validation, and test partitions do not have "
                "identical columns."
            )

        train_dtypes = train_df.dtypes.astype(str).to_dict()
        for name, part_df in (("validation", validation_df), ("test", test_df)):
            part_dtypes = part_df.dtypes.astype(str).to_dict()
            mismatched = {
                col: (train_dtypes[col], part_dtypes[col])
                for col in train_cols
                if train_dtypes[col] != part_dtypes[col]
            }
            if mismatched:
                raise ValueError(
                    f"Schema mismatch between train and {name} for columns: {mismatched}"
                )

        # schema fingerprint excludes the internal row id, which is dropped
        # before this is used for the persisted artifact fingerprint anyway
        schema_repr = json.dumps(
            {
                col: dtype
                for col, dtype in sorted(train_dtypes.items())
                if col != _ROW_ID_COL
            },
            sort_keys=True,
        )
        return hashlib.sha256(schema_repr.encode("utf-8")).hexdigest()

    def _calculate_ratio_audit(
        self,
        df: pd.DataFrame,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict[str, float]:
        cfg = self.config
        n = len(df)
        actual = {
            "train": len(train_df) / n,
            "validation": len(validation_df) / n,
            "test": len(test_df) / n,
        }
        requested = {
            "train": cfg.train_size,
            "validation": cfg.validation_size,
            "test": cfg.test_size,
        }
        for split_name in actual:
            deviation = abs(actual[split_name] - requested[split_name])
            if deviation > cfg.max_ratio_deviation:
                logging.warning(
                    f"{split_name} ratio deviates from requested by {deviation*100:.2f} "
                    f"percentage points (requested={requested[split_name]*100:.1f}%, "
                    f"actual={actual[split_name]*100:.2f}%). This is expected behavior "
                    "of group-aware splitting and is not treated as a failure."
                )
        return actual

    # ==================================================================
    # Stats / summary
    # ==================================================================
    @staticmethod
    def _target_stats(series: pd.Series) -> Dict[str, float]:
        return {
            "count": int(series.count()),
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
        }

    def _build_summary(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
        actual_ratios: Dict[str, float],
        overlap_counts: Dict[str, int],
        schema_hash: str,
        execution_time: float,
    ) -> SalaryDatasetSplitSummary:

        cfg = self.config

        # -------------------------------------------------------------
        # Ratio deviations
        # -------------------------------------------------------------
        train_ratio_deviation = abs(actual_ratios["train"] - cfg.train_size)

        validation_ratio_deviation = abs(
            actual_ratios["validation"] - cfg.validation_size
        )

        test_ratio_deviation = abs(actual_ratios["test"] - cfg.test_size)

        # -------------------------------------------------------------
        # Target null counts
        # -------------------------------------------------------------
        target_cols = [
            cfg.target_annual_col,
            cfg.target_log_col,
        ]

        train_target_null_count = int(train_df[target_cols].isna().any(axis=1).sum())

        validation_target_null_count = int(
            validation_df[target_cols].isna().any(axis=1).sum()
        )

        test_target_null_count = int(test_df[target_cols].isna().any(axis=1).sum())

        # -------------------------------------------------------------
        # Build summary
        # -------------------------------------------------------------
        return SalaryDatasetSplitSummary(
            dataset_id=dataset_id,
            # Input
            input_row_count=len(df),
            input_group_count=int(df[cfg.group_col].nunique()),
            # Row counts
            train_row_count=len(train_df),
            validation_row_count=len(validation_df),
            test_row_count=len(test_df),
            # Group counts
            train_group_count=int(train_df[cfg.group_col].nunique()),
            validation_group_count=int(validation_df[cfg.group_col].nunique()),
            test_group_count=int(test_df[cfg.group_col].nunique()),
            # Actual ratios
            train_ratio=round(actual_ratios["train"], 6),
            validation_ratio=round(actual_ratios["validation"], 6),
            test_ratio=round(actual_ratios["test"], 6),
            # Requested ratios
            requested_train_ratio=cfg.train_size,
            requested_validation_ratio=cfg.validation_size,
            requested_test_ratio=cfg.test_size,
            # Ratio deviations
            train_ratio_deviation=round(train_ratio_deviation, 6),
            validation_ratio_deviation=round(validation_ratio_deviation, 6),
            test_ratio_deviation=round(test_ratio_deviation, 6),
            # Group leakage
            train_validation_group_overlap=(overlap_counts["train_validation"]),
            train_test_group_overlap=(overlap_counts["train_test"]),
            validation_test_group_overlap=(overlap_counts["validation_test"]),
            # Target null counts
            train_target_null_count=train_target_null_count,
            validation_target_null_count=validation_target_null_count,
            test_target_null_count=test_target_null_count,
            # Annual target statistics
            target_annual_train_stats=self._target_stats(
                train_df[cfg.target_annual_col]
            ),
            target_annual_validation_stats=self._target_stats(
                validation_df[cfg.target_annual_col]
            ),
            target_annual_test_stats=self._target_stats(test_df[cfg.target_annual_col]),
            # Log target statistics
            target_log_train_stats=self._target_stats(train_df[cfg.target_log_col]),
            target_log_validation_stats=self._target_stats(
                validation_df[cfg.target_log_col]
            ),
            target_log_test_stats=self._target_stats(test_df[cfg.target_log_col]),
            # Reproducibility / integrity
            random_state=cfg.random_state,
            schema_hash=schema_hash,
            execution_time_seconds=round(execution_time, 3),
            integrity_passed=True,
        )

    # ==================================================================
    # Writing artifacts
    # ==================================================================
    def _write_split_artifacts(
        self, train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        cfg = self.config
        cfg.latest_dir.mkdir(parents=True, exist_ok=True)

        # write to temp names first, then rename, so a mid-write crash never
        # leaves a partially-written file at the canonical "latest" path
        for target_path, part_df in (
            (cfg.train_output_path, train_df),
            (cfg.validation_output_path, validation_df),
            (cfg.test_output_path, test_df),
        ):
            tmp_path = target_path.with_suffix(".parquet.tmp")
            part_df.to_parquet(tmp_path, index=False)
            tmp_path.replace(target_path)

    def _write_report(
        self, dataset_id: str, summary: SalaryDatasetSplitSummary, status: str
    ) -> SalaryDatasetSplitReport:
        cfg = self.config
        report = SalaryDatasetSplitReport(
            dataset_id=dataset_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            input_dataset_path=str(cfg.input_dataset_path),
            train_output_path=str(cfg.train_output_path),
            validation_output_path=str(cfg.validation_output_path),
            test_output_path=str(cfg.test_output_path),
            split_strategy=cfg.split_strategy,
            group_column=cfg.group_col,
            status=status,
            summary=summary,
        )
        cfg.split_report_path.write_text(report.to_json())

        schema_fp_payload = {
            "schema_hash": summary.schema_hash,
            "dataset_id": dataset_id,
        }
        cfg.schema_fingerprint_path.write_text(json.dumps(schema_fp_payload, indent=2))

        return report

    def _write_metadata(
        self,
        dataset_id: str,
        summary: SalaryDatasetSplitSummary,
        input_fingerprint: str,
        split_signature: str,
        status: str,
    ) -> None:
        cfg = self.config
        metadata = {
            "dataset_id": dataset_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_dataset_path": str(cfg.input_dataset_path),
            "input_fingerprint": input_fingerprint,
            "split_signature": split_signature,
            "split_strategy": cfg.split_strategy,
            "group_column": cfg.group_col,
            "requested_ratios": {
                "train": cfg.train_size,
                "validation": cfg.validation_size,
                "test": cfg.test_size,
            },
            "actual_ratios": {
                "train": summary.train_ratio,
                "validation": summary.validation_ratio,
                "test": summary.test_ratio,
            },
            "random_state": cfg.random_state,
            "input_row_count": summary.input_row_count,
            "input_group_count": summary.input_group_count,
            "train_row_count": summary.train_row_count,
            "train_group_count": summary.train_group_count,
            "validation_row_count": summary.validation_row_count,
            "validation_group_count": summary.validation_group_count,
            "test_row_count": summary.test_row_count,
            "test_group_count": summary.test_group_count,
            "schema_hash": summary.schema_hash,
            "integrity_status": "PASSED" if summary.integrity_passed else "FAILED",
            "status": status,
        }
        cfg.split_metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    def _write_split_state(
        self,
        dataset_id: str,
        input_fingerprint: str,
        split_signature: str,
    ) -> None:
        """
        Persist the state required for safe idempotent reuse.

        The state records:

        - fingerprint of the input salary dataset
        - split configuration signature
        - fingerprints of generated train/validation/test artifacts
        - configuration snapshot

        On future runs these values are validated before existing artifacts
        are reused.
        """
        cfg = self.config

        # --------------------------------------------------------------
        # Fingerprint generated split artifacts
        # --------------------------------------------------------------

        output_fingerprints = {
            "train": self._calculate_input_fingerprint(cfg.train_output_path),
            "validation": self._calculate_input_fingerprint(cfg.validation_output_path),
            "test": self._calculate_input_fingerprint(cfg.test_output_path),
        }

        state = {
            "dataset_id": dataset_id,
            # Upstream input identity
            "input_fingerprint": input_fingerprint,
            # Input + split configuration identity
            "split_signature": split_signature,
            # Persisted artifact identities
            "output_fingerprints": output_fingerprints,
            # Configuration used to create these artifacts
            "config_snapshot": {
                "split_strategy": cfg.split_strategy,
                "group_col": cfg.group_col,
                "target_annual_col": cfg.target_annual_col,
                "target_log_col": cfg.target_log_col,
                "train_size": cfg.train_size,
                "validation_size": cfg.validation_size,
                "test_size": cfg.test_size,
                "random_state": cfg.random_state,
                "max_ratio_deviation": cfg.max_ratio_deviation,
                "min_unique_groups": cfg.min_unique_groups,
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        cfg.split_state_path.write_text(
            json.dumps(
                state,
                indent=2,
                default=str,
            )
        )

        logging.info(
            "Split state written with train/validation/test artifact fingerprints."
        )

    def _archive_artifacts(self, dataset_id: str) -> Dict[str, Path]:
        cfg = self.config
        archive_run_dir = cfg.archive_dir / dataset_id
        archive_run_dir.mkdir(parents=True, exist_ok=True)

        mapping = {
            "train": (cfg.train_output_path, archive_run_dir / "train.parquet"),
            "validation": (
                cfg.validation_output_path,
                archive_run_dir / "validation.parquet",
            ),
            "test": (cfg.test_output_path, archive_run_dir / "test.parquet"),
            "report": (cfg.split_report_path, archive_run_dir / "split_report.json"),
            "metadata": (
                cfg.split_metadata_path,
                archive_run_dir / "split_metadata.json",
            ),
            "schema_fingerprint": (
                cfg.schema_fingerprint_path,
                archive_run_dir / "schema_fingerprint.json",
            ),
        }

        archived_paths = {}
        for key, (src, dst) in mapping.items():
            shutil.copy2(src, dst)
            archived_paths[key] = dst

        return archived_paths


if __name__ == "__main__":
    splitter = SalaryDatasetSplitter()
    result = splitter.initiate_dataset_splitting()
    print(f"Status: {result.status}")
    print(f"Train: {result.train_dataset_path} ({result.summary.train_row_count} rows)")
    print(
        f"Validation: {result.validation_dataset_path} ({result.summary.validation_row_count} rows)"
    )
    print(f"Test: {result.test_dataset_path} ({result.summary.test_row_count} rows)")

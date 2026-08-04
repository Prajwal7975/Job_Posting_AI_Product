"""
src/entity/salary_dataset_splitter_entity.py

Typed data containers for the Salary Dataset Splitting stage.

These entities contain no dataset-processing logic. They provide
structured contracts between the salary dataset splitter, pipeline
orchestration, artifact reporting, and downstream model-training stages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional


@dataclass
class SalaryDatasetSplitSummary:
    """
    Statistical and integrity summary of a salary dataset split run.
    """

    dataset_id: str

    # --------------------------------------------------------------
    # Input
    # --------------------------------------------------------------

    input_row_count: int
    input_group_count: int

    # --------------------------------------------------------------
    # Partition row counts
    # --------------------------------------------------------------

    train_row_count: int
    validation_row_count: int
    test_row_count: int

    # --------------------------------------------------------------
    # Partition group counts
    # --------------------------------------------------------------

    train_group_count: int
    validation_group_count: int
    test_group_count: int

    # --------------------------------------------------------------
    # Actual split ratios
    # --------------------------------------------------------------

    train_ratio: float
    validation_ratio: float
    test_ratio: float

    # --------------------------------------------------------------
    # Requested split ratios
    # --------------------------------------------------------------

    requested_train_ratio: float
    requested_validation_ratio: float
    requested_test_ratio: float

    # --------------------------------------------------------------
    # Ratio deviations
    # --------------------------------------------------------------

    train_ratio_deviation: float
    validation_ratio_deviation: float
    test_ratio_deviation: float

    # --------------------------------------------------------------
    # Group leakage checks
    # --------------------------------------------------------------

    train_validation_group_overlap: int
    train_test_group_overlap: int
    validation_test_group_overlap: int

    # --------------------------------------------------------------
    # Target integrity
    # --------------------------------------------------------------

    train_target_null_count: int
    validation_target_null_count: int
    test_target_null_count: int

    # --------------------------------------------------------------
    # Annual salary target statistics
    # --------------------------------------------------------------

    target_annual_train_stats: Dict[str, float]
    target_annual_validation_stats: Dict[str, float]
    target_annual_test_stats: Dict[str, float]

    # --------------------------------------------------------------
    # Log salary target statistics
    # --------------------------------------------------------------

    target_log_train_stats: Dict[str, float]
    target_log_validation_stats: Dict[str, float]
    target_log_test_stats: Dict[str, float]

    # --------------------------------------------------------------
    # Reproducibility / integrity
    # --------------------------------------------------------------

    random_state: int
    schema_hash: str
    execution_time_seconds: float
    integrity_passed: bool

    def as_log_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(
            asdict(self),
            indent=2,
            default=str,
        )

    @staticmethod
    def from_dict(d: dict) -> "SalaryDatasetSplitSummary":
        return SalaryDatasetSplitSummary(**d)


@dataclass
class SalaryDatasetSplitReport:
    """
    Persistent report describing the run that created the split artifacts.
    """

    dataset_id: str
    created_at: str

    input_dataset_path: str

    train_output_path: str
    validation_output_path: str
    test_output_path: str

    split_strategy: str
    group_column: str
    status: str

    summary: SalaryDatasetSplitSummary

    def as_log_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(
            self.as_log_dict(),
            indent=2,
            default=str,
        )

    @staticmethod
    def from_json(text: str) -> "SalaryDatasetSplitReport":
        raw = json.loads(text)

        raw["summary"] = SalaryDatasetSplitSummary.from_dict(
            raw["summary"]
        )

        return SalaryDatasetSplitReport(**raw)

@dataclass
class SalaryDatasetSplitResult:
    """
    Runtime return object exposed to pipeline.py and downstream stages.
    """

    train_dataset_path: Path
    validation_dataset_path: Path
    test_dataset_path: Path

    report_path: Path
    metadata_path: Path
    schema_fingerprint_path: Path

    summary: SalaryDatasetSplitSummary

    archive_train_path: Optional[Path] = None
    archive_validation_path: Optional[Path] = None
    archive_test_path: Optional[Path] = None

    archive_report_path: Optional[Path] = None
    archive_metadata_path: Optional[Path] = None
    archive_schema_fingerprint_path: Optional[Path] = None

    status: str = "EXECUTED"
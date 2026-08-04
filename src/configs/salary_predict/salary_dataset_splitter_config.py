"""
src/configs/salary_dataset_splitter_config.py

Configuration for the Salary Dataset Splitting component.

Defines paths, split proportions, reproducibility settings, integrity
thresholds, and artifact names for the salary dataset splitting stage.

Production code should pass base_artifacts_dir from the project's
PipelineConfig.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SalaryDatasetSplitterConfig:

    # ==============================================================
    # Base paths
    # ==============================================================

    base_artifacts_dir: Path = Path("artifacts")

    input_subdir: str = "salary_feature_store/latest"
    input_filename: str = "salary_modeling_dataset.parquet"
    input_metadata_filename: str = "salary_feature_metadata.json"

    output_subdir: str = "salary_dataset_splits"

    keep_archive_snapshots: bool = True

    # ==============================================================
    # Output artifact filenames
    # ==============================================================

    train_filename: str = "train.parquet"
    validation_filename: str = "validation.parquet"
    test_filename: str = "test.parquet"

    split_report_filename: str = "split_report.json"
    split_metadata_filename: str = "split_metadata.json"
    schema_fingerprint_filename: str = "schema_fingerprint.json"
    split_state_filename: str = "salary_split_state.json"

    # ==============================================================
    # Split columns
    # ==============================================================

    group_col: str = "posting_group_id"

    target_annual_col: str = "target_annual_salary"
    target_log_col: str = "target_log_salary"

    # ==============================================================
    # Split configuration
    # ==============================================================

    train_size: float = 0.70
    validation_size: float = 0.10
    test_size: float = 0.20

    random_state: int = 42

    split_strategy: str = "GroupShuffleSplit"

    # ==============================================================
    # Integrity / quality configuration
    # ==============================================================

    max_ratio_deviation: float = 0.05

    min_unique_groups: int = 10

    # ==============================================================
    # Configuration validation
    # ==============================================================

    def __post_init__(self) -> None:

        ratios = {
            "train_size": self.train_size,
            "validation_size": self.validation_size,
            "test_size": self.test_size,
        }

        for name, value in ratios.items():

            if not 0 < value < 1:
                raise ValueError(
                    f"{name} must be between 0 and 1 "
                    f"(exclusive), got {value}"
                )

        total = (
            self.train_size
            + self.validation_size
            + self.test_size
        )

        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(
                "train_size + validation_size + test_size "
                f"must equal 1.0, got {total}"
            )

        if not 0 <= self.max_ratio_deviation <= 1:
            raise ValueError(
                "max_ratio_deviation must be between 0 and 1, "
                f"got {self.max_ratio_deviation}"
            )

        if self.min_unique_groups < 3:
            raise ValueError(
                "min_unique_groups must be at least 3 "
                "for train/validation/test splitting."
            )

    # ==============================================================
    # Derived split mathematics
    # ==============================================================

    @property
    def relative_validation_size(self) -> float:
        """
        Validation fraction used during the second GroupShuffleSplit.

        Example:

        Requested final split:
            Train      = 70%
            Validation = 10%
            Test       = 20%

        After removing Test:
            Train + Validation = 80%

        Therefore:

            0.10 / 0.80 = 0.125

        The second split therefore allocates 12.5% of the remaining
        dataset to validation.
        """

        remaining_size = self.train_size + self.validation_size

        return self.validation_size / remaining_size

    # ==============================================================
    # Input paths
    # ==============================================================

    @property
    def input_dataset_path(self) -> Path:
        return (
            self.base_artifacts_dir
            / self.input_subdir
            / self.input_filename
        )

    @property
    def salary_feature_metadata_path(self) -> Path:
        return (
            self.base_artifacts_dir
            / self.input_subdir
            / self.input_metadata_filename
        )

    # ==============================================================
    # Output directories
    # ==============================================================

    @property
    def output_dir(self) -> Path:
        return self.base_artifacts_dir / self.output_subdir

    @property
    def latest_dir(self) -> Path:
        return self.output_dir / "latest"

    @property
    def archive_dir(self) -> Path:
        return self.output_dir / "archive"

    # ==============================================================
    # Output artifact paths
    # ==============================================================

    @property
    def train_output_path(self) -> Path:
        return self.latest_dir / self.train_filename

    @property
    def validation_output_path(self) -> Path:
        return self.latest_dir / self.validation_filename

    @property
    def test_output_path(self) -> Path:
        return self.latest_dir / self.test_filename

    @property
    def split_report_path(self) -> Path:
        return self.latest_dir / self.split_report_filename

    @property
    def split_metadata_path(self) -> Path:
        return self.latest_dir / self.split_metadata_filename

    @property
    def schema_fingerprint_path(self) -> Path:
        return self.latest_dir / self.schema_fingerprint_filename

    @property
    def split_state_path(self) -> Path:
        return self.latest_dir / self.split_state_filename

    # ==============================================================
    # Required schema
    # ==============================================================

    @property
    def required_columns(self) -> list[str]:
        return [
            self.group_col,
            self.target_annual_col,
            self.target_log_col,
        ]
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class SalaryFeatureEngineeringConfig:

    # ================================================================
    # INPUT FEATURE STORE
    # ================================================================

    base_artifacts_dir: Path = Path("artifacts")

    feature_store_subdir: str = (
        "feature_store/latest"
    )

    feature_store_filename: str = (
        "feature_store.csv"
    )

    # ================================================================
    # OUTPUT
    # ================================================================

    output_subdir: str = (
        "salary_feature_store"
    )

    keep_archive_snapshots: bool = True

    # ================================================================
    # SALARY SOURCE COLUMNS
    # ================================================================

    min_salary_col: str = "min_salary"

    med_salary_col: str = "med_salary"

    max_salary_col: str = "max_salary"

    pay_period_col: str = "pay_period"

    currency_col: str = "currency"

    # ================================================================
    # CURRENCY
    # ================================================================

    allowed_currency: str = "usd"

    # ================================================================
    # TARGET BOUNDS
    # ================================================================

    min_annual_salary: float = 10_000.0

    max_annual_salary: float = 1_000_000.0

    # ================================================================
    # PAY PERIOD MULTIPLIERS
    # ================================================================

    pay_period_multipliers: Dict[str, float] = field(
        default_factory=lambda: {
            "yearly": 1.0,
            "annual": 1.0,
            "monthly": 12.0,
            "biweekly": 26.0,
            "weekly": 52.0,
            "hourly": 2080.0,
        }
    )

    # ================================================================
    # HOURLY DATA QUALITY GUARD
    #
    # This is NOT the hourly multiplier.
    #
    # 2080 is still correct.
    #
    # This prevents corrupted source records such as:
    #
    # $240,000/hour
    #
    # from becoming:
    #
    # $499,200,000/year
    # ================================================================

    hourly_max_rate: float = 300.0

    # ================================================================
    # TARGET OUTPUT COLUMNS
    # ================================================================

    target_annual_col: str = (
        "target_annual_salary"
    )

    target_log_col: str = (
        "target_log_salary"
    )

    target_source_col: str = (
        "salary_target_source"
    )

    source_range_midpoint_label: str = (
        "RANGE_MIDPOINT"
    )

    source_median_salary_label: str = (
        "MEDIAN_SALARY"
    )

    # ================================================================
    # REQUIRED INPUT COLUMNS
    # ================================================================

    required_target_construction_columns: List[str] = field(
        default_factory=lambda: [
            "min_salary",
            "med_salary",
            "max_salary",
            "pay_period",
            "currency",
        ]
    )

    # ================================================================
    # MODEL PREDICTORS
    # ================================================================

    candidate_predictor_columns: List[str] = field(
        default_factory=lambda: [
            "title",
            "formatted_experience_level",
            "formatted_work_type",
            "company_state",
            "company_country",
            "company_size",
            "company_employee_count",
            "company_follower_count",
            "top_industry",
            "top_skill",
            "skill_list",
            "skill_count",
        ]
    )

    # ================================================================
    # METADATA
    # ================================================================

    metadata_columns: List[str] = field(
        default_factory=lambda: [
            "company_name",
            "location",
            "original_listed_time",
            "listed_time",
            "dataset_version",
        ]
    )

    # ================================================================
    # LEAKAGE COLUMNS
    # ================================================================

    leakage_columns: List[str] = field(
        default_factory=lambda: [
            "min_salary",
            "med_salary",
            "max_salary",
            "normalized_salary",
            "listed_min_salary",
            "listed_med_salary",
            "listed_max_salary",
            "salary_available",
            "views",
            "applies",
            "closed_time",
            "expiry",
            "dataset_version",
            "pay_period",
            "currency",
            "listed_currency",
            "listed_pay_period",
            "listed_compensation_type",
        ]
    )

    # ================================================================
    # POSTING GROUP
    # ================================================================

    posting_group_id_col: str = (
        "posting_group_id"
    )

    posting_group_source_columns: List[str] = field(
        default_factory=lambda: [
            "title",
            "company_name",
            "location",
        ]
    )

    # ================================================================
    # COMPANY NUMERIC FEATURES
    # ================================================================

    company_employee_count_col: str = (
        "company_employee_count"
    )

    company_follower_count_col: str = (
        "company_follower_count"
    )

    log_company_employee_count_col: str = (
        "log_company_employee_count"
    )

    log_company_follower_count_col: str = (
        "log_company_follower_count"
    )

    # ================================================================
    # PATHS
    # ================================================================

    @property
    def feature_store_path(self) -> Path:

        return (
            self.base_artifacts_dir
            / self.feature_store_subdir
            / self.feature_store_filename
        )

    @property
    def output_dir(self) -> Path:

        return (
            self.base_artifacts_dir
            / self.output_subdir
        )

    @property
    def latest_dir(self) -> Path:

        return (
            self.output_dir
            / "latest"
        )

    @property
    def archive_dir(self) -> Path:

        return (
            self.output_dir
            / "archive"
        )
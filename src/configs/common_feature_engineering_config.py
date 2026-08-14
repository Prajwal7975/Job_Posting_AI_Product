"""
src/configs/common_feature_engineering_config.py

Configuration for the model-agnostic Common Feature Engineering stage.

Pipeline contract:

master_dataset.parquet
        ↓
CommonFeatureEngineering
        ↓
feature_store/latest/feature_store.csv
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.logger import logging
from src.exception import CustomException


@dataclass
class CommonFeatureEngineeringConfig:

    # ================================================================
    # PATHS
    # ================================================================

    master_dataset_path: str = (
        "artifacts/master_dataset/latest/master_dataset.parquet"
    )

    latest_output_dir: str = (
        "artifacts/feature_store/latest"
    )

    archive_root_dir: str = (
        "artifacts/feature_store/archive"
    )

    save_archive_copy: bool = True
    save_metadata: bool = True
    save_report: bool = True
    save_schema: bool = True

    # ================================================================
    # IDENTIFIERS
    #
    # These are NOT model features.
    # They are removed from the common feature store.
    # ================================================================

    identifier_columns: List[str] = field(
        default_factory=lambda: [
            "job_id",
            "company_id",
            "salary_id",
        ]
    )

    # ================================================================
    # DATETIME
    # ================================================================

    datetime_columns: List[str] = field(
        default_factory=lambda: [
            "listed_time",
            "original_listed_time",
            "expiry",
            "closed_time",
            "time_recorded",
        ]
    )

    epoch_ms_datetime_columns: List[str] = field(
        default_factory=lambda: [
            "listed_time",
            "original_listed_time",
            "expiry",
            "closed_time",
            "time_recorded",
        ]
    )

    # ================================================================
    # BOOLEAN
    # ================================================================

    boolean_columns: List[str] = field(
        default_factory=lambda: [
            "remote_allowed",
            "sponsored",
        ]
    )

    boolean_truthy_values: List[Any] = field(
        default_factory=lambda: [
            1,
            1.0,
            "1",
            "true",
            "True",
            "TRUE",
            "t",
            "T",
            "yes",
            "Yes",
            "YES",
            "y",
            "Y",
        ]
    )

    # ================================================================
    # CATEGORICAL COLUMNS
    # ================================================================

    categorical_columns: List[str] = field(
        default_factory=lambda: [
            "formatted_work_type",
            "work_type",
            "formatted_experience_level",
            "pay_period",
            "compensation_type",
            "currency",
            "application_type",
            "company_size",
            "company_state",
            "company_country",
            "company_city",
            "top_industry",
            "top_skill",
        ]
    )

    # ================================================================
    # TEXT COLUMNS
    #
    # IMPORTANT:
    # Description is preserved.
    # We are NOT extracting skills here yet.
    # Skill extraction will be handled in the salary-specific stage.
    # ================================================================

    text_columns: List[str] = field(
        default_factory=lambda: [
            "title",
            "description",
            "company_name",
            "company_description",
            "location",
            "company_address",
        ]
    )

    # ================================================================
    # LIST COLUMNS
    # ================================================================

    list_columns: List[str] = field(
        default_factory=lambda: [
            "skill_list",
            "benefit_list",
            "industry_list",
            "speciality_list",
        ]
    )

    list_column_delimiter: str = "|"

    # ================================================================
    # NUMERIC COLUMNS
    #
    # Empty means infer from actual dataframe.
    # ================================================================

    numeric_columns: List[str] = field(
        default_factory=list
    )

    # ================================================================
    # MISSING VALUES
    # ================================================================

    missing_value_strategy: Dict[str, str] = field(
        default_factory=lambda: {

            # --------------------------------------------------------
            # Salary fields
            # Leave untouched because Salary FE constructs target.
            # --------------------------------------------------------

            "min_salary": "leave",
            "med_salary": "leave",
            "max_salary": "leave",
            "normalized_salary": "leave",

            # --------------------------------------------------------
            # Engagement
            # --------------------------------------------------------

            "views": "median",
            "applies": "median",

            # --------------------------------------------------------
            # Company numeric features
            # --------------------------------------------------------

            "company_employee_count": "median",
            "company_follower_count": "median",

            # --------------------------------------------------------
            # Location numeric features
            # --------------------------------------------------------

            "zip_code": "leave",
            "fips": "leave",

            # --------------------------------------------------------
            # Categorical
            # --------------------------------------------------------

            "formatted_experience_level": "constant",
            "company_size": "constant",
        }
    )

    default_numeric_strategy: str = "median"

    # Do not use mode blindly on arbitrary text columns.
    default_categorical_strategy: str = "mode"

    default_text_strategy: str = "constant"

    constant_fill_values: Dict[str, Any] = field(
        default_factory=lambda: {
            "formatted_experience_level": "UNKNOWN",
            "company_size": "UNKNOWN",
        }
    )

    default_constant_numeric_fill: float = -1.0

    default_constant_text_fill: str = "UNKNOWN"

    # ================================================================
    # DERIVED FEATURES
    # ================================================================

    derived_feature_flags: Dict[str, bool] = field(
        default_factory=lambda: {

            "salary_available": True,

            "has_company_description": True,

            "has_description": True,

            "listing_age_available": True,

            "skill_count": True,

            "benefit_count": True,

            "industry_count": True,

            "speciality_count": True,
        }
    )

    # ================================================================
    # DUPLICATES
    # ================================================================

    drop_duplicates: bool = True

    # None = remove completely identical rows.
    #
    # We deliberately DO NOT use job_id here because job_id is removed
    # before duplicate processing.
    duplicate_subset: Optional[List[str]] = None

    # ================================================================
    # VALIDATION
    # ================================================================

    fail_on_duplicate_column_names: bool = True

    fail_on_empty_output: bool = True

    # ================================================================
    # OUTPUT CONTRACT
    #
    # SalaryFeatureEngineering consumes this exact file.
    # ================================================================

    feature_store_filename: str = "feature_store.csv"

    # ================================================================
    # INITIALIZATION
    # ================================================================

    def __post_init__(self) -> None:

        try:

            logging.info(
                "CommonFeatureEngineeringConfig initialized successfully."
            )

            logging.info(
                "Master dataset: %s",
                self.master_dataset_path,
            )

            logging.info(
                "Feature store output: %s/%s",
                self.latest_output_dir,
                self.feature_store_filename,
            )

        except Exception as e:

            raise CustomException(
                e,
                sys,
            )
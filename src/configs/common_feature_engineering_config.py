"""
Configuration for the Common Feature Engineering stage.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.logger import logging
from src.exception import CustomException


@dataclass
class CommonFeatureEngineeringConfig:
    master_dataset_path: str = "artifacts/master_dataset/latest/master_dataset.parquet"
    latest_output_dir: str = "artifacts/feature_store/latest"
    archive_root_dir: str = "artifacts/feature_store/archive"
    
    save_archive_copy: bool = True
    save_metadata: bool = True
    save_report: bool = True
    save_schema: bool = True

    identifier_columns: List[str] = field(
        default_factory=lambda: ["job_id", "company_id", "salary_id"]
    )

    datetime_columns: List[str] = field(
        default_factory=lambda: [
            "listed_time", "original_listed_time", "expiry",
            "closed_time", "time_recorded",
        ]
    )
    epoch_ms_datetime_columns: List[str] = field(
        default_factory=lambda: [
            "listed_time", "original_listed_time", "expiry",
            "closed_time", "time_recorded",
        ]
    )

    boolean_columns: List[str] = field(
        default_factory=lambda: ["remote_allowed", "sponsored"]
    )
    boolean_truthy_values: List[Any] = field(
        default_factory=lambda: [
            1, 1.0, "1", "true", "True", "TRUE", "t", "T", "yes", "Yes", "YES", "y", "Y",
        ]
    )

    categorical_columns: List[str] = field(
        default_factory=lambda: [
            "formatted_work_type", "work_type", "formatted_experience_level",
            "pay_period", "compensation_type", "currency", "application_type",
            "company_size", "company_state", "company_country", "company_city",
        ]
    )

    text_columns: List[str] = field(
        default_factory=lambda: [
            "title", "description", "company_name", "company_description",
            "location", "company_address",
        ]
    )

    list_columns: List[str] = field(
        default_factory=lambda: [
            "skill_list", "benefit_list", "industry_list", "speciality_list",
        ]
    )
    list_column_delimiter: str = "|"
    numeric_columns: List[str] = field(default_factory=list)

    missing_value_strategy: Dict[str, str] = field(
        default_factory=lambda: {
            "max_salary": "leave",
            "med_salary": "leave",
            "min_salary": "leave",
            "normalized_salary": "leave",
            "views": "median",
            "applies": "median",
            "company_employee_count": "median",
            "company_follower_count": "median",
            "zip_code": "leave",
            "fips": "leave",
            "formatted_experience_level": "constant",
            "company_size": "constant",
        }
    )
    default_numeric_strategy: str = "median"
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

    drop_duplicates: bool = True
    duplicate_subset: List[str] | None = None
    fail_on_duplicate_column_names: bool = True
    fail_on_empty_output: bool = True

    def __post_init__(self):
        try:
            logging.info("CommonFeatureEngineeringConfig initialized successfully.")
        except Exception as e:
            raise CustomException(e, sys)
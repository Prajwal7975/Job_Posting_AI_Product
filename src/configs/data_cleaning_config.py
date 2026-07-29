from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple, Any, Optional

from src.configs.schema_alignment_config import SchemaAlignmentConfig


class DuplicatePolicy(str, Enum):
    KEEP_FIRST = "first"
    KEEP_LAST = "last"
    DROP_ALL = "drop_all"


class NullPolicy(str, Enum):
    DROP_ROW = "drop_row"
    FILL_DEFAULT = "fill_default"
    LEAVE = "leave"


class ForeignKeyPolicy(str, Enum):
    DROP_CHILD = "drop_child"
    LEAVE = "leave"


class InvalidValuePolicy(str, Enum):
    DROP_ROW = "drop_row"
    NULLIFY = "nullify"
    LEAVE = "leave"


@dataclass(frozen=True)
class TableCleaningRule:
    """Configurable cleaning policies per table."""

    primary_keys: Tuple[str, ...] = field(default_factory=tuple)
    null_policies: Dict[str, NullPolicy] = field(default_factory=dict)
    default_fill_values: Dict[str, Any] = field(default_factory=dict)

    # Parent table references for foreign key enforcement:
    # Format: {"fk_column": ("parent_table_name", "parent_key_column")}
    foreign_keys: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    # String manipulation policies
    lowercase_columns: Tuple[str, ...] = field(default_factory=tuple)

    # Numeric boundary policies: {"col_name": min_allowed_value}
    numeric_min_bounds: Dict[str, float] = field(default_factory=dict)

    # Execution policies
    duplicate_policy: DuplicatePolicy = DuplicatePolicy.KEEP_FIRST
    fk_policy: ForeignKeyPolicy = ForeignKeyPolicy.DROP_CHILD
    invalid_value_policy: InvalidValuePolicy = InvalidValuePolicy.LEAVE


@dataclass(frozen=True)
class DataCleaningConfig:
    """Global configuration for Data Cleaning component."""

    trim_strings: bool = True
    normalize_whitespace: bool = True
    convert_empty_strings_to_null: bool = True
    continue_on_error: bool = True

    # Single source of truth for schema definitions
    schema_alignment_config: SchemaAlignmentConfig = field(
        default_factory=SchemaAlignmentConfig
    )

    # Per-table override configurations matching schema relationships
    table_rules: Dict[str, TableCleaningRule] = field(
        default_factory=lambda: {
            "postings": TableCleaningRule(
                primary_keys=("job_id",),
                null_policies={"company_id": NullPolicy.DROP_ROW},
                foreign_keys={"company_id": ("companies", "company_id")},
            ),
            "companies": TableCleaningRule(
                primary_keys=("company_id",),
                null_policies={"name": NullPolicy.FILL_DEFAULT},
                default_fill_values={"name": "Unknown Company"},
            ),
            "industries": TableCleaningRule(
                primary_keys=("industry_id",),
                null_policies={"industry_name": NullPolicy.FILL_DEFAULT},
                default_fill_values={"industry_name": "Unspecified"},
                lowercase_columns=("industry_name",),
            ),
            "job_industries": TableCleaningRule(
                primary_keys=("job_id", "industry_id"),
                foreign_keys={
                    "job_id": ("postings", "job_id"),
                    "industry_id": ("industries", "industry_id"),
                },
            ),
            "benefits": TableCleaningRule(
                foreign_keys={"job_id": ("postings", "job_id")},
                lowercase_columns=("type",),
            ),
            "company_specialities": TableCleaningRule(
                foreign_keys={"company_id": ("companies", "company_id")},
                lowercase_columns=("speciality",),
            ),
            "employee_counts": TableCleaningRule(
                foreign_keys={"company_id": ("companies", "company_id")},
            ),
            "job_skills": TableCleaningRule(
                primary_keys=("job_id", "skill_abr"),
                foreign_keys={
                    "job_id": ("postings", "job_id"),
                    "skill_abr": ("skills", "skill_abr"),
                },
            ),
            "salaries": TableCleaningRule(
                foreign_keys={"job_id": ("postings", "job_id")},
                numeric_min_bounds={
                    "med_salary": 0.0,
                    "min_salary": 0.0,
                    "max_salary": 0.0,
                },
                invalid_value_policy=InvalidValuePolicy.DROP_ROW,
            ),
            "skills": TableCleaningRule(
                primary_keys=("skill_abr",),
                lowercase_columns=("skill_name"),
            ),
        }
    )
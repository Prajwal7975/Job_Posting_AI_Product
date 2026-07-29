from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from src.configs.schema_alignment_config import SchemaAlignmentConfig


@dataclass(frozen=True)
class ValueConstraint:
    allowed_values: Optional[Tuple[object, ...]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    regex_pattern: Optional[str] = None


@dataclass(frozen=True)
class ForeignKeyRule:
    column: str
    references_table: str
    references_column: str


@dataclass(frozen=True)
class TableValidationRules:

    unique_key_columns: Tuple[str, ...] = field(default_factory=tuple)
    not_null_columns: Tuple[str, ...] = field(default_factory=tuple)
    value_constraints: Dict[str, ValueConstraint] = field(default_factory=dict)
    foreign_keys: Tuple[ForeignKeyRule, ...] = field(default_factory=tuple)


def _build_default_validation_rules() -> Dict[str, TableValidationRules]:

    return {
        "postings": TableValidationRules(
            unique_key_columns=("job_id",),
            not_null_columns=("job_id", "company_id", "title"),
            value_constraints={
                # Range Validation -- realistic numeric bounds.
                "min_salary": ValueConstraint(min_value=0),
                "max_salary": ValueConstraint(min_value=0),
                # Range Validation -- dates stored as epoch milliseconds.
                # Bounds are a sanity placeholder (year 2000 - year 2100);
                # narrow them once you know your real ingestion window.
                "listed_time": ValueConstraint(min_value=946684800000, max_value=4102444800000),
                "original_listed_time": ValueConstraint(min_value=946684800000, max_value=4102444800000),
                "expiry": ValueConstraint(min_value=946684800000, max_value=4102444800000),
                "closed_time": ValueConstraint(min_value=946684800000, max_value=4102444800000),
                # Domain Validation -- closed vocabularies.
                "remote_allowed": ValueConstraint(allowed_values=(0, 1)),
                "work_type": ValueConstraint(allowed_values=(
                    "FULL_TIME", "PART_TIME", "CONTRACT", "TEMPORARY",
                    "INTERNSHIP", "VOLUNTEER", "OTHER",
                )),
                "formatted_experience_level": ValueConstraint(allowed_values=(
                    "Internship", "Entry level", "Associate",
                    "Mid-Senior level", "Director", "Executive",
                )),
                "pay_period": ValueConstraint(allowed_values=(
                    "HOURLY", "WEEKLY", "BIWEEKLY", "MONTHLY", "YEARLY", "ONCE",
                )),
            },
            foreign_keys=(
                ForeignKeyRule("company_id", "companies", "company_id"),
            ),
        ),
        "companies": TableValidationRules(
            unique_key_columns=("company_id",),
            not_null_columns=("company_id", "name"),
        ),
        "industries": TableValidationRules(
            unique_key_columns=("industry_id",),
            not_null_columns=("industry_id", "industry_name"),
        ),
        "job_industries": TableValidationRules(
            unique_key_columns=("job_id", "industry_id"),
            not_null_columns=("job_id", "industry_id"),
            foreign_keys=(
                ForeignKeyRule("job_id", "postings", "job_id"),
                ForeignKeyRule("industry_id", "industries", "industry_id"),
            ),
        ),
        "benefits": TableValidationRules(
            not_null_columns=("job_id",),
            foreign_keys=(
                ForeignKeyRule("job_id", "postings", "job_id"),
            ),
        ),
        "company_specialities": TableValidationRules(
            not_null_columns=("company_id",),
            foreign_keys=(
                ForeignKeyRule("company_id", "companies", "company_id"),
            ),
        ),
        "employee_counts": TableValidationRules(
            not_null_columns=("company_id",),
            value_constraints={
                "employee_count": ValueConstraint(min_value=0),
                "follower_count": ValueConstraint(min_value=0),
            },
            foreign_keys=(
                ForeignKeyRule("company_id", "companies", "company_id"),
            ),
        ),
        "job_skills": TableValidationRules(
            not_null_columns=("job_id", "skill_abr"),
            foreign_keys=(
                ForeignKeyRule("job_id", "postings", "job_id"),
                ForeignKeyRule("skill_abr", "skills", "skill_abr"),
            ),
        ),
        "skills": TableValidationRules(
            unique_key_columns=("skill_abr",),
            not_null_columns=("skill_abr",),
        ),
        "salaries": TableValidationRules(
            not_null_columns=("job_id",),
            value_constraints={
                "min_salary": ValueConstraint(min_value=0),
                "max_salary": ValueConstraint(min_value=0),
            },
            foreign_keys=(
                ForeignKeyRule("job_id", "postings", "job_id"),
            ),
        ),
    }


@dataclass(frozen=True)
class DataValidationConfig:

    table_rules: Dict[str, TableValidationRules] = field(default_factory=_build_default_validation_rules)
    schema_alignment_config: SchemaAlignmentConfig = field(default_factory=SchemaAlignmentConfig)

    raise_on_null_violations: bool = False
    raise_on_duplicate_keys: bool = False
    raise_on_invalid_values: bool = False
    raise_on_dtype_conformance_issues: bool = False
    raise_on_broken_relationships: bool = False
    raise_on_structural_issues: bool = False
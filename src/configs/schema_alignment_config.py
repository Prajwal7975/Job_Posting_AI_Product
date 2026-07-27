from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class TableSchema:
    
    column_order: Tuple[str, ...]
    aliases: Dict[str, Tuple[str, ...]]
    canonical_dtypes: Dict[str, str] = field(default_factory=dict)
    required_columns: Tuple[str, ...] = field(default_factory=tuple)


def _build_default_table_schemas() -> Dict[str, TableSchema]:

    return {
        "postings": TableSchema(
            column_order=(
                "job_id", "company_id", "title", "description", "location",
                "views", "applies", "min_salary", "med_salary", "max_salary",
                "pay_period", "currency", "compensation_type",
                "normalized_salary", "formatted_work_type", "work_type",
                "remote_allowed", "formatted_experience_level",
                "listed_time", "original_listed_time", "expiry",
                "closed_time", "sponsored", "zip_code", "fips",
            ),
            aliases={
                "job_id": ("job_id", "id"),
                "company_id": ("company_id",),
                "title": ("title", "job_title"),
                "description": ("description", "job_description"),
                "location": ("location",),
                "views": ("views",),
                "applies": ("applies",),
                "min_salary": ("min_salary",),
                "med_salary": ("med_salary",),
                "max_salary": ("max_salary",),
                "pay_period": ("pay_period",),
                "currency": ("currency",),
                "compensation_type": ("compensation_type",),
                "normalized_salary": ("normalized_salary",),
                "formatted_work_type": ("formatted_work_type", "work_type_name"),
                "work_type": ("work_type",),
                "remote_allowed": ("remote_allowed",),
                "formatted_experience_level": ("formatted_experience_level", "experience_level"),
                "listed_time": ("listed_time",),
                "original_listed_time": ("original_listed_time",),
                "expiry": ("expiry",),
                "closed_time": ("closed_time",),
                "sponsored": ("sponsored",),
                "zip_code": ("zip_code",),
                "fips": ("fips",),
            },
            canonical_dtypes={
                "job_id": "Int64",
                "company_id": "Int64",
                "title": "string",
                "description": "string",
                "location": "string",
                "views": "Int64",
                "applies": "Int64",
                "min_salary": "float64",
                "med_salary": "float64",
                "max_salary": "float64",
                "pay_period": "string",
                "currency": "string",
                "compensation_type": "string",
                "normalized_salary": "float64",
                "formatted_work_type": "string",
                "work_type": "string",
                "remote_allowed": "float64",
                "formatted_experience_level": "string",
                "listed_time": "float64",
                "original_listed_time": "float64",
                "expiry": "float64",
                "closed_time": "float64",
                "sponsored": "Int64",
                "zip_code": "string",
                "fips": "float64",
            },
            required_columns=("job_id",),
        ),
        "companies": TableSchema(
            column_order=(
                "company_id", "name", "description", "company_size",
                "state", "country", "city", "zip_code", "address", "url",
            ),
            aliases={
                "company_id": ("company_id",),
                "name": ("name", "company_name"),
                "description": ("description",),
                "company_size": ("company_size",),
                "state": ("state",),
                "country": ("country",),
                "city": ("city",),
                "zip_code": ("zip_code",),
                "address": ("address",),
                "url": ("url",),
            },
            canonical_dtypes={
                "company_id": "Int64",
                "name": "string",
                "description": "string",
                "company_size": "float64",
                "state": "string",
                "country": "string",
                "city": "string",
                "zip_code": "string",
                "address": "string",
                "url": "string",
            },
            required_columns=("company_id",),
        ),
        "industries": TableSchema(
            column_order=("industry_id", "industry_name"),
            aliases={
                "industry_id": ("industry_id",),
                "industry_name": ("industry_name",),
            },
            canonical_dtypes={
                "industry_id": "Int64",
                "industry_name": "string",
            },
            required_columns=("industry_id",),
        ),
        "job_industries": TableSchema(
            column_order=("job_id", "industry_id"),
            aliases={
                "job_id": ("job_id",),
                "industry_id": ("industry_id",),
            },
            canonical_dtypes={
                "job_id": "Int64",
                "industry_id": "Int64",
            },
            required_columns=("job_id", "industry_id"),
        ),
        "benefits": TableSchema(
            column_order=("job_id", "inferred", "type"),
            aliases={
                "job_id": ("job_id",),
                "inferred": ("inferred",),
                "type": ("type",),
            },
            canonical_dtypes={
                "job_id": "Int64",
                "inferred": "Int64",
                "type": "string",
            },
            required_columns=("job_id",),
        ),
        "company_specialities": TableSchema(
            column_order=("company_id", "speciality"),
            aliases={
                "company_id": ("company_id",),
                "speciality": ("speciality",),
            },
            canonical_dtypes={
                "company_id": "Int64",
                "speciality": "string",
            },
            required_columns=("company_id",),
        ),
        "employee_counts": TableSchema(
            column_order=(
                "company_id", "employee_count", "follower_count", "time_recorded",
            ),
            aliases={
                "company_id": ("company_id",),
                "employee_count": ("employee_count",),
                "follower_count": ("follower_count",),
                "time_recorded": ("time_recorded",),
            },
            canonical_dtypes={
                "company_id": "Int64",
                "employee_count": "Int64",
                "follower_count": "Int64",
                "time_recorded": "float64",
            },
            required_columns=("company_id",),
        ),
        "job_skills": TableSchema(
            column_order=("job_id", "skill_abr"),
            aliases={
                "job_id": ("job_id",),
                "skill_abr": ("skill_abr",),
            },
            canonical_dtypes={
                "job_id": "Int64",
                "skill_abr": "string",
            },
            required_columns=("job_id",),
        ),
        "skills": TableSchema(
            column_order=("skill_abr", "skill_name"),
            aliases={
                "skill_abr": ("skill_abr",),
                "skill_name": ("skill_name",),
            },
            canonical_dtypes={
                "skill_abr": "string",
                "skill_name": "string",
            },
            required_columns=("skill_abr",),
        ),
        "salaries": TableSchema(
            column_order=(
                "salary_id", "job_id", "max_salary", "med_salary", "min_salary",
                "pay_period", "currency", "compensation_type",
            ),
            aliases={
                "salary_id": ("salary_id",),
                "job_id": ("job_id",),
                "max_salary": ("max_salary",),
                "med_salary": ("med_salary",),
                "min_salary": ("min_salary",),
                "pay_period": ("pay_period",),
                "currency": ("currency",),
                "compensation_type": ("compensation_type",),
            },
            canonical_dtypes={
                "salary_id": "Int64",
                "job_id": "Int64",
                "max_salary": "float64",
                "med_salary": "float64",
                "min_salary": "float64",
                "pay_period": "string",
                "currency": "string",
                "compensation_type": "string",
            },
            required_columns=("job_id",),
        ),
    }


@dataclass(frozen=True)
class SchemaAlignmentConfig:

    table_schemas: Dict[str, TableSchema] = field(
        default_factory=_build_default_table_schemas
    )
    preserve_unmapped_columns: bool = True
    unmapped_column_prefix: str = "unmapped__"
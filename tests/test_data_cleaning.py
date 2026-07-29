"""
Unit tests for the Data Cleaning component.
Designed for local pytest execution and GitHub Actions CI/CD pipeline automation.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
import pandas as pd
import pytest

from src.components.data_cleaning import DataCleaning
from src.configs.data_cleaning_config import (
    DataCleaningConfig,
    DuplicatePolicy,
    ForeignKeyPolicy,
    InvalidValuePolicy,
    NullPolicy,
    TableCleaningRule,
)


# ==============================================================================
# MOCK SCHEMA CONFIGURATION FOR ISOLATED TESTING
# ==============================================================================

@dataclass
class MockTableSchema:
    column_order: Tuple[str, ...] = field(default_factory=tuple)
    aliases: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    canonical_dtypes: Dict[str, str] = field(default_factory=dict)
    required_columns: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class MockSchemaAlignmentConfig:
    table_schemas: Dict[str, MockTableSchema] = field(
        default_factory=lambda: {
            "postings": MockTableSchema(canonical_dtypes={"job_id": "Int64", "company_id": "Int64"}),
            "companies": MockTableSchema(canonical_dtypes={"company_id": "Int64", "name": "string"}),
            "salaries": MockTableSchema(canonical_dtypes={"job_id": "Int64", "med_salary": "float64"}),
            "industries": MockTableSchema(canonical_dtypes={"industry_id": "Int64", "industry_name": "string"}),
            "job_skills": MockTableSchema(canonical_dtypes={"job_id": "Int64", "skill_abr": "string"}),
            "skills": MockTableSchema(canonical_dtypes={"skill_abr": "string", "skill_name": "string"}),
        }
    )


# ==============================================================================
# PYTEST FIXTURES
# ==============================================================================

@pytest.fixture
def mock_cleaner_config() -> DataCleaningConfig:
    """Provides a controlled DataCleaningConfig injected with MockSchemaAlignmentConfig."""
    return DataCleaningConfig(
        schema_alignment_config=MockSchemaAlignmentConfig()
    )


@pytest.fixture
def sample_validated_data() -> Dict[str, Dict[str, pd.DataFrame]]:
    """Generates a realistic multi-version, multi-table dataset with relational anomalies."""
    companies_df = pd.DataFrame(
        {
            "company_id": [1, 2],
            "name": [" Google ", None],  # Contains whitespace and null
        }
    )

    postings_df = pd.DataFrame(
        {
            "job_id": [101, 102, 103, 104],
            "company_id": [1, 2, None, 999],  # 1 missing FK, 1 orphan FK (999)
            "title": [" Senior Software Engineer ", "Data  Scientist", "DevOps", " Designer "],
        }
    )

    job_skills_df = pd.DataFrame(
        {
            "job_id": [101, 102, 8888],  # 8888 is an orphan job_id
            "skill_abr": ["ENG", "IT", "MKT"],
        }
    )

    industries_df = pd.DataFrame(
        {
            "industry_id": [10, 20],
            "industry_name": [" TECHNOLOGY ", "HEALTHCARE"],  # Tests lowercasing & trimming
        }
    )
    
    skills_df = pd.DataFrame(
        {
        "skill_abr": ["ENG", "IT"],
        "skill_name": ["Engineering", "Information Technology"],
        }
    )

    return {
        "Version_1": {
            "companies": companies_df,
            "postings": postings_df,
            "job_skills": job_skills_df,
            "industries": industries_df,
            "skills": skills_df,
        }
    }


# ==============================================================================
# TEST CASES: STRING CLEANING & LOWERCASE POLICIES
# ==============================================================================

def test_string_trimming_whitespace_and_lowercase(sample_validated_data, mock_cleaner_config):
    """Verifies that strings are trimmed, multiple spaces normalized, and lowercase policies enforced."""
    cleaner = DataCleaning(config=mock_cleaner_config)
    result = cleaner.initiate_data_cleaning(sample_validated_data)

    df_postings = result.cleaned_data["Version_1"]["postings"]
    df_industries = result.cleaned_data["Version_1"]["industries"]

    # Trim & normalize space checks
    assert df_postings.loc[df_postings["job_id"] == 101, "title"].values[0] == "Senior Software Engineer"
    assert df_postings.loc[df_postings["job_id"] == 102, "title"].values[0] == "Data Scientist"

    # Lowercase policy check on industries table
    assert df_industries.loc[df_industries["industry_id"] == 10, "industry_name"].values[0] == "technology"
    assert df_industries.loc[df_industries["industry_id"] == 20, "industry_name"].values[0] == "healthcare"


def test_empty_string_conversion_to_null_and_filling():
    """Ensures empty strings and whitespace-only strings become NaN and are subsequently filled by policy."""
    df = pd.DataFrame({"company_id": [1, 2, 3], "name": ["Google", "   ", ""]})
    raw_data = {"V1": {"companies": df}}

    config = DataCleaningConfig(
        schema_alignment_config=MockSchemaAlignmentConfig(),
        table_rules={
            "companies": TableCleaningRule(
                null_policies={"name": NullPolicy.FILL_DEFAULT},
                default_fill_values={"name": "Unknown Company"},
            )
        },
    )

    cleaner = DataCleaning(config=config)
    result = cleaner.initiate_data_cleaning(raw_data)

    cleaned_names = result.cleaned_data["V1"]["companies"]["name"].tolist()
    assert cleaned_names == ["Google", "Unknown Company", "Unknown Company"]


# ==============================================================================
# TEST CASES: NULL HANDLING POLICIES
# ==============================================================================

def test_null_handling_drop_row_and_fill_default(sample_validated_data, mock_cleaner_config):
    """Tests DROP_ROW and FILL_DEFAULT null policies on relational columns."""
    cleaner = DataCleaning(config=mock_cleaner_config)
    result = cleaner.initiate_data_cleaning(sample_validated_data)

    df_companies = result.cleaned_data["Version_1"]["companies"]
    df_postings = result.cleaned_data["Version_1"]["postings"]

    # Companies: 'name' is NullPolicy.FILL_DEFAULT with 'Unknown Company'
    assert (df_companies["name"] == "Unknown Company").sum() == 1

    # Postings: 'company_id' is NullPolicy.DROP_ROW
    assert df_postings["company_id"].isna().sum() == 0


def test_null_handling_leave_policy():
    """Tests NullPolicy.LEAVE to ensure nulls are preserved when explicitly configured."""
    df = pd.DataFrame({"company_id": [1, 2], "name": ["Google", None]})
    raw_data = {"V1": {"companies": df}}

    config = DataCleaningConfig(
        schema_alignment_config=MockSchemaAlignmentConfig(),
        table_rules={
            "companies": TableCleaningRule(
                null_policies={"name": NullPolicy.LEAVE}
            )
        },
    )
    cleaner = DataCleaning(config=config)
    result = cleaner.initiate_data_cleaning(raw_data)

    assert result.cleaned_data["V1"]["companies"]["name"].isna().sum() == 1


# ==============================================================================
# TEST CASES: DTYPE CONVERSION & SCHEMA ALIGNMENT
# ==============================================================================

def test_dtype_conversion_integration(sample_validated_data, mock_cleaner_config):
    """Ensures schema alignment target dtypes (e.g., nullable Int64) are enforced."""
    cleaner = DataCleaning(config=mock_cleaner_config)
    result = cleaner.initiate_data_cleaning(sample_validated_data)

    df_postings = result.cleaned_data["Version_1"]["postings"]
    assert str(df_postings["job_id"].dtype) == "Int64"
    assert result.summary.total_dtype_conversions > 0


# ==============================================================================
# TEST CASES: INVALID VALUE HANDLING POLICIES
# ==============================================================================

@pytest.mark.parametrize(
    "policy,expected_rows,expected_nulls",
    [
        (InvalidValuePolicy.DROP_ROW, 2, 0),
        (InvalidValuePolicy.NULLIFY, 3, 1),
        (InvalidValuePolicy.LEAVE, 3, 0),
    ],
)
def test_invalid_numeric_value_policies(policy, expected_rows, expected_nulls):
    """Tests out-of-bounds numeric filtering across DROP_ROW, NULLIFY, and LEAVE policies."""
    salaries_df = pd.DataFrame(
        {"job_id": [101, 102, 103], "med_salary": [60000.0, -500.0, 90000.0]}
    )
    raw_data = {"V1": {"salaries": salaries_df}}

    config = DataCleaningConfig(
        schema_alignment_config=MockSchemaAlignmentConfig(),
        table_rules={
            "salaries": TableCleaningRule(
                numeric_min_bounds={"med_salary": 0.0},
                invalid_value_policy=policy,
            )
        },
    )

    cleaner = DataCleaning(config=config)
    result = cleaner.initiate_data_cleaning(raw_data)

    cleaned_df = result.cleaned_data["V1"]["salaries"]
    assert len(cleaned_df) == expected_rows
    assert cleaned_df["med_salary"].isna().sum() == expected_nulls


# ==============================================================================
# TEST CASES: DUPLICATE REMOVAL POLICIES (SPLIT FOR ISOLATED DIAGNOSTICS)
# ==============================================================================

def test_duplicate_policy_keep_first():
    """Validates DuplicatePolicy.KEEP_FIRST retains the first matching primary key."""
    dup_df = pd.DataFrame({"job_id": [1, 1, 2], "value": ["first_record", "second_record", "unique_record"]})
    raw_data = {"V1": {"postings": dup_df}}

    cfg = DataCleaningConfig(
        schema_alignment_config=MockSchemaAlignmentConfig(),
        table_rules={"postings": TableCleaningRule(primary_keys=("job_id",), duplicate_policy=DuplicatePolicy.KEEP_FIRST)},
    )
    res = DataCleaning(cfg).initiate_data_cleaning(raw_data)
    assert res.cleaned_data["V1"]["postings"]["value"].tolist() == ["first_record", "unique_record"]


def test_duplicate_policy_keep_last():
    """Validates DuplicatePolicy.KEEP_LAST retains the last matching primary key."""
    dup_df = pd.DataFrame({"job_id": [1, 1, 2], "value": ["first_record", "second_record", "unique_record"]})
    raw_data = {"V1": {"postings": dup_df}}

    cfg = DataCleaningConfig(
        schema_alignment_config=MockSchemaAlignmentConfig(),
        table_rules={"postings": TableCleaningRule(primary_keys=("job_id",), duplicate_policy=DuplicatePolicy.KEEP_LAST)},
    )
    res = DataCleaning(cfg).initiate_data_cleaning(raw_data)
    assert res.cleaned_data["V1"]["postings"]["value"].tolist() == ["second_record", "unique_record"]


def test_duplicate_policy_drop_all():
    """Validates DuplicatePolicy.DROP_ALL purges all instances of duplicated primary keys."""
    dup_df = pd.DataFrame({"job_id": [1, 1, 2], "value": ["first_record", "second_record", "unique_record"]})
    raw_data = {"V1": {"postings": dup_df}}

    cfg = DataCleaningConfig(
        schema_alignment_config=MockSchemaAlignmentConfig(),
        table_rules={"postings": TableCleaningRule(primary_keys=("job_id",), duplicate_policy=DuplicatePolicy.DROP_ALL)},
    )
    res = DataCleaning(cfg).initiate_data_cleaning(raw_data)
    assert res.cleaned_data["V1"]["postings"]["value"].tolist() == ["unique_record"]


# ==============================================================================
# REGRESSION TEST: PIPELINE SEQUENCE & MIXED-TYPE DEDUPLICATION
# ==============================================================================

def test_mixed_type_duplicate_key_resolution():
    """Regression test: Verifies dtype conversion precedes deduplication so mixed-type keys collapse correctly."""
    mixed_df = pd.DataFrame({
        "job_id": ["101", 101, 101.0],
        "title": ["Software Engineer", "Software Engineer", "Software Engineer"]
    })
    raw_data = {"V1": {"postings": mixed_df}}

    config = DataCleaningConfig(
        schema_alignment_config=MockSchemaAlignmentConfig(),
        table_rules={
            "postings": TableCleaningRule(
                primary_keys=("job_id",),
                duplicate_policy=DuplicatePolicy.KEEP_FIRST,
            )
        },
    )

    cleaner = DataCleaning(config=config)
    result = cleaner.initiate_data_cleaning(raw_data)

    cleaned_df = result.cleaned_data["V1"]["postings"]

    assert len(cleaned_df) == 1
    assert str(cleaned_df["job_id"].dtype) == "Int64"
    assert cleaned_df["job_id"].iloc[0] == 101


# ==============================================================================
# TEST CASES: FOREIGN KEY & ORPHAN PURGING
# ==============================================================================

def test_orphan_purging_and_fk_leave_policy(sample_validated_data, mock_cleaner_config):
    """Verifies orphan records are dropped under DROP_CHILD policy and retained under LEAVE policy."""
    cleaner = DataCleaning(config=mock_cleaner_config)
    result = cleaner.initiate_data_cleaning(sample_validated_data)

    df_postings = result.cleaned_data["Version_1"]["postings"]
    df_skills = result.cleaned_data["Version_1"]["job_skills"]

    # Company 999 (orphan) in postings should be purged
    assert 999 not in df_postings["company_id"].values
    # Job ID 8888 (orphan) in job_skills should be purged
    assert 8888 not in df_skills["job_id"].values


def test_foreign_key_leave_policy(sample_validated_data):
    """Tests ForeignKeyPolicy.LEAVE to confirm orphans remain untouched."""
    config = DataCleaningConfig(
        schema_alignment_config=MockSchemaAlignmentConfig(),
        table_rules={
            "postings": TableCleaningRule(
                foreign_keys={"company_id": ("companies", "company_id")},
                fk_policy=ForeignKeyPolicy.LEAVE,
            )
        },
    )
    cleaner = DataCleaning(config=config)
    result = cleaner.initiate_data_cleaning(sample_validated_data)

    df_postings = result.cleaned_data["Version_1"]["postings"]
    assert 999 in df_postings["company_id"].values


# ==============================================================================
# TEST CASES: PIPELINE RESILIENCE & SUMMARY / REPORT METRICS
# ==============================================================================

def test_continue_on_error_resilience(mock_cleaner_config):
    """Validates pipeline fault-tolerance when continue_on_error is enabled."""
    faulty_data = {
        "V1": {
            "companies": pd.DataFrame({"company_id": [1, 2], "name": ["Co1", "Co2"]}),
            "corrupted_table": None,  # Will raise an exception when accessed
        }
    }

    cleaner = DataCleaning(config=mock_cleaner_config)
    result = cleaner.initiate_data_cleaning(faulty_data)

    assert result.summary.tables_failed == 1
    assert result.summary.tables_passed == 1
    assert result.summary.tables_processed == result.summary.tables_passed + result.summary.tables_failed
    assert "companies" in result.cleaned_data["V1"]


def test_detailed_report_and_summary_metrics(sample_validated_data, mock_cleaner_config):
    """Validates accuracy of TableCleaningReport and DataCleaningSummary including execution timing."""
    cleaner = DataCleaning(config=mock_cleaner_config)
    result = cleaner.initiate_data_cleaning(sample_validated_data)
    
    expected_tables = len(sample_validated_data["Version_1"])
    
    assert len(result.reports) == expected_tables
    
    summary = result.summary
    
    for report in result.reports:
        assert report.rows_before >= report.rows_after
        assert report.rows_removed == report.rows_before - report.rows_after
        assert report.execution_time_seconds >= 0.0

    assert summary.tables_processed == expected_tables
    assert summary.tables_passed == expected_tables
    assert summary.tables_failed == 0
    assert summary.tables_processed == summary.tables_passed + summary.tables_failed
    assert summary.total_rows_before > summary.total_rows_after
    assert summary.total_rows_removed == summary.total_rows_before - summary.total_rows_after
    assert summary.execution_time_seconds >= 0.0


# ==============================================================================
# TEST CASES: PIPELINE INVARIANTS (IDEMPOTENCE & IMMUTABILITY)
# ==============================================================================

def test_pipeline_idempotence(sample_validated_data, mock_cleaner_config):
    """Ensures that executing the cleaner a second time on cleaned output produces zero mutations."""
    cleaner = DataCleaning(config=mock_cleaner_config)

    first_pass = cleaner.initiate_data_cleaning(sample_validated_data)
    second_pass = cleaner.initiate_data_cleaning(first_pass.cleaned_data)

    assert first_pass.summary.total_rows_after == second_pass.summary.total_rows_after
    assert second_pass.summary.total_rows_removed == 0
    assert second_pass.summary.total_nulls_filled == 0
    assert second_pass.summary.total_orphan_rows_removed == 0


def test_input_data_immutability(sample_validated_data, mock_cleaner_config):
    """Ensures raw input dictionaries and DataFrames are never mutated during processing."""
    original_company_series = sample_validated_data["Version_1"]["companies"]["name"].copy()
    original_postings_df = sample_validated_data["Version_1"]["postings"].copy(deep=True)

    cleaner = DataCleaning(config=mock_cleaner_config)
    cleaner.initiate_data_cleaning(sample_validated_data)

    pd.testing.assert_series_equal(
        sample_validated_data["Version_1"]["companies"]["name"], original_company_series
    )
    pd.testing.assert_frame_equal(
        sample_validated_data["Version_1"]["postings"], original_postings_df
    )
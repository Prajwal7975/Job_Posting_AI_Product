

from __future__ import annotations

import json
import sys
import time

import pandas as pd
import pytest

from src.components.data_validation import (
    DataValidation,
    DataValidationSummary,
    TableValidationReport,
)
from src.configs.data_validation_config import (
    DataValidationConfig,
    ForeignKeyRule,
    TableValidationRules,
    ValueConstraint,
)
from src.configs.schema_alignment_config import SchemaAlignmentConfig, TableSchema
from src.exception import CustomException


# --------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------

@pytest.fixture
def widgets_rules() -> TableValidationRules:
    return TableValidationRules(
        unique_key_columns=("id",),
        not_null_columns=("id", "name"),
        value_constraints={
            "qty": ValueConstraint(min_value=0, max_value=1000),
            "status": ValueConstraint(allowed_values=("active", "retired")),
        },
        foreign_keys=(ForeignKeyRule("warehouse_id", "warehouses", "id"),),
    )


@pytest.fixture
def warehouses_rules() -> TableValidationRules:
    return TableValidationRules(
        unique_key_columns=("id",),
        not_null_columns=("id",),
    )


@pytest.fixture
def widgets_schema() -> TableSchema:
    return TableSchema(
        column_order=("id", "name", "qty", "status", "warehouse_id"),
        aliases={c: (c,) for c in ("id", "name", "qty", "status", "warehouse_id")},
        canonical_dtypes={"id": "Int64", "qty": "Int64"},
        required_columns=("id",),
    )


@pytest.fixture
def config(
    widgets_rules: TableValidationRules,
    warehouses_rules: TableValidationRules,
    widgets_schema: TableSchema,
) -> DataValidationConfig:
    return DataValidationConfig(
        table_rules={"widgets": widgets_rules, "warehouses": warehouses_rules},
        schema_alignment_config=SchemaAlignmentConfig(
            table_schemas={"widgets": widgets_schema}
        ),
    )


@pytest.fixture
def validator(config: DataValidationConfig) -> DataValidation:
    return DataValidation(config=config)


@pytest.fixture
def warehouses_df() -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 2]})


# --------------------------------------------------------------------
# Null violations
# --------------------------------------------------------------------

def test_null_violation_is_detected(validator: DataValidation, warehouses_df: pd.DataFrame) -> None:
    widgets = pd.DataFrame({
        "id": [1, 2], "name": [None, "b"], "qty": [1, 2],
        "status": ["active", "active"], "warehouse_id": [1, 1],
    })

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})

    assert report.null_violations == {"name": 1}


def test_no_null_violation_when_all_populated(
    validator: DataValidation, warehouses_df: pd.DataFrame
) -> None:
    widgets = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "qty": [1, 2],
        "status": ["active", "active"], "warehouse_id": [1, 1],
    })

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})

    assert report.null_violations == {}


def test_not_null_column_absent_from_dataframe_is_skipped_not_errored(
    validator: DataValidation, warehouses_df: pd.DataFrame
) -> None:
    """A configured not_null column that isn't even a column in this
    DataFrame is a Schema Alignment concern, not something Data
    Validation should crash on."""
    widgets = pd.DataFrame({"id": [1, 2], "qty": [1, 2]})  # "name" entirely absent

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})

    assert "name" not in report.null_violations


# --------------------------------------------------------------------
# Structural validation (category 1)
# --------------------------------------------------------------------

def test_structural_issue_detected_for_missing_configured_column(
    validator: DataValidation, warehouses_df: pd.DataFrame
) -> None:
    """
    If a rule references a column ("name" in not_null_columns) that isn't
    in the DataFrame at all, that must surface as an explicit structural
    issue -- not just silently vanish from every other check's output.
    """
    widgets = pd.DataFrame({"id": [1, 2], "qty": [1, 2]})  # "name" missing entirely

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})

    assert any("name" in issue for issue in report.structural_issues)
    assert not report.passed


def test_no_structural_issues_when_every_configured_column_is_present(
    validator: DataValidation, warehouses_df: pd.DataFrame
) -> None:
    widgets = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "qty": [1, 2],
        "status": ["active", "active"], "warehouse_id": [1, 1],
    })

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})

    assert report.structural_issues == ()


# --------------------------------------------------------------------
# Pass / fail semantics
# --------------------------------------------------------------------

def test_table_passes_when_no_issues_found(
    validator: DataValidation, warehouses_df: pd.DataFrame
) -> None:
    widgets = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "qty": [1, 2],
        "status": ["active", "active"], "warehouse_id": [1, 1],
    })

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})

    assert report.passed is True


def test_table_fails_when_any_issue_found(
    validator: DataValidation, warehouses_df: pd.DataFrame
) -> None:
    widgets = pd.DataFrame({
        "id": [1, 1], "name": ["a", "b"], "qty": [1, 2],  # duplicate id
        "status": ["active", "active"], "warehouse_id": [1, 1],
    })

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})

    assert report.passed is False


def test_table_with_no_rules_trivially_passes(validator: DataValidation) -> None:
    df = pd.DataFrame({"anything": [1, 2, 3]})

    report = validator.validate_table("v1", "unknown_table", df, {"unknown_table": df})

    assert report.rules_defined is False
    assert report.passed is True


def test_summary_tables_passed_failed_and_skipped_counts(
    validator: DataValidation, warehouses_df: pd.DataFrame
) -> None:
    good_widgets = pd.DataFrame({
        "id": [1], "name": ["a"], "qty": [1], "status": ["active"], "warehouse_id": [1],
    })
    bad_widgets_version = {
        "widgets": pd.DataFrame({
            "id": [1, 1], "name": ["a", "b"], "qty": [1, 2],  # duplicate -> fails
            "status": ["active", "active"], "warehouse_id": [1, 1],
        }),
        "warehouses": warehouses_df,
        "unmodeled_table": pd.DataFrame({"x": [1]}),  # no rules -> skipped
    }

    result = validator.initiate_data_validation({
        "v1": {"widgets": good_widgets, "warehouses": warehouses_df},
        "v2": bad_widgets_version,
    })

    assert result.summary.tables_passed == 3   # v1 widgets, v1 warehouses, v2 warehouses
    assert result.summary.tables_failed == 1   # v2 widgets (duplicate)
    assert result.summary.tables_skipped_no_rules == 1  # v2 unmodeled_table
    assert result.summary.passed is False


def test_summary_passed_is_true_when_all_rule_backed_tables_pass(
    validator: DataValidation, warehouses_df: pd.DataFrame
) -> None:
    widgets = pd.DataFrame({
        "id": [1], "name": ["a"], "qty": [1], "status": ["active"], "warehouse_id": [1],
    })

    result = validator.initiate_data_validation({"v1": {"widgets": widgets, "warehouses": warehouses_df}})

    assert result.summary.passed is True


# --------------------------------------------------------------------
# Duplicate keys
# --------------------------------------------------------------------

def test_duplicate_keys_are_detected_and_sampled(
    validator: DataValidation, warehouses_df: pd.DataFrame
) -> None:
    widgets = pd.DataFrame({
        "id": [1, 1, 2], "name": ["a", "a2", "b"], "qty": [1, 2, 3],
        "status": ["active", "active", "active"], "warehouse_id": [1, 1, 1],
    })

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})

    assert report.duplicate_key_row_count == 2
    assert report.duplicate_key_sample == ((1,),)


def test_no_duplicates_when_keys_are_unique(
    validator: DataValidation, warehouses_df: pd.DataFrame
) -> None:
    widgets = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "qty": [1, 2],
        "status": ["active", "active"], "warehouse_id": [1, 1],
    })

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})

    assert report.duplicate_key_row_count == 0
    assert report.duplicate_key_sample == ()


def test_unique_key_column_absent_from_dataframe_is_skipped_not_errored(
    validator: DataValidation, warehouses_df: pd.DataFrame
) -> None:
    widgets = pd.DataFrame({"name": ["a", "b"], "qty": [1, 2]})  # "id" entirely absent

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})

    assert report.duplicate_key_row_count == 0
    assert report.duplicate_key_sample == ()


# --------------------------------------------------------------------
# Invalid values (value constraints)
# --------------------------------------------------------------------

def test_out_of_range_value_is_detected(
    validator: DataValidation, warehouses_df: pd.DataFrame
) -> None:
    widgets = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "qty": [-5, 2000],  # both out of [0, 1000]
        "status": ["active", "active"], "warehouse_id": [1, 1],
    })

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})

    assert report.invalid_value_counts.get("qty") == 2


def test_disallowed_enum_value_is_detected(
    validator: DataValidation, warehouses_df: pd.DataFrame
) -> None:
    widgets = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "qty": [1, 2],
        "status": ["active", "discontinued"],  # "discontinued" not allowed
        "warehouse_id": [1, 1],
    })

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})

    assert report.invalid_value_counts.get("status") == 1


def test_null_values_are_not_flagged_as_invalid(
    validator: DataValidation, warehouses_df: pd.DataFrame
) -> None:
    """Nulls are a separate concern (not_null_columns); constraints ignore them."""
    widgets = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "qty": [None, 2],
        "status": ["active", "active"], "warehouse_id": [1, 1],
    })

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})

    assert "qty" not in report.invalid_value_counts


def test_entirely_null_constrained_column_yields_no_violations(
    validator: DataValidation, warehouses_df: pd.DataFrame
) -> None:
    """When every value in a constrained column is null, there's nothing
    to range-check -- must return cleanly, not error on an empty series."""
    widgets = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "qty": [None, None],
        "status": ["active", "active"], "warehouse_id": [1, 1],
    })

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})

    assert "qty" not in report.invalid_value_counts


def test_regex_constraint_violation_is_detected(
    warehouses_rules: TableValidationRules, widgets_schema: TableSchema, warehouses_df: pd.DataFrame
) -> None:
    rules = TableValidationRules(
        unique_key_columns=("id",),
        value_constraints={"name": ValueConstraint(regex_pattern=r"^widget-\d+$")},
    )
    config = DataValidationConfig(
        table_rules={"widgets": rules, "warehouses": warehouses_rules},
        schema_alignment_config=SchemaAlignmentConfig(table_schemas={"widgets": widgets_schema}),
    )
    validator = DataValidation(config=config)

    widgets = pd.DataFrame({"id": [1, 2], "name": ["widget-1", "not-a-widget"]})

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})

    assert report.invalid_value_counts.get("name") == 1


# --------------------------------------------------------------------
# Dtype conformance
# --------------------------------------------------------------------

def test_dtype_conformance_issue_is_detected(
    validator: DataValidation, warehouses_df: pd.DataFrame
) -> None:
    widgets = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "qty": [1, "not_a_number"],
        "status": ["active", "active"], "warehouse_id": [1, 1],
    })

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})

    assert report.dtype_conformance_issues.get("qty") == 1


def test_dtype_conformance_check_never_mutates_source_column(
    validator: DataValidation, warehouses_df: pd.DataFrame
) -> None:
    widgets = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "qty": [1, "not_a_number"],
        "status": ["active", "active"], "warehouse_id": [1, 1],
    })
    original_qty = widgets["qty"].copy()

    validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})

    pd.testing.assert_series_equal(widgets["qty"], original_qty)


def test_entirely_null_dtype_checked_column_yields_no_issues(
    validator: DataValidation, warehouses_df: pd.DataFrame
) -> None:
    widgets = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "qty": [None, None],
        "status": ["active", "active"], "warehouse_id": [1, 1],
    })

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})

    assert "qty" not in report.dtype_conformance_issues


# --------------------------------------------------------------------
# Broken relationships (foreign keys)
# --------------------------------------------------------------------

def test_broken_relationship_is_detected(validator: DataValidation) -> None:
    widgets = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "qty": [1, 2],
        "status": ["active", "active"], "warehouse_id": [1, 999],  # 999 doesn't exist
    })
    warehouses = pd.DataFrame({"id": [1]})

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses})

    assert report.broken_relationship_counts.get("warehouse_id") == 1


def test_no_broken_relationship_when_all_keys_exist(validator: DataValidation) -> None:
    widgets = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "qty": [1, 2],
        "status": ["active", "active"], "warehouse_id": [1, 1],
    })
    warehouses = pd.DataFrame({"id": [1]})

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses})

    assert report.broken_relationship_counts == {}


def test_missing_referenced_table_skips_check_without_raising(validator: DataValidation) -> None:
    """If the referenced table isn't present in this version, log & skip -- don't crash."""
    widgets = pd.DataFrame({
        "id": [1], "name": ["a"], "qty": [1], "status": ["active"], "warehouse_id": [1],
    })

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets})  # no "warehouses"

    assert report.broken_relationship_counts == {}


def test_fk_column_absent_from_dataframe_is_skipped_not_errored(validator: DataValidation) -> None:
    widgets = pd.DataFrame({"id": [1], "name": ["a"], "qty": [1], "status": ["active"]})  # no warehouse_id
    warehouses = pd.DataFrame({"id": [1]})

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses})

    assert report.broken_relationship_counts == {}


def test_fk_column_entirely_null_yields_no_broken_relationships(validator: DataValidation) -> None:
    widgets = pd.DataFrame({
        "id": [1], "name": ["a"], "qty": [1], "status": ["active"], "warehouse_id": [None],
    })
    warehouses = pd.DataFrame({"id": [1]})

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses})

    assert report.broken_relationship_counts == {}


# --------------------------------------------------------------------
# No rules defined
# --------------------------------------------------------------------

def test_table_with_no_rules_is_skipped_cleanly(validator: DataValidation) -> None:
    df = pd.DataFrame({"anything": [1, 2, 3]})

    report = validator.validate_table("v1", "unknown_table", df, {"unknown_table": df})

    assert report.rules_defined is False
    assert not report.has_issues()


# --------------------------------------------------------------------
# Data is never mutated
# --------------------------------------------------------------------

def test_full_run_never_mutates_input_data(validator: DataValidation, warehouses_df: pd.DataFrame) -> None:
    widgets = pd.DataFrame({
        "id": [1, 1], "name": [None, "b"], "qty": [-1, "bad"],
        "status": ["nope", "active"], "warehouse_id": [1, 999],
    })
    all_versions = {"v1": {"widgets": widgets, "warehouses": warehouses_df}}

    result = validator.initiate_data_validation(all_versions)

    assert result.validated_data is all_versions
    assert result.validated_data["v1"]["widgets"] is widgets


# --------------------------------------------------------------------
# Report / summary serialisation
# --------------------------------------------------------------------

def test_report_to_json_round_trips(validator: DataValidation, warehouses_df: pd.DataFrame) -> None:
    widgets = pd.DataFrame({
        "id": [1, 1], "name": ["a", "b"], "qty": [1, 2],
        "status": ["active", "active"], "warehouse_id": [1, 1],
    })

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})

    parsed = json.loads(report.to_json())
    assert parsed["duplicate_key_row_count"] == 2


def test_report_as_log_dict_matches_fields(validator: DataValidation, warehouses_df: pd.DataFrame) -> None:
    widgets = pd.DataFrame({
        "id": [1, 1], "name": ["a", "b"], "qty": [1, 2],
        "status": ["active", "active"], "warehouse_id": [1, 1],
    })

    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})
    log_dict = report.as_log_dict()

    assert log_dict["table"] == "widgets"
    assert log_dict["duplicate_key_row_count"] == 2


def test_summary_to_json_round_trips(validator: DataValidation, warehouses_df: pd.DataFrame) -> None:
    widgets = pd.DataFrame({
        "id": [1], "name": ["a"], "qty": [1], "status": ["active"], "warehouse_id": [1],
    })
    result = validator.initiate_data_validation({"v1": {"widgets": widgets, "warehouses": warehouses_df}})

    parsed = json.loads(result.summary.to_json())
    assert parsed["versions_processed"] == 1


# --------------------------------------------------------------------
# Raise policy (configurable hard-fail)
# --------------------------------------------------------------------

@pytest.mark.parametrize(
    "flag_name",
    [
        "raise_on_null_violations",
        "raise_on_duplicate_keys",
        "raise_on_invalid_values",
        "raise_on_dtype_conformance_issues",
        "raise_on_broken_relationships",
        "raise_on_structural_issues",
    ],
)
def test_raise_policy_defaults_to_report_only(
    flag_name: str,
    widgets_rules: TableValidationRules,
    warehouses_rules: TableValidationRules,
    widgets_schema: TableSchema,
) -> None:
    """Every raise_on_* flag defaults to False; a bad run must not raise by default."""
    config = DataValidationConfig(
        table_rules={"widgets": widgets_rules, "warehouses": warehouses_rules},
        schema_alignment_config=SchemaAlignmentConfig(table_schemas={"widgets": widgets_schema}),
    )
    assert getattr(config, flag_name) is False


def test_raise_on_duplicate_keys_when_enabled(
    widgets_rules: TableValidationRules,
    warehouses_rules: TableValidationRules,
    widgets_schema: TableSchema,
    warehouses_df: pd.DataFrame,
) -> None:
    config = DataValidationConfig(
        table_rules={"widgets": widgets_rules, "warehouses": warehouses_rules},
        schema_alignment_config=SchemaAlignmentConfig(table_schemas={"widgets": widgets_schema}),
        raise_on_duplicate_keys=True,
    )
    validator = DataValidation(config=config)

    widgets = pd.DataFrame({
        "id": [1, 1], "name": ["a", "b"], "qty": [1, 2],
        "status": ["active", "active"], "warehouse_id": [1, 1],
    })

    with pytest.raises(CustomException):
        validator.initiate_data_validation({"v1": {"widgets": widgets, "warehouses": warehouses_df}})


@pytest.mark.parametrize(
    "flag_name, widgets_overrides",
    [
        ("raise_on_null_violations", {"name": [None, "b"]}),
        ("raise_on_invalid_values", {"qty": [-5, 2]}),
        ("raise_on_dtype_conformance_issues", {"qty": [1, "not_a_number"]}),
        ("raise_on_broken_relationships", {"warehouse_id": [999, 1]}),
    ],
)
def test_raise_on_each_policy_flag_when_enabled(
    flag_name: str,
    widgets_overrides: dict,
    widgets_rules: TableValidationRules,
    warehouses_rules: TableValidationRules,
    widgets_schema: TableSchema,
    warehouses_df: pd.DataFrame,
) -> None:
    base = {
        "id": [1, 2], "name": ["a", "b"], "qty": [1, 2],
        "status": ["active", "active"], "warehouse_id": [1, 1],
    }
    base.update(widgets_overrides)
    widgets = pd.DataFrame(base)

    config = DataValidationConfig(
        table_rules={"widgets": widgets_rules, "warehouses": warehouses_rules},
        schema_alignment_config=SchemaAlignmentConfig(table_schemas={"widgets": widgets_schema}),
        **{flag_name: True},
    )
    validator = DataValidation(config=config)

    with pytest.raises(CustomException):
        validator.initiate_data_validation({"v1": {"widgets": widgets, "warehouses": warehouses_df}})


def test_raise_on_structural_issues_when_enabled(
    widgets_rules: TableValidationRules,
    warehouses_rules: TableValidationRules,
    widgets_schema: TableSchema,
    warehouses_df: pd.DataFrame,
) -> None:
    """A configured column that's genuinely absent from the DataFrame,
    with raise_on_structural_issues enabled, must hard-fail the run."""
    config = DataValidationConfig(
        table_rules={"widgets": widgets_rules, "warehouses": warehouses_rules},
        schema_alignment_config=SchemaAlignmentConfig(table_schemas={"widgets": widgets_schema}),
        raise_on_structural_issues=True,
    )
    validator = DataValidation(config=config)

    widgets = pd.DataFrame({"id": [1], "qty": [1]})  # "name" missing entirely

    with pytest.raises(CustomException):
        validator.initiate_data_validation({"v1": {"widgets": widgets, "warehouses": warehouses_df}})


def test_does_not_raise_when_flag_disabled_despite_issues(
    validator: DataValidation, warehouses_df: pd.DataFrame
) -> None:
    widgets = pd.DataFrame({
        "id": [1, 1], "name": ["a", "b"], "qty": [1, 2],
        "status": ["active", "active"], "warehouse_id": [1, 1],
    })
    # duplicate keys exist, but raise_on_duplicate_keys defaults to False
    result = validator.initiate_data_validation({"v1": {"widgets": widgets, "warehouses": warehouses_df}})
    assert result.summary.total_duplicate_key_rows == 2  # detected, not raised


# --------------------------------------------------------------------
# Unexpected (non-CustomException) failures are wrapped, not leaked
# --------------------------------------------------------------------

def test_unexpected_error_in_validate_table_is_wrapped(
    validator: DataValidation, warehouses_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(validator, "_validate_against_rules", _boom)

    widgets = pd.DataFrame({
        "id": [1], "name": ["a"], "qty": [1], "status": ["active"], "warehouse_id": [1],
    })

    with pytest.raises(CustomException):
        validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})


def test_custom_exception_from_validate_table_is_not_double_wrapped(
    validator: DataValidation, warehouses_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = CustomException(ValueError("already wrapped"),sys )

    def _raise_custom(*args, **kwargs):
        raise original

    monkeypatch.setattr(validator, "_validate_against_rules", _raise_custom)

    widgets = pd.DataFrame({
        "id": [1], "name": ["a"], "qty": [1], "status": ["active"], "warehouse_id": [1],
    })

    with pytest.raises(CustomException) as exc_info:
        validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})

    assert exc_info.value is original


def test_unexpected_error_during_full_run_is_wrapped(
    validator: DataValidation, warehouses_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(validator, "validate_version", _boom)

    widgets = pd.DataFrame({
        "id": [1], "name": ["a"], "qty": [1], "status": ["active"], "warehouse_id": [1],
    })

    with pytest.raises(CustomException):
        validator.initiate_data_validation({"v1": {"widgets": widgets, "warehouses": warehouses_df}})


# --------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------

def test_validates_100k_rows_in_under_two_seconds(
    validator: DataValidation, warehouses_df: pd.DataFrame
) -> None:
    n = 100_000
    widgets = pd.DataFrame({
        "id": range(n), "name": ["w"] * n, "qty": [1] * n,
        "status": ["active"] * n, "warehouse_id": [1] * n,
    })

    start = time.perf_counter()
    report = validator.validate_table("v1", "widgets", widgets, {"widgets": widgets, "warehouses": warehouses_df})
    elapsed = time.perf_counter() - start

    assert report.row_count == n
    assert elapsed < 2.0, f"Validation of {n} rows took {elapsed:.2f}s (budget: 2.0s)"


# --------------------------------------------------------------------
# Production config sanity checks
# --------------------------------------------------------------------

def test_default_config_defines_all_expected_tables() -> None:
    config = DataValidationConfig()

    expected_tables = {
        "postings", "companies", "industries", "job_industries",
        "benefits", "company_specialities", "employee_counts",
        "job_skills", "skills", "salaries",
    }

    assert set(config.table_rules.keys()) == expected_tables


def test_default_config_foreign_keys_reference_declared_tables() -> None:
    """Every FK rule's references_table must itself be a known canonical table."""
    config = DataValidationConfig()
    known_tables = set(config.table_rules.keys())

    for table_name, rules in config.table_rules.items():
        for fk in rules.foreign_keys:
            assert fk.references_table in known_tables, (
                f"{table_name}: FK references unknown table '{fk.references_table}'"
            )


def test_default_config_validates_a_real_postings_frame() -> None:
    """End-to-end smoke test against the real production 'postings' rules."""
    validator = DataValidation()  # default config, not the test fixture

    postings = pd.DataFrame({
        "job_id": [1, 2], "company_id": [10, 999], "title": ["Engineer", "Analyst"],
        "min_salary": [50000, -1],
    })
    companies = pd.DataFrame({"company_id": [10], "name": ["Acme"]})

    report = validator.validate_table(
        "Version_1", "postings", postings, {"postings": postings, "companies": companies}
    )

    assert report.broken_relationship_counts.get("company_id") == 1  # 999 is orphaned
    assert report.invalid_value_counts.get("min_salary") == 1        # -1 is negative
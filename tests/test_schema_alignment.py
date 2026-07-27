from __future__ import annotations

import json
import sys
import time

import pandas as pd
import pytest

from src.components.schema_alignment import (
    AlignmentSummary,
    SchemaAlignment,
    TableAlignmentReport,
)
from src.configs.schema_alignment_config import SchemaAlignmentConfig, TableSchema
from src.exception import CustomException


# --------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------

@pytest.fixture
def widgets_schema() -> TableSchema:
    """A small, representative canonical schema used across tests."""
    return TableSchema(
        column_order=("id", "name", "qty"),
        aliases={
            "id": ("id", "widget_id"),
            "name": ("name",),
            "qty": ("qty", "quantity"),
        },
        canonical_dtypes={
            "id": "Int64",
            "name": "string",
            "qty": "Int64",
        },
        required_columns=("id",),
    )


@pytest.fixture
def config(widgets_schema: TableSchema) -> SchemaAlignmentConfig:
    return SchemaAlignmentConfig(
        table_schemas={"widgets": widgets_schema},
        preserve_unmapped_columns=True,
    )


@pytest.fixture
def aligner(config: SchemaAlignmentConfig) -> SchemaAlignment:
    return SchemaAlignment(config=config)


# --------------------------------------------------------------------
# Alias resolution
# --------------------------------------------------------------------

def test_alias_resolution_renames_column(aligner: SchemaAlignment) -> None:
    """A column under an accepted alias must be resolved to its canonical name."""
    df = pd.DataFrame({"widget_id": [1, 2], "name": ["a", "b"], "qty": [5, 6]})

    aligned_df, report = aligner.align_table("v1", "widgets", df)

    assert list(aligned_df.columns) == ["id", "name", "qty"]
    assert report.renamed_columns == {"id": "widget_id"}
    assert list(aligned_df["id"]) == [1, 2]


def test_case_insensitive_alias_resolution(aligner: SchemaAlignment) -> None:
    """Column matching must ignore case (e.g. 'ID' should match 'id')."""
    df = pd.DataFrame({"ID": [1, 2], "NAME": ["a", "b"], "QTY": [5, 6]})

    aligned_df, report = aligner.align_table("v1", "widgets", df)

    assert list(aligned_df.columns) == ["id", "name", "qty"]
    assert list(aligned_df["id"]) == [1, 2]
    # the source names were "ID"/"NAME"/"QTY", which differ from the
    # canonical names by case, so they should show up as renames
    assert report.renamed_columns == {"id": "ID", "name": "NAME", "qty": "QTY"}


# --------------------------------------------------------------------
# Unmapped / unknown columns
# --------------------------------------------------------------------

def test_preserve_unknown_columns_by_default(aligner: SchemaAlignment) -> None:
    df = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "qty": [5, 6],
        "warehouse_notes": ["x", "y"],
    })

    aligned_df, report = aligner.align_table("v1", "widgets", df)

    assert "unmapped__warehouse_notes" in aligned_df.columns
    assert list(aligned_df["unmapped__warehouse_notes"]) == ["x", "y"]
    assert report.unmapped_source_columns == ("warehouse_notes",)
    assert report.unmapped_columns_preserved is True


def test_drop_unknown_columns_when_configured(widgets_schema: TableSchema) -> None:
    config = SchemaAlignmentConfig(
        table_schemas={"widgets": widgets_schema},
        preserve_unmapped_columns=False,
    )
    aligner = SchemaAlignment(config=config)

    df = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "qty": [5, 6],
        "warehouse_notes": ["x", "y"],
    })

    aligned_df, report = aligner.align_table("v1", "widgets", df)

    assert "warehouse_notes" not in aligned_df.columns
    assert not any(col.startswith("unmapped__") for col in aligned_df.columns)
    # still reported, even though dropped
    assert report.unmapped_source_columns == ("warehouse_notes",)
    assert report.unmapped_columns_preserved is False


# --------------------------------------------------------------------
# Missing vs. empty required columns
# --------------------------------------------------------------------

def test_missing_required_column_raises(aligner: SchemaAlignment) -> None:
    """A required column with no resolvable alias at all must raise."""
    df = pd.DataFrame({"name": ["a", "b"], "qty": [5, 6]})  # no id / widget_id

    with pytest.raises(CustomException):
        aligner.align_table("v1", "widgets", df)


def test_empty_required_column_warns_but_does_not_raise(aligner: SchemaAlignment) -> None:

    df = pd.DataFrame({"id": [pd.NA, pd.NA], "name": ["a", "b"], "qty": [5, 6]})

    aligned_df, report = aligner.align_table("v1", "widgets", df)  # should not raise

    assert report.empty_required_columns == ("id",)
    assert report.missing_columns == ()  # structurally present, just empty


def test_missing_optional_column_is_created_empty_and_typed(aligner: SchemaAlignment) -> None:
    df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})  # qty missing entirely

    aligned_df, report = aligner.align_table("v1", "widgets", df)

    assert "qty" in aligned_df.columns
    assert aligned_df["qty"].isna().all()
    assert str(aligned_df["qty"].dtype) == "Int64"
    assert report.missing_columns == ("qty",)


def test_missing_column_with_no_declared_dtype_falls_back_to_untyped_na(
    widgets_schema: TableSchema,
) -> None:

    untyped_schema = TableSchema(
        column_order=("id", "name", "notes"),
        aliases={"id": ("id",), "name": ("name",), "notes": ("notes",)},
        canonical_dtypes={"id": "Int64", "name": "string"},  # "notes" intentionally omitted
        required_columns=("id",),
    )
    config = SchemaAlignmentConfig(table_schemas={"widgets": untyped_schema})
    aligner = SchemaAlignment(config=config)

    df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})  # notes missing

    aligned_df, report = aligner.align_table("v1", "widgets", df)

    assert "notes" in aligned_df.columns
    assert aligned_df["notes"].isna().all()
    assert report.missing_columns == ("notes",)


# --------------------------------------------------------------------
# No schema defined for a table
# --------------------------------------------------------------------

def test_table_with_no_defined_schema_passes_through_unaligned(aligner: SchemaAlignment) -> None:
    df = pd.DataFrame({"anything": [1, 2, 3]})

    aligned_df, report = aligner.align_table("v1", "unknown_table", df)

    assert list(aligned_df.columns) == ["anything"]  # untouched
    assert report.unmapped_source_columns == ("anything",)
    assert report.missing_columns == ()
    assert not report.has_structural_issues()


# --------------------------------------------------------------------
# Report creation / serialisation
# --------------------------------------------------------------------

def test_report_fields_are_populated_correctly(aligner: SchemaAlignment) -> None:
    df = pd.DataFrame({"widget_id": [1, 2, 3], "name": ["a", "b", "c"]})

    aligned_df, report = aligner.align_table("v1", "widgets", df)

    assert isinstance(report, TableAlignmentReport)
    assert report.version == "v1"
    assert report.table == "widgets"
    assert report.row_count == 3
    assert report.column_count == len(aligned_df.columns)


def test_report_as_log_dict_matches_fields(aligner: SchemaAlignment) -> None:
    df = pd.DataFrame({"widget_id": [1], "name": ["a"], "qty": [1]})

    _, report = aligner.align_table("v1", "widgets", df)
    log_dict = report.as_log_dict()

    assert log_dict["version"] == "v1"
    assert log_dict["table"] == "widgets"
    assert log_dict["renamed_columns"] == {"id": "widget_id"}


def test_report_to_json_round_trips(aligner: SchemaAlignment) -> None:
    df = pd.DataFrame({"widget_id": [1, 2], "name": ["a", "b"], "qty": [1, 2]})

    _, report = aligner.align_table("v1", "widgets", df)

    parsed = json.loads(report.to_json())
    assert parsed["version"] == "v1"
    assert parsed["table"] == "widgets"
    assert parsed["renamed_columns"] == {"id": "widget_id"}


# --------------------------------------------------------------------
# Multi-version orchestration + summary
# --------------------------------------------------------------------

def test_multiple_versions_are_all_aligned(aligner: SchemaAlignment) -> None:
    all_versions = {
        "Version_1": {
            "widgets": pd.DataFrame({"widget_id": [1], "name": ["a"], "qty": [1]}),
        },
        "Version_2": {
            "widgets": pd.DataFrame({"id": [2], "name": ["b"], "qty": [2]}),
        },
    }

    result = aligner.initiate_schema_alignment(all_versions)

    assert set(result.aligned_data.keys()) == {"Version_1", "Version_2"}
    assert len(result.reports) == 2
    assert isinstance(result.summary, AlignmentSummary)
    assert result.summary.versions_processed == 2
    assert result.summary.tables_processed == 2
    assert result.summary.columns_renamed == 1  # only Version_1's widget_id -> id
    assert result.summary.execution_time_seconds >= 0


def test_summary_counts_missing_and_preserved_columns(aligner: SchemaAlignment) -> None:
    all_versions = {
        "Version_1": {
            "widgets": pd.DataFrame({
                "id": [1], "name": ["a"],  # qty missing
                "extra_col": ["z"],         # unmapped, preserved
            }),
        },
    }

    result = aligner.initiate_schema_alignment(all_versions)

    assert result.summary.columns_missing == 1        # qty
    assert result.summary.columns_preserved == 1       # extra_col
    assert result.summary.columns_dropped == 0
    assert result.summary.tables_with_structural_issues == 1  # has a missing column
    assert result.summary.tables_with_empty_required == 0


def test_summary_to_json_round_trips(aligner: SchemaAlignment) -> None:
    all_versions = {
        "Version_1": {"widgets": pd.DataFrame({"id": [1], "name": ["a"], "qty": [1]})},
    }

    result = aligner.initiate_schema_alignment(all_versions)

    parsed = json.loads(result.summary.to_json())
    assert parsed["versions_processed"] == 1
    assert parsed["tables_processed"] == 1


# --------------------------------------------------------------------
# Unexpected (non-CustomException) failures are wrapped, not leaked
# --------------------------------------------------------------------

def test_unexpected_error_in_align_table_is_wrapped_in_custom_exception(
    aligner: SchemaAlignment, monkeypatch: pytest.MonkeyPatch
) -> None:

    def _boom(*args, **kwargs):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(aligner, "_align_against_schema", _boom)

    df = pd.DataFrame({"id": [1], "name": ["a"], "qty": [1]})

    with pytest.raises(CustomException):
        aligner.align_table("v1", "widgets", df)


def test_unexpected_error_during_full_run_is_wrapped_in_custom_exception(
    aligner: SchemaAlignment, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(aligner, "align_version", _boom)

    with pytest.raises(CustomException):
        aligner.initiate_schema_alignment({
            "v1": {"widgets": pd.DataFrame({"id": [1], "name": ["a"], "qty": [1]})}
        })


def test_custom_exception_from_align_table_is_not_double_wrapped(
    aligner: SchemaAlignment, monkeypatch: pytest.MonkeyPatch
) -> None:
    
    original = CustomException(ValueError("already wrapped"), sys)

    def _raise_custom(*args, **kwargs):
        raise original

    monkeypatch.setattr(aligner, "_align_against_schema", _raise_custom)

    df = pd.DataFrame({"id": [1], "name": ["a"], "qty": [1]})

    with pytest.raises(CustomException) as exc_info:
        aligner.align_table("v1", "widgets", df)

    assert exc_info.value is original  # same instance, not re-wrapped


def test_custom_exception_during_full_run_is_not_double_wrapped(
    aligner: SchemaAlignment, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = CustomException(ValueError("already wrapped"), sys)

    def _raise_custom(*args, **kwargs):
        raise original

    monkeypatch.setattr(aligner, "align_version", _raise_custom)

    with pytest.raises(CustomException) as exc_info:
        aligner.initiate_schema_alignment({
            "v1": {"widgets": pd.DataFrame({"id": [1], "name": ["a"], "qty": [1]})}
        })

    assert exc_info.value is original


# --------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------

def test_aligns_100k_rows_in_under_two_seconds(aligner: SchemaAlignment) -> None:

    n = 100_000
    df = pd.DataFrame({
        "widget_id": range(n),
        "name": ["widget"] * n,
        "quantity": [1] * n,
        "unmapped_col": ["x"] * n,
    })

    start = time.perf_counter()
    aligned_df, report = aligner.align_table("v1", "widgets", df)
    elapsed = time.perf_counter() - start

    assert len(aligned_df) == n
    assert elapsed < 2.0, f"Alignment of {n} rows took {elapsed:.2f}s (budget: 2.0s)"


# --------------------------------------------------------------------
# Production config sanity checks
#
# The tests above deliberately use a small hand-built "widgets" schema to
# test the *mechanism*. These tests instead sanity-check the *actual*
# default config in schema_alignment_config.py, so a typo or structural
# mistake in the real 10-table schema doesn't slip through unnoticed.
# --------------------------------------------------------------------

def test_default_config_defines_all_expected_tables() -> None:
    config = SchemaAlignmentConfig()

    expected_tables = {
        "postings", "companies", "industries", "job_industries",
        "benefits", "company_specialities", "employee_counts",
        "job_skills", "skills", "salaries",
    }

    assert set(config.table_schemas.keys()) == expected_tables


def test_default_config_every_table_has_consistent_schema() -> None:
    """Every required column must appear in column_order, and every
    canonical_dtypes key must be a real canonical column."""
    config = SchemaAlignmentConfig()

    for table_name, schema in config.table_schemas.items():
        for required_col in schema.required_columns:
            assert required_col in schema.column_order, (
                f"{table_name}: required column '{required_col}' not in column_order"
            )
        for dtype_col in schema.canonical_dtypes:
            assert dtype_col in schema.column_order, (
                f"{table_name}: canonical_dtypes key '{dtype_col}' not in column_order"
            )


def test_default_config_aligns_a_real_postings_frame() -> None:
    """End-to-end smoke test against the real production 'postings' schema."""
    aligner = SchemaAlignment()  # default config, not the test fixture

    df = pd.DataFrame({
        "id": [101, 102],           # alias for job_id
        "company_id": [1, 2],
        "title": ["Engineer", "Analyst"],
    })

    aligned_df, report = aligner.align_table("Version_1", "postings", df)

    assert "job_id" in aligned_df.columns
    assert report.renamed_columns == {"job_id": "id"}
    # job_id (the only required column) resolved fine -- the many
    # `missing_columns` here are the OPTIONAL columns this deliberately
    # partial test frame doesn't supply, which is expected, not a failure.
    assert "job_id" not in report.missing_columns
    assert "job_id" not in report.empty_required_columns
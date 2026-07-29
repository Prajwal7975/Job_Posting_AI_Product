from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.configs.data_validation_config import (
    DataValidationConfig,
    ForeignKeyRule,
    TableValidationRules,
    ValueConstraint,
)
from src.exception import CustomException
from src.logger import logging


@dataclass(frozen=True)
class TableValidationReport:
    """
    Structured record of every issue found while validating one table in
    one dataset version. Counts, not fixes -- see the module docstring.
    """

    version: str
    table: str
    row_count: int
    rules_defined: bool
    structural_issues: Tuple[str, ...] = field(default_factory=tuple)
    null_violations: Dict[str, int] = field(default_factory=dict)
    duplicate_key_row_count: int = 0
    duplicate_key_sample: Tuple[Tuple[object, ...], ...] = field(default_factory=tuple)
    invalid_value_counts: Dict[str, int] = field(default_factory=dict)
    dtype_conformance_issues: Dict[str, int] = field(default_factory=dict)
    broken_relationship_counts: Dict[str, int] = field(default_factory=dict)

    def has_issues(self) -> bool:
        return bool(
            self.structural_issues
            or self.null_violations
            or self.duplicate_key_row_count
            or self.invalid_value_counts
            or self.dtype_conformance_issues
            or self.broken_relationship_counts
        )

    @property
    def passed(self) -> bool:
        return not self.has_issues()

    def as_log_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass(frozen=True)
class DataValidationSummary:


    versions_processed: int
    tables_processed: int
    tables_passed: int
    tables_failed: int
    tables_skipped_no_rules: int
    total_structural_issues: int
    total_null_violations: int
    total_duplicate_key_rows: int
    total_invalid_values: int
    total_dtype_conformance_issues: int
    total_broken_relationships: int
    execution_time_seconds: float

    @property
    def passed(self) -> bool:
        """True only if every table that had rules defined also passed."""
        return self.tables_failed == 0

    def as_log_dict(self) -> dict:
        log_dict = asdict(self)
        log_dict["passed"] = self.passed
        return log_dict

    def to_json(self) -> str:
        return json.dumps(self.as_log_dict(), indent=2)


@dataclass(frozen=True)
class DataValidationResult:

    validated_data: Dict[str, Dict[str, pd.DataFrame]]
    reports: Tuple[TableValidationReport, ...]
    summary: DataValidationSummary


def _build_summary(
    reports: Tuple[TableValidationReport, ...],
    versions_processed: int,
    execution_time_seconds: float,
) -> DataValidationSummary:
    rules_defined_reports = [r for r in reports if r.rules_defined]

    return DataValidationSummary(
        versions_processed=versions_processed,
        tables_processed=len(reports),
        tables_passed=sum(1 for r in rules_defined_reports if r.passed),
        tables_failed=sum(1 for r in rules_defined_reports if not r.passed),
        tables_skipped_no_rules=sum(1 for r in reports if not r.rules_defined),
        total_structural_issues=sum(len(r.structural_issues) for r in reports),
        total_null_violations=sum(sum(r.null_violations.values()) for r in reports),
        total_duplicate_key_rows=sum(r.duplicate_key_row_count for r in reports),
        total_invalid_values=sum(sum(r.invalid_value_counts.values()) for r in reports),
        total_dtype_conformance_issues=sum(
            sum(r.dtype_conformance_issues.values()) for r in reports
        ),
        total_broken_relationships=sum(
            sum(r.broken_relationship_counts.values()) for r in reports
        ),
        execution_time_seconds=round(execution_time_seconds, 4),
    )


class DataValidation:
    """
    Validates the aligned, per-version tables produced by
    SchemaAlignment.initiate_schema_alignment() against configured
    business rules, without modifying the data.
    """

    def __init__(self, config: Optional[DataValidationConfig] = None) -> None:
        self.config = config or DataValidationConfig()

    # ------------------------------------------------------------------
    # Individual check implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _find_structural_issues(
        df: pd.DataFrame,
        rules: TableValidationRules,
        canonical_dtype_columns: Tuple[str, ...],
    ) -> Tuple[str, ...]:
        """
        Category 1 -- Structural Validation. Every other check silently
        skips a configured column that isn't in the DataFrame (so one
        missing column can't crash the whole run) -- but silently
        skipping everywhere would hide a real config/data mismatch. This
        collects those gaps into one explicit, visible signal instead.
        """
        issues: List[str] = []

        def _check(columns: Tuple[str, ...], rule_kind: str) -> None:
            for column in columns:
                if column not in df.columns:
                    issues.append(f"{rule_kind}: configured column '{column}' not found")

        _check(rules.not_null_columns, "not_null_columns")
        _check(rules.unique_key_columns, "unique_key_columns")
        _check(tuple(rules.value_constraints.keys()), "value_constraints")
        _check(tuple(fk.column for fk in rules.foreign_keys), "foreign_keys")
        _check(canonical_dtype_columns, "canonical_dtypes")

        return tuple(issues)

    @staticmethod
    def _count_nulls(df: pd.DataFrame, not_null_columns: Tuple[str, ...]) -> Dict[str, int]:
        violations: Dict[str, int] = {}
        for column in not_null_columns:
            if column not in df.columns:
                continue
            null_count = int(df[column].isna().sum())
            if null_count > 0:
                violations[column] = null_count
        return violations

    @staticmethod
    def _find_duplicate_keys(
        df: pd.DataFrame,
        unique_key_columns: Tuple[str, ...],
        sample_size: int = 5,
    ) -> Tuple[int, Tuple[Tuple[object, ...], ...]]:
        missing_key_columns = [c for c in unique_key_columns if c not in df.columns]
        if not unique_key_columns or missing_key_columns:
            return 0, ()

        duplicate_mask = df.duplicated(subset=list(unique_key_columns), keep=False)
        duplicate_rows = df.loc[duplicate_mask, list(unique_key_columns)]

        if duplicate_rows.empty:
            return 0, ()

        sample = (
            duplicate_rows.drop_duplicates()
            .head(sample_size)
            .apply(tuple, axis=1)
            .tolist()
        )
        return int(duplicate_mask.sum()), tuple(sample)

    @staticmethod
    def _count_value_constraint_violations(
        series: pd.Series, constraint: ValueConstraint
    ) -> int:
        non_null = series.dropna()
        if non_null.empty:
            return 0

        violation_mask = pd.Series(False, index=non_null.index)

        if constraint.allowed_values is not None:
            violation_mask |= ~non_null.isin(constraint.allowed_values)

        if constraint.min_value is not None or constraint.max_value is not None:
            numeric = pd.to_numeric(non_null, errors="coerce")
            # values that fail to parse as numeric are a dtype-conformance
            # concern, not a range concern -- exclude them here so they
            # aren't double-counted against this check.
            in_range_checkable = numeric.notna()
            if constraint.min_value is not None:
                violation_mask |= in_range_checkable & (numeric < constraint.min_value)
            if constraint.max_value is not None:
                violation_mask |= in_range_checkable & (numeric > constraint.max_value)

        if constraint.regex_pattern is not None:
            pattern = re.compile(constraint.regex_pattern)
            violation_mask |= ~non_null.astype(str).str.match(pattern)

        return int(violation_mask.sum())

    @staticmethod
    def _count_dtype_conformance_issues(
        series: pd.Series, canonical_dtype: str
    ) -> int:
        """
        Count non-null values that fail to convert to `canonical_dtype`.
        The conversion result is used only for counting -- it is never
        written back to the source DataFrame (see module docstring).
        """
        non_null = series.dropna()
        if non_null.empty:
            return 0

        if canonical_dtype in ("Int64", "float64", "float32", "int64", "int32"):
            converted = pd.to_numeric(non_null, errors="coerce")
            return int(converted.isna().sum())

        # "string" and any other declared dtype: every non-null value is
        # representable as a string, so there is nothing to conform-check.
        return 0

    @staticmethod
    def _count_broken_relationships(
        df: pd.DataFrame,
        rule: ForeignKeyRule,
        version_data: Dict[str, pd.DataFrame],
    ) -> Optional[int]:
        if rule.column not in df.columns:
            return None

        referenced_df = version_data.get(rule.references_table)
        if referenced_df is None or rule.references_column not in referenced_df.columns:
            logging.warning(
                "Cannot check foreign key '%s' -> '%s.%s': referenced table/column "
                "not present in this dataset version.",
                rule.column, rule.references_table, rule.references_column,
            )
            return None

        non_null_values = df[rule.column].dropna()
        if non_null_values.empty:
            return 0

        referenced_values = set(referenced_df[rule.references_column].dropna())
        broken_mask = ~non_null_values.isin(referenced_values)
        return int(broken_mask.sum())

    # ------------------------------------------------------------------
    # Table-level validation
    # ------------------------------------------------------------------

    def validate_table(
        self,
        version_name: str,
        table_name: str,
        df: pd.DataFrame,
        version_data: Dict[str, pd.DataFrame],
    ) -> TableValidationReport:
        """
        Validate a single table against its configured rules. `version_data`
        (every table in this dataset version) is passed through so
        foreign-key checks can look up referenced tables.
        """

        rules = self.config.table_rules.get(table_name)

        if rules is None:
            logging.warning(
                "[%s/%s] No validation rules defined; skipping.",
                version_name, table_name,
            )
            return TableValidationReport(
                version=version_name,
                table=table_name,
                row_count=len(df),
                rules_defined=False,
            )

        try:
            report = self._validate_against_rules(
                version_name, table_name, df, rules, version_data
            )
        except CustomException:
            raise
        except Exception as e:
            raise CustomException(e, sys) from e

        self._log_report(report)
        return report

    def _validate_against_rules(
        self,
        version_name: str,
        table_name: str,
        df: pd.DataFrame,
        rules: TableValidationRules,
        version_data: Dict[str, pd.DataFrame],
    ) -> TableValidationReport:

        table_schema = self.config.schema_alignment_config.table_schemas.get(table_name)
        canonical_dtype_columns: Tuple[str, ...] = (
            tuple(table_schema.canonical_dtypes.keys()) if table_schema is not None else ()
        )

        structural_issues = self._find_structural_issues(df, rules, canonical_dtype_columns)

        null_violations = self._count_nulls(df, rules.not_null_columns)

        duplicate_count, duplicate_sample = self._find_duplicate_keys(
            df, rules.unique_key_columns
        )

        invalid_value_counts: Dict[str, int] = {}
        for column, constraint in rules.value_constraints.items():
            if column not in df.columns:
                continue
            count = self._count_value_constraint_violations(df[column], constraint)
            if count > 0:
                invalid_value_counts[column] = count

        dtype_conformance_issues: Dict[str, int] = {}
        if table_schema is not None:
            for column, dtype in table_schema.canonical_dtypes.items():
                if column not in df.columns:
                    continue
                count = self._count_dtype_conformance_issues(df[column], dtype)
                if count > 0:
                    dtype_conformance_issues[column] = count

        broken_relationship_counts: Dict[str, int] = {}
        for fk_rule in rules.foreign_keys:
            count = self._count_broken_relationships(df, fk_rule, version_data)
            if count:
                broken_relationship_counts[fk_rule.column] = count

        return TableValidationReport(
            version=version_name,
            table=table_name,
            row_count=len(df),
            rules_defined=True,
            structural_issues=structural_issues,
            null_violations=null_violations,
            duplicate_key_row_count=duplicate_count,
            duplicate_key_sample=duplicate_sample,
            invalid_value_counts=invalid_value_counts,
            dtype_conformance_issues=dtype_conformance_issues,
            broken_relationship_counts=broken_relationship_counts,
        )

    @staticmethod
    def _log_report(report: TableValidationReport) -> None:
        if report.structural_issues:
            logging.warning(
                "[%s/%s] Structural issues (configured column not found): %s",
                report.version, report.table, report.structural_issues,
            )

        if report.null_violations:
            logging.warning(
                "[%s/%s] Null violations: %s",
                report.version, report.table, report.null_violations,
            )

        if report.duplicate_key_row_count:
            logging.warning(
                "[%s/%s] %d row(s) share a duplicate unique key. Sample: %s",
                report.version, report.table,
                report.duplicate_key_row_count, report.duplicate_key_sample,
            )

        if report.invalid_value_counts:
            logging.warning(
                "[%s/%s] Invalid values: %s",
                report.version, report.table, report.invalid_value_counts,
            )

        if report.dtype_conformance_issues:
            logging.warning(
                "[%s/%s] Values not conforming to canonical dtype: %s",
                report.version, report.table, report.dtype_conformance_issues,
            )

        if report.broken_relationship_counts:
            logging.warning(
                "[%s/%s] Broken relationships (orphan foreign keys): %s",
                report.version, report.table, report.broken_relationship_counts,
            )

        logging.info(
            "[%s/%s] Validation complete -> %d row(s), passed: %s",
            report.version, report.table, report.row_count, report.passed,
        )

    # ------------------------------------------------------------------
    # Version- and pipeline-level orchestration
    # ------------------------------------------------------------------

    def validate_version(
        self,
        version_name: str,
        version_data: Dict[str, pd.DataFrame],
    ) -> List[TableValidationReport]:
        """Validate every table belonging to a single dataset version."""

        return [
            self.validate_table(version_name, table_name, df, version_data)
            for table_name, df in version_data.items()
        ]

    def initiate_data_validation(
        self,
        aligned_data: Dict[str, Dict[str, pd.DataFrame]],
    ) -> DataValidationResult:
        """
        Validate every table across every dataset version returned by
        SchemaAlignment.initiate_schema_alignment(). Never modifies
        `aligned_data` -- it is returned unchanged as `validated_data`.
        """

        logging.info("=" * 70)
        logging.info("DATA VALIDATION STARTED")
        logging.info("=" * 70)

        start_time = time.perf_counter()

        try:
            all_reports: List[TableValidationReport] = []

            for version_name, version_data in aligned_data.items():
                logging.info("Validating %s", version_name)
                all_reports.extend(self.validate_version(version_name, version_data))

            summary = _build_summary(
                reports=tuple(all_reports),
                versions_processed=len(aligned_data),
                execution_time_seconds=time.perf_counter() - start_time,
            )

            logging.info("=" * 70)
            logging.info("Validation summary: %s", summary.as_log_dict())
            logging.info("DATA VALIDATION COMPLETED")
            logging.info("=" * 70)

            self._enforce_raise_policy(summary)

            return DataValidationResult(
                validated_data=aligned_data,
                reports=tuple(all_reports),
                summary=summary,
            )

        except CustomException:
            raise
        except Exception as e:
            logging.exception("Data Validation Failed")
            raise CustomException(e, sys) from e

    def _enforce_raise_policy(self, summary: DataValidationSummary) -> None:
        """
        Translate the config's `raise_on_*` flags into a single combined
        failure, so a caller sees every violated policy at once instead
        of only the first one checked.
        """

        violations: List[str] = []

        if self.config.raise_on_structural_issues and summary.total_structural_issues:
            violations.append(f"{summary.total_structural_issues} structural issue(s)")

        if self.config.raise_on_null_violations and summary.total_null_violations:
            violations.append(f"{summary.total_null_violations} null violation(s)")

        if self.config.raise_on_duplicate_keys and summary.total_duplicate_key_rows:
            violations.append(f"{summary.total_duplicate_key_rows} duplicate key row(s)")

        if self.config.raise_on_invalid_values and summary.total_invalid_values:
            violations.append(f"{summary.total_invalid_values} invalid value(s)")

        if (
            self.config.raise_on_dtype_conformance_issues
            and summary.total_dtype_conformance_issues
        ):
            violations.append(
                f"{summary.total_dtype_conformance_issues} dtype conformance issue(s)"
            )

        if (
            self.config.raise_on_broken_relationships
            and summary.total_broken_relationships
        ):
            violations.append(
                f"{summary.total_broken_relationships} broken relationship(s)"
            )

        if violations:
            raise CustomException(
                ValueError(
                    "Data Validation failed policy checks: " + "; ".join(violations)
                ),
                sys,
            )
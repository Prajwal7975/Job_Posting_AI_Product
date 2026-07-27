from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.configs.schema_alignment_config import SchemaAlignmentConfig, TableSchema


@dataclass(frozen=True)
class TableAlignmentReport:

    version: str
    table: str
    row_count: int
    column_count: int
    renamed_columns: Dict[str, str]
    missing_columns: Tuple[str, ...]
    empty_required_columns: Tuple[str, ...]
    unmapped_source_columns: Tuple[str, ...]
    unmapped_columns_preserved: bool

    def has_structural_issues(self) -> bool:
        """True if any canonical column could not be resolved at all."""
        return bool(self.missing_columns)

    def as_log_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass(frozen=True)
class AlignmentSummary:

    versions_processed: int
    tables_processed: int
    columns_renamed: int
    columns_missing: int
    columns_preserved: int
    columns_dropped: int
    tables_with_empty_required: int
    tables_with_structural_issues: int
    execution_time_seconds: float

    def as_log_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _build_alignment_summary(
    reports: Tuple["TableAlignmentReport", ...],
    versions_processed: int,
    execution_time_seconds: float,) -> AlignmentSummary:
    """Aggregate a run's individual table reports into one summary."""

    columns_preserved = sum(
        len(r.unmapped_source_columns) for r in reports if r.unmapped_columns_preserved
    )
    columns_dropped = sum(
        len(r.unmapped_source_columns) for r in reports if not r.unmapped_columns_preserved
    )

    return AlignmentSummary(
        versions_processed=versions_processed,
        tables_processed=len(reports),
        columns_renamed=sum(len(r.renamed_columns) for r in reports),
        columns_missing=sum(len(r.missing_columns) for r in reports),
        columns_preserved=columns_preserved,
        columns_dropped=columns_dropped,
        tables_with_empty_required=sum(1 for r in reports if r.empty_required_columns),
        tables_with_structural_issues=sum(1 for r in reports if r.has_structural_issues()),
        execution_time_seconds=round(execution_time_seconds, 4),
    )


@dataclass(frozen=True)
class SchemaAlignmentResult:
    """Full output of the Schema Alignment stage."""

    aligned_data: Dict[str, Dict[str, pd.DataFrame]]
    reports: Tuple[TableAlignmentReport, ...]
    summary: AlignmentSummary


_NON_NULLABLE_NA_FILL: Dict[str, object] = {
    "float64": float("nan"),
    "float32": float("nan"),
}


class SchemaAlignment:

    def __init__(self, config: Optional[SchemaAlignmentConfig] = None) -> None:
        self.config = config or SchemaAlignmentConfig()

    # ------------------------------------------------------------------
    # Column resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_column(
        df: pd.DataFrame,
        accepted_names: Tuple[str, ...],
    ) -> Optional[str]:

        lower_to_actual = {col.lower(): col for col in df.columns}

        for candidate in accepted_names:
            actual = lower_to_actual.get(candidate.lower())
            if actual is not None:
                return actual

        return None

    @staticmethod
    def _empty_column(length: int, index: pd.Index, dtype: Optional[str]) -> pd.Series:

        if dtype is None:
            return pd.Series([None] * length, index=index)

        fill_value = _NON_NULLABLE_NA_FILL.get(dtype, pd.NA)
        return pd.Series(fill_value, index=index, dtype=dtype)

    # ------------------------------------------------------------------
    # Table-level alignment
    # ------------------------------------------------------------------

    def align_table(
        self,
        version_name: str,
        table_name: str,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, TableAlignmentReport]:

        schema = self.config.table_schemas.get(table_name)

        if schema is None:
            logging.warning(
                "[%s/%s] No canonical schema defined; passing through unaligned.",
                version_name, table_name,
            )
            report = TableAlignmentReport(
                version=version_name,
                table=table_name,
                row_count=len(df),
                column_count=df.shape[1],
                renamed_columns={},
                missing_columns=(),
                empty_required_columns=(),
                unmapped_source_columns=tuple(df.columns),
                unmapped_columns_preserved=True,
            )
            return df, report

        try:
            aligned_df, report = self._align_against_schema(
                version_name, table_name, df, schema
            )
        except CustomException:
            raise
        except Exception as e:
            raise CustomException(e, sys) from e

        self._enforce_required_columns(version_name, table_name, report, schema)
        self._log_report(report)

        return aligned_df, report

    def _align_against_schema(
        self,
        version_name: str,
        table_name: str,
        df: pd.DataFrame,
        schema: TableSchema,
    ) -> Tuple[pd.DataFrame, TableAlignmentReport]:

        resolved_columns: Dict[str, pd.Series] = {}
        renamed_columns: Dict[str, str] = {}
        missing_columns: List[str] = []
        empty_required_columns: List[str] = []
        matched_source_names: set = set()

        for canonical_name in schema.column_order:
            accepted_names = schema.aliases.get(canonical_name, (canonical_name,))
            source_name = self._resolve_column(df, accepted_names)

            if source_name is None:
                missing_columns.append(canonical_name)
                resolved_columns[canonical_name] = self._empty_column(
                    length=len(df),
                    index=df.index,
                    dtype=schema.canonical_dtypes.get(canonical_name),
                )
                continue

            matched_source_names.add(source_name)
            if source_name != canonical_name:
                renamed_columns[canonical_name] = source_name

            column = df[source_name]
            if (
                canonical_name in schema.required_columns
                and column.isna().all()
            ):
                empty_required_columns.append(canonical_name)

            resolved_columns[canonical_name] = column

        aligned_df = pd.DataFrame(resolved_columns, index=df.index)

        unmapped_source_names = tuple(
            col for col in df.columns if col not in matched_source_names
        )

        if unmapped_source_names:
            if self.config.preserve_unmapped_columns:
                prefix = self.config.unmapped_column_prefix
                for source_name in unmapped_source_names:
                    aligned_df[f"{prefix}{source_name}"] = df[source_name]
            # else: intentionally dropped; still reported below.

        report = TableAlignmentReport(
            version=version_name,
            table=table_name,
            row_count=aligned_df.shape[0],
            column_count=aligned_df.shape[1],
            renamed_columns=renamed_columns,
            missing_columns=tuple(missing_columns),
            empty_required_columns=tuple(empty_required_columns),
            unmapped_source_columns=unmapped_source_names,
            unmapped_columns_preserved=self.config.preserve_unmapped_columns,
        )

        return aligned_df, report

    @staticmethod
    def _enforce_required_columns(
        version_name: str,
        table_name: str,
        report: TableAlignmentReport,
        schema: TableSchema,
    ) -> None:
        missing_required = [
            col for col in schema.required_columns if col in report.missing_columns
        ]

        if missing_required:
            raise CustomException(
                ValueError(
                    f"[{version_name}] Table '{table_name}' is missing required "
                    f"column(s) with no resolvable alias: {missing_required}"
                ),
                sys,
            )

    @staticmethod
    def _log_report(report: TableAlignmentReport) -> None:
        if report.renamed_columns:
            logging.info(
                "[%s/%s] Renamed columns: %s",
                report.version, report.table, report.renamed_columns,
            )

        if report.missing_columns:
            logging.warning(
                "[%s/%s] Columns not found in source, created empty: %s",
                report.version, report.table, report.missing_columns,
            )

        if report.empty_required_columns:
            logging.warning(
                "[%s/%s] Required column(s) resolved but entirely null "
                "(structurally present, data-quality issue for Data "
                "Validation): %s",
                report.version, report.table, report.empty_required_columns,
            )

        if report.unmapped_source_columns:
            action = "preserved (prefixed)" if report.unmapped_columns_preserved else "dropped"
            logging.warning(
                "[%s/%s] %d unmapped source column(s) %s: %s",
                report.version, report.table,
                len(report.unmapped_source_columns), action,
                report.unmapped_source_columns,
            )

        logging.info(
            "[%s/%s] Alignment complete -> %d columns, %d rows.",
            report.version, report.table, report.column_count, report.row_count,
        )

    # ------------------------------------------------------------------
    # Version- and pipeline-level orchestration
    # ------------------------------------------------------------------

    def align_version(
        self,
        version_name: str,
        version_data: Dict[str, pd.DataFrame],
    ) -> Tuple[Dict[str, pd.DataFrame], List[TableAlignmentReport]]:
        """Align every table belonging to a single dataset version."""

        aligned_version: Dict[str, pd.DataFrame] = {}
        reports: List[TableAlignmentReport] = []

        for table_name, df in version_data.items():
            aligned_df, report = self.align_table(version_name, table_name, df)
            aligned_version[table_name] = aligned_df
            reports.append(report)

        return aligned_version, reports

    def initiate_schema_alignment(
        self,
        all_versions: Dict[str, Dict[str, pd.DataFrame]],
    ) -> SchemaAlignmentResult:

        logging.info("=" * 70)
        logging.info("SCHEMA ALIGNMENT STARTED")
        logging.info("=" * 70)

        start_time = time.perf_counter()

        try:
            aligned_versions: Dict[str, Dict[str, pd.DataFrame]] = {}
            all_reports: List[TableAlignmentReport] = []

            for version_name, version_data in all_versions.items():
                logging.info("Aligning %s", version_name)

                aligned_version, version_reports = self.align_version(
                    version_name, version_data
                )
                aligned_versions[version_name] = aligned_version
                all_reports.extend(version_reports)

            summary = _build_alignment_summary(
                reports=tuple(all_reports),
                versions_processed=len(aligned_versions),
                execution_time_seconds=time.perf_counter() - start_time,
            )

            logging.info("=" * 70)
            logging.info("Alignment summary: %s", summary.as_log_dict())
            logging.info("SCHEMA ALIGNMENT COMPLETED")
            logging.info("=" * 70)

            return SchemaAlignmentResult(
                aligned_data=aligned_versions,
                reports=tuple(all_reports),
                summary=summary,
            )

        except CustomException:
            raise
        except Exception as e:
            logging.exception("Schema Alignment Failed")
            raise CustomException(e, sys) from e
"""
Master Dataset Builder component (formerly "Dataset Consolidation" --
renamed because "consolidation" undersold what this stage produces: the
single, analysis-ready master dataset that Feature Engineering consumes
directly, not just a generic merge step).

Pipeline position: ... -> Data Cleaning -> **Master Dataset Builder**
                    -> Feature Engineering -> ...

Takes the cleaned, per-version relational tables produced by Data
Cleaning (`Dict[version_name, Dict[table_name, DataFrame]]`) and merges
them into one denormalized, analysis-ready table per version -- one row
per job posting, every foreign key resolved, every bridge table
aggregated -- then concatenates all versions into a single
`master_dataset` for Feature Engineering to consume directly.

This component takes ONLY cleaned data. It never reads a CSV, and it
never sees raw or merely-aligned/validated data -- by the time this stage
runs, every table has already been schema-aligned, validated, and
cleaned, so this stage's only job is joining, not fixing.

--------------------------------------------------------------------------
The one hard invariant: joins must never change the fact table's row count
--------------------------------------------------------------------------
Every join implemented here is a LEFT join from the fact table
(`postings`) outward. A left join only changes row count if the table
being joined in has more than one row per join key -- so every join type
in this module (`DimensionJoinSpec`, `SnapshotJoinSpec`,
`BridgeAggregationSpec`; see dataset_consolidation_config.py) is designed
to guarantee at-most-one-row-per-key *before* the merge happens, whether
via defensive de-duplication, snapshot selection, or aggregation. After
every version is built, `row_count_preserved` is checked and enforced --
if it's ever violated, that's a bug in a join spec, not a data-quality
nuance, so it raises rather than silently producing a row-multiplied
dataset.

--------------------------------------------------------------------------
Why this stage never modifies the tables it was given
--------------------------------------------------------------------------
Consistent with every prior stage: the fact table is copied before any
joins are applied, and no table from `cleaned_data` is ever mutated in
place. The *fact table's own values* are never altered here either --
only new columns are added via left joins. This stage restructures data,
it does not clean it a second time.

--------------------------------------------------------------------------
Join quality reporting: match rates and coverage, not just success/fail
--------------------------------------------------------------------------
A join can "succeed" (the table was present, the merge ran) while still
telling an important story about data quality -- e.g. only 60% of
postings actually got a salary row. Every dimension/snapshot/bridge join
now records a `JoinMatchStats` entry (rows matched vs. unmatched, and a
match rate) in addition to the plain success/failure list, and every
newly created column gets a `column_coverage` entry (% non-null) in the
final report. This is what lets a report reader see "salary coverage:
60%" instead of only "salary column is nullable".

--------------------------------------------------------------------------
Schema fingerprint and dataset_id
--------------------------------------------------------------------------
Each run's master dataset gets a content-derived `schema_hash` (sha256 of
its sorted column-name/dtype pairs) and a human-readable `dataset_id`
(e.g. `ds_20260730_014512`). Feature Engineering can compare its expected
schema hash against the one written here to fail fast on drift, and the
dataset_id gives every run a stable handle for experiment tracking (e.g.
MLflow run tags) that's more legible than a raw timestamp.

--------------------------------------------------------------------------
Why saving to disk is a separate, explicitly-called method
--------------------------------------------------------------------------
`build_master_dataset()` is a pure function of its input: no filesystem
access, fully unit-testable without a temp directory. Writing
parquet/JSON artifacts is a distinct concern with its own failure modes
(disk space, permissions, existing files) and is handled by
`save_result()`, called explicitly by the pipeline entrypoint -- the same
separation this project uses everywhere: components don't have execution
side effects, `src/pipeline.py` does.

`initiate_dataset_consolidation()` is kept as a deprecated alias for
`build_master_dataset()` during the pipeline migration; it emits a
`DeprecationWarning` and should be swapped out at each call site as the
pipeline migrates, then removed.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.configs.data_consolidation_config import (
    BridgeAggregationSpec,
    DimensionJoinSpec,
    MasterDatasetBuilderConfig,
    SnapshotJoinSpec,
)
from src.exception import CustomException
from src.logger import logging


@dataclass(frozen=True)
class JoinMatchStats:
    """
    How well one joined table's key actually matched the fact table --
    distinct from whether the join *ran* (that's tracked separately in
    `tables_joined` / `join_failures`). A join can succeed mechanically
    while still matching only a fraction of rows.

    Attributes:
        table_name: the joined/aggregated table this stat is about.
        join_type: "dimension", "snapshot", or "bridge".
        key_column: the key the match was computed on (left-hand side).
        rows_total: rows in the consolidated frame at merge time.
        rows_matched: rows that found at least one corresponding row in
            the joined table (for bridge joins: rows with >=1 associated
            bridge entry).
    """

    table_name: str
    join_type: str
    key_column: str
    rows_total: int
    rows_matched: int

    @property
    def rows_unmatched(self) -> int:
        return self.rows_total - self.rows_matched

    @property
    def match_rate(self) -> float:
        return round(self.rows_matched / self.rows_total, 4) if self.rows_total else 0.0

    def as_log_dict(self) -> dict:
        log_dict = asdict(self)
        log_dict["rows_unmatched"] = self.rows_unmatched
        log_dict["match_rate"] = self.match_rate
        return log_dict


@dataclass(frozen=True)
class DatasetBuildReport:
    """Structured record of one dataset version's build run."""

    version: str
    rows_before: int
    rows_after: int
    tables_joined: Tuple[str, ...]
    join_failures: Tuple[str, ...]
    skipped_tables: Tuple[str, ...]
    columns_created: Tuple[str, ...]
    columns_dropped_due_to_collision: Tuple[str, ...]
    duplicate_primary_keys: int
    missing_primary_keys: int
    join_match_stats: Tuple[JoinMatchStats, ...]
    column_coverage: Dict[str, float]
    execution_time_seconds: float

    @property
    def row_count_preserved(self) -> bool:
        return self.rows_before == self.rows_after

    @property
    def integrity_passed(self) -> bool:
        return (
            self.row_count_preserved
            and self.duplicate_primary_keys == 0
            and self.missing_primary_keys == 0
        )

    def as_log_dict(self) -> dict:
        log_dict = asdict(self)
        log_dict["row_count_preserved"] = self.row_count_preserved
        log_dict["integrity_passed"] = self.integrity_passed
        log_dict["join_match_stats"] = [s.as_log_dict() for s in self.join_match_stats]
        return log_dict

    def to_json(self) -> str:
        return json.dumps(self.as_log_dict(), indent=2)


@dataclass(frozen=True)
class DatasetBuildSummary:
    """Aggregate, pipeline-run-level view across every version's report."""

    versions_processed: int
    tables_processed: int
    rows_processed: int
    successful_merges: int
    failed_merges: int
    new_columns: int
    memory_usage_mb: float
    execution_time_seconds: float
    dataset_id: str
    schema_hash: str

    @property
    def integrity_passed(self) -> bool:
        return self.failed_merges == 0

    def as_log_dict(self) -> dict:
        log_dict = asdict(self)
        log_dict["integrity_passed"] = self.integrity_passed
        return log_dict

    def to_json(self) -> str:
        return json.dumps(self.as_log_dict(), indent=2)


@dataclass(frozen=True)
class DatasetBuildResult:
    """
    Full output of the Master Dataset Builder stage.

    Attributes:
        consolidated_data: version -> one denormalized DataFrame (one row
            per job posting), for per-version inspection/debugging.
        master_dataset: every version's built frame concatenated into one
            DataFrame, tagged with a `dataset_version` column -- this is
            what Feature Engineering should consume. Job IDs are NOT
            deduplicated across versions (a job_id from Version_9 and one
            from Version_13 aren't assumed to be the same real-world
            posting without further evidence); that decision is left to
            Feature Engineering / a future stage, deliberately.
        reports: per-version DatasetBuildReport.
        summary: run-level DatasetBuildSummary.
        schema_fingerprint: dict with `schema_hash`, `columns`, and
            `generated_at` for `master_dataset` -- Feature Engineering can
            compare against this to catch schema drift before it causes
            a silent, confusing failure downstream.
    """

    consolidated_data: Dict[str, pd.DataFrame]
    master_dataset: pd.DataFrame
    reports: Tuple[DatasetBuildReport, ...]
    summary: DatasetBuildSummary
    schema_fingerprint: dict


class MasterDatasetBuilder:
    """
    Merges cleaned, per-version relational tables into one analytical
    dataset per version, then into a single concatenated master dataset.
    """

    def __init__(self, config: Optional[MasterDatasetBuilderConfig] = None) -> None:
        self.config = config or MasterDatasetBuilderConfig()

    # ------------------------------------------------------------------
    # Column-collision-safe merge helper
    # ------------------------------------------------------------------

    def _safe_left_merge(
        self,
        consolidated: pd.DataFrame,
        right: pd.DataFrame,
        left_key: str,
        right_key: str,
        table_name: str,
        columns_created: List[str],
        columns_dropped: List[str],
        output_columns: Optional[Tuple[str, ...]] = None,
    ) -> Tuple[pd.DataFrame, int, int]:
        """
        Left-merge `right` into `consolidated`, dropping (and reporting)
        any right-hand column that would collide with an existing
        consolidated column, instead of letting pandas silently apply
        `_x`/`_y` suffixes. The join key itself is never considered a
        collision. If `output_columns` is given, only those (already
        renamed) columns from `right` are candidates for merging at all.

        Returns `(merged, rows_matched, rows_total)` so callers can build
        a `JoinMatchStats` entry without a second pass over the data.
        """
        if left_key not in consolidated.columns:
            raise CustomException(
                ValueError(
                    f"Cannot join '{table_name}': left key '{left_key}' not "
                    f"present in the consolidated frame."
                ),
                sys,
            )

        incoming_columns = [c for c in right.columns if c != right_key]
        if output_columns is not None:
            allowed = set(output_columns)
            incoming_columns = [c for c in incoming_columns if c in allowed]

        keep_columns: List[str] = []

        for column in incoming_columns:
            if column in consolidated.columns:
                logging.warning(
                    "Column '%s' from '%s' collides with an existing consolidated "
                    "column; dropping the incoming column rather than overwriting.",
                    column, table_name,
                )
                columns_dropped.append(f"{table_name}.{column}")
                continue
            keep_columns.append(column)
            columns_created.append(column)

        right_to_merge = right[[right_key] + keep_columns]

        merged = consolidated.merge(
            right_to_merge,
            how="left",
            left_on=left_key,
            right_on=right_key,
            indicator="_join_indicator",
        )

        rows_total = len(merged)
        rows_matched = int((merged["_join_indicator"] == "both").sum())
        merged = merged.drop(columns=["_join_indicator"])

        if right_key != left_key and right_key in merged.columns:
            merged = merged.drop(columns=[right_key])

        return merged, rows_matched, rows_total

    # ------------------------------------------------------------------
    # Join type 1 -- dimension lookups
    # ------------------------------------------------------------------

    def _apply_dimension_join(
        self,
        consolidated: pd.DataFrame,
        version_data: Dict[str, pd.DataFrame],
        spec: DimensionJoinSpec,
        columns_created: List[str],
        columns_dropped: List[str],
    ) -> Tuple[pd.DataFrame, bool, Optional[JoinMatchStats]]:
        table = version_data.get(spec.table_name)
        if table is None:
            logging.warning(
                "Dimension table '%s' not present in this version; skipping join.",
                spec.table_name,
            )
            return consolidated, False, None

        if table[spec.right_key].duplicated().any():
            before = len(table)
            table = table.drop_duplicates(subset=[spec.right_key], keep=spec.dedupe_keep)
            logging.warning(
                "Dimension table '%s' had %d duplicate '%s' row(s); kept '%s' per key.",
                spec.table_name, before - len(table), spec.right_key, spec.dedupe_keep,
            )

        renamed = table.rename(columns=spec.rename_map)
        renamed_right_key = spec.rename_map.get(spec.right_key, spec.right_key)

        merged, rows_matched, rows_total = self._safe_left_merge(
            consolidated, renamed, spec.left_key, renamed_right_key,
            spec.table_name, columns_created, columns_dropped,
            output_columns=spec.output_columns,
        )
        stats = JoinMatchStats(
            table_name=spec.table_name,
            join_type="dimension",
            key_column=spec.left_key,
            rows_total=rows_total,
            rows_matched=rows_matched,
        )
        return merged, True, stats

    # ------------------------------------------------------------------
    # Join type 2 -- snapshot tables (pick one row per key)
    # ------------------------------------------------------------------

    def _apply_snapshot_join(
        self,
        consolidated: pd.DataFrame,
        version_data: Dict[str, pd.DataFrame],
        spec: SnapshotJoinSpec,
        columns_created: List[str],
        columns_dropped: List[str],
    ) -> Tuple[pd.DataFrame, bool, Optional[JoinMatchStats]]:
        table = version_data.get(spec.table_name)
        if table is None:
            logging.warning(
                "Snapshot table '%s' not present in this version; skipping join.",
                spec.table_name,
            )
            return consolidated, False, None

        if spec.order_by_column and spec.order_by_column in table.columns:
            table = table.sort_values(spec.order_by_column, ascending=False)

        before = len(table)
        table = table.drop_duplicates(subset=[spec.right_key], keep="first")
        if before != len(table):
            logging.info(
                "Snapshot table '%s': reduced %d row(s) to %d (one per '%s').",
                spec.table_name, before, len(table), spec.right_key,
            )

        renamed = table.rename(columns=spec.rename_map)
        renamed_right_key = spec.rename_map.get(spec.right_key, spec.right_key)

        merged, rows_matched, rows_total = self._safe_left_merge(
            consolidated, renamed, spec.left_key, renamed_right_key,
            spec.table_name, columns_created, columns_dropped,
            output_columns=spec.output_columns,
        )
        stats = JoinMatchStats(
            table_name=spec.table_name,
            join_type="snapshot",
            key_column=spec.left_key,
            rows_total=rows_total,
            rows_matched=rows_matched,
        )
        return merged, True, stats

    # ------------------------------------------------------------------
    # Join type 3 -- bridge tables (must be aggregated before joining)
    # ------------------------------------------------------------------

    def _aggregate_bridge_table(
        self,
        version_data: Dict[str, pd.DataFrame],
        spec: BridgeAggregationSpec,
    ) -> Optional[pd.DataFrame]:
        """
        Aggregate a bridge table per `spec.group_by_key` into count/list/top
        columns, entirely with vectorized pandas operations -- no Python
        callable is invoked once per group. That distinction matters at
        scale: a naive `.agg(lambda s: ...)` calls Python once per group,
        which for a dataset with tens of thousands of distinct keys
        dominates runtime regardless of how little work each call does.
        Here, `nunique`/`size`/`sort_values`/`drop_duplicates` are all
        vectorized; the only per-group Python step (building the capped
        list) runs over a pre-capped, already-deduplicated frame that's
        far smaller than the raw bridge table.
        """
        bridge = version_data.get(spec.table_name)
        if bridge is None:
            logging.warning(
                "Bridge table '%s' not present in this version; skipping aggregation.",
                spec.table_name,
            )
            return None

        value_column = spec.value_column

        if spec.dimension_table is not None:
            dimension = version_data.get(spec.dimension_table)
            if dimension is None:
                logging.warning(
                    "Dimension table '%s' (for resolving '%s') not present; "
                    "aggregating '%s' using raw '%s' values instead.",
                    spec.dimension_table, spec.value_column,
                    spec.table_name, spec.value_column,
                )
            else:
                bridge = bridge.merge(
                    dimension[[spec.dimension_key_column, spec.dimension_name_column]],
                    how="left",
                    left_on=spec.dimension_join_key,
                    right_on=spec.dimension_key_column,
                )
                value_column = spec.dimension_name_column

        if spec.group_by_key not in bridge.columns or value_column not in bridge.columns:
            logging.warning(
                "Bridge table '%s' is missing '%s' or '%s'; skipping aggregation.",
                spec.table_name, spec.group_by_key, value_column,
            )
            return None

        key = spec.group_by_key
        prefix = spec.output_prefix

        working = bridge[[key, value_column]].dropna(subset=[value_column])
        if working.empty:
            return pd.DataFrame(columns=[key])

        # One row per (key, distinct value), with how many times it occurred.
        pair_counts = (
            working.groupby([key, value_column]).size().reset_index(name="_occurrences")
        )

        agg_df = pair_counts[[key]].drop_duplicates(subset=[key])

        # Count -- vectorized: number of distinct values per key.
        if spec.include_count:
            count_df = (
                pair_counts.groupby(key).size().reset_index(name=f"{prefix}_count")
            )
            agg_df = agg_df.merge(count_df, on=key, how="left")

        # Top value -- vectorized: stable-sort by occurrence count
        # descending, then keep the first row per key. Ties keep
        # whichever value appeared first in the source data.
        if spec.include_top:
            top_df = (
                pair_counts.sort_values("_occurrences", ascending=False, kind="mergesort")
                .drop_duplicates(subset=[key], keep="first")[[key, value_column]]
                .rename(columns={value_column: f"top_{prefix}"})
            )
            agg_df = agg_df.merge(top_df, on=key, how="left")

        # Capped list -- the only per-group Python step, but it runs over
        # at most `top_n_list` rows per key (via the vectorized
        # sort + groupby.head), not the full bridge table.
        if spec.include_list:
            capped = (
                pair_counts.sort_values([key, value_column])
                .groupby(key)
                .head(spec.top_n_list)
            )
            list_df = (
                capped.groupby(key)[value_column].agg(list).reset_index(name=f"{prefix}_list")
            )
            agg_df = agg_df.merge(list_df, on=key, how="left")

        return agg_df

    def _apply_bridge_aggregation(
        self,
        consolidated: pd.DataFrame,
        version_data: Dict[str, pd.DataFrame],
        spec: BridgeAggregationSpec,
        columns_created: List[str],
        columns_dropped: List[str],
    ) -> Tuple[pd.DataFrame, bool, Optional[JoinMatchStats]]:
        agg_df = self._aggregate_bridge_table(version_data, spec)
        if agg_df is None:
            return consolidated, False, None

        merged, rows_matched, rows_total = self._safe_left_merge(
            consolidated, agg_df, spec.group_by_key, spec.group_by_key,
            spec.table_name, columns_created, columns_dropped,
        )

        count_col = f"{spec.output_prefix}_count"
        if count_col in merged.columns:
            merged[count_col] = merged[count_col].fillna(0).astype("Int64")

        stats = JoinMatchStats(
            table_name=spec.table_name,
            join_type="bridge",
            key_column=spec.group_by_key,
            rows_total=rows_total,
            rows_matched=rows_matched,
        )
        return merged, True, stats

    # ------------------------------------------------------------------
    # Version-level build
    # ------------------------------------------------------------------

    def consolidate_version(
        self,
        version_name: str,
        version_data: Dict[str, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, DatasetBuildReport]:
        """Build the master table for one dataset version."""

        start_time = time.perf_counter()

        fact_table = version_data.get(self.config.fact_table)
        if fact_table is None:
            raise CustomException(
                ValueError(
                    f"[{version_name}] Fact table '{self.config.fact_table}' not "
                    f"present -- cannot build the master dataset without it."
                ),
                sys,
            )

        try:
            consolidated = fact_table.copy(deep=True)
            rows_before = len(consolidated)

            tables_joined: List[str] = []
            join_failures: List[str] = []
            skipped_tables: List[str] = []
            columns_created: List[str] = []
            columns_dropped: List[str] = []
            join_match_stats: List[JoinMatchStats] = []

            all_specs: List[Tuple[str, object]] = (
                [("dimension", s) for s in self.config.dimension_joins]
                + [("snapshot", s) for s in self.config.snapshot_joins]
                + [("bridge", s) for s in self.config.bridge_aggregations]
            )

            for kind, spec in all_specs:
                if not spec.enabled:
                    logging.info(
                        "[%s] Table '%s' is disabled via config; skipping.",
                        version_name, spec.table_name,
                    )
                    skipped_tables.append(spec.table_name)
                    continue

                if kind == "dimension":
                    consolidated, succeeded, stats = self._apply_dimension_join(
                        consolidated, version_data, spec, columns_created, columns_dropped
                    )
                elif kind == "snapshot":
                    consolidated, succeeded, stats = self._apply_snapshot_join(
                        consolidated, version_data, spec, columns_created, columns_dropped
                    )
                else:
                    consolidated, succeeded, stats = self._apply_bridge_aggregation(
                        consolidated, version_data, spec, columns_created, columns_dropped
                    )

                (tables_joined if succeeded else join_failures).append(spec.table_name)
                if stats is not None:
                    join_match_stats.append(stats)

            rows_after = len(consolidated)

            primary_key = self.config.fact_primary_key
            duplicate_primary_keys = int(consolidated[primary_key].duplicated().sum())
            missing_primary_keys = int(consolidated[primary_key].isna().sum())

            column_coverage = {
                column: round(float(consolidated[column].notna().mean()), 4)
                for column in columns_created
                if column in consolidated.columns
            }

            report = DatasetBuildReport(
                version=version_name,
                rows_before=rows_before,
                rows_after=rows_after,
                tables_joined=tuple(tables_joined),
                join_failures=tuple(join_failures),
                skipped_tables=tuple(skipped_tables),
                columns_created=tuple(columns_created),
                columns_dropped_due_to_collision=tuple(columns_dropped),
                duplicate_primary_keys=duplicate_primary_keys,
                missing_primary_keys=missing_primary_keys,
                join_match_stats=tuple(join_match_stats),
                column_coverage=column_coverage,
                execution_time_seconds=round(time.perf_counter() - start_time, 4),
            )

            self._enforce_integrity(version_name, report)
            self._log_report(report)

            return consolidated, report

        except CustomException:
            raise
        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def _enforce_integrity(version_name: str, report: DatasetBuildReport) -> None:
        """
        Row-count preservation and primary-key integrity are not
        data-quality nuances to report and move on from -- they are
        correctness invariants of the merge engine itself. A violation
        means a join spec is wrong (e.g. a "dimension" table actually has
        duplicate keys that weren't caught), not that the input data is
        merely imperfect, so this raises rather than just logging.
        """
        problems: List[str] = []

        if not report.row_count_preserved:
            problems.append(
                f"row count changed from {report.rows_before} to {report.rows_after}"
            )
        if report.duplicate_primary_keys:
            problems.append(f"{report.duplicate_primary_keys} duplicate primary key(s)")
        if report.missing_primary_keys:
            problems.append(f"{report.missing_primary_keys} missing primary key(s)")

        if problems:
            raise CustomException(
                ValueError(
                    f"[{version_name}] Build integrity check failed: "
                    + "; ".join(problems)
                ),
                sys,
            )

    @staticmethod
    def _log_report(report: DatasetBuildReport) -> None:
        if report.join_failures:
            logging.warning(
                "[%s] Tables not joined (missing from this version): %s",
                report.version, report.join_failures,
            )
        if report.skipped_tables:
            logging.info(
                "[%s] Tables skipped (disabled via config): %s",
                report.version, report.skipped_tables,
            )
        if report.columns_dropped_due_to_collision:
            logging.warning(
                "[%s] Columns dropped due to name collision: %s",
                report.version, report.columns_dropped_due_to_collision,
            )
        for stats in report.join_match_stats:
            logging.info(
                "[%s] %s join '%s' matched %d/%d rows (%.1f%%)",
                report.version, stats.join_type, stats.table_name,
                stats.rows_matched, stats.rows_total, stats.match_rate * 100,
            )
        logging.info(
            "[%s] Build complete -> %d row(s), %d table(s) joined, "
            "%d new column(s), integrity_passed: %s",
            report.version, report.rows_after, len(report.tables_joined),
            len(report.columns_created), report.integrity_passed,
        )

    # ------------------------------------------------------------------
    # Schema fingerprint / dataset id helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_schema_fingerprint(master_dataset: pd.DataFrame) -> dict:
        """
        A content-derived fingerprint of the master dataset's schema
        (column name + dtype pairs, sorted for stability), so Feature
        Engineering -- or anything downstream -- can check
        `schema_hash` against what it expects and fail fast on drift
        instead of hitting a confusing KeyError three steps later.
        """
        columns_with_types = sorted(
            f"{column}:{str(master_dataset[column].dtype)}"
            for column in master_dataset.columns
        )
        fingerprint_source = "|".join(columns_with_types)
        schema_hash = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()
        return {
            "schema_hash": schema_hash,
            "columns": list(master_dataset.columns),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _generate_dataset_id() -> str:
        """
        A human-legible, sortable run identifier (e.g.
        `ds_20260730_014512`) -- more useful as an MLflow tag or archive
        directory name than a raw ISO timestamp, while still being
        trivially derivable from one.
        """
        return f"ds_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    # ------------------------------------------------------------------
    # Pipeline-level orchestration
    # ------------------------------------------------------------------

    def build_master_dataset(
        self,
        cleaned_data: Dict[str, Dict[str, pd.DataFrame]],
    ) -> DatasetBuildResult:
        """
        Build every dataset version discovered in `cleaned_data` (no
        version names are ever hard-coded) and concatenate the results
        into one `master_dataset` for Feature Engineering.

        This is the primary entrypoint for this stage. `initiate_dataset_
        consolidation()` is kept as a deprecated alias during the
        pipeline migration -- prefer this method in new and updated call
        sites.
        """

        logging.info("=" * 70)
        logging.info("MASTER DATASET BUILD STARTED")
        logging.info("=" * 70)

        start_time = time.perf_counter()

        try:
            consolidated_data: Dict[str, pd.DataFrame] = {}
            reports: List[DatasetBuildReport] = []

            for version_name, version_data in cleaned_data.items():
                logging.info("Building %s", version_name)
                consolidated_df, report = self.consolidate_version(version_name, version_data)
                consolidated_df = consolidated_df.copy()
                consolidated_df["dataset_version"] = version_name
                consolidated_data[version_name] = consolidated_df
                reports.append(report)

            master_dataset = (
                pd.concat(consolidated_data.values(), ignore_index=True)
                if consolidated_data
                else pd.DataFrame()
            )

            schema_fingerprint = self._compute_schema_fingerprint(master_dataset)
            dataset_id = self._generate_dataset_id()

            summary = self._build_summary(
                reports=tuple(reports),
                master_dataset=master_dataset,
                execution_time_seconds=time.perf_counter() - start_time,
                dataset_id=dataset_id,
                schema_hash=schema_fingerprint["schema_hash"],
            )

            logging.info("=" * 70)
            logging.info("Build summary: %s", summary.as_log_dict())
            logging.info("MASTER DATASET BUILD COMPLETED")
            logging.info("=" * 70)

            return DatasetBuildResult(
                consolidated_data=consolidated_data,
                master_dataset=master_dataset,
                reports=tuple(reports),
                summary=summary,
                schema_fingerprint=schema_fingerprint,
            )

        except CustomException:
            raise
        except Exception as e:
            logging.exception("Master Dataset Build Failed")
            raise CustomException(e, sys) from e

    def initiate_dataset_consolidation(
        self,
        cleaned_data: Dict[str, Dict[str, pd.DataFrame]],
    ) -> DatasetBuildResult:
        """
        Deprecated alias for `build_master_dataset()`, kept only so
        existing pipeline call sites (e.g. `src/pipeline.py`) keep
        working during the migration. Emits a `DeprecationWarning`.
        Update call sites to `build_master_dataset()` and remove this
        method once nothing references it.
        """
        warnings.warn(
            "initiate_dataset_consolidation() is deprecated and will be "
            "removed; use build_master_dataset() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.build_master_dataset(cleaned_data)

    @staticmethod
    def _build_summary(
        reports: Tuple[DatasetBuildReport, ...],
        master_dataset: pd.DataFrame,
        execution_time_seconds: float,
        dataset_id: str,
        schema_hash: str,
    ) -> DatasetBuildSummary:
        total_tables_attempted = sum(
            len(r.tables_joined) + len(r.join_failures) for r in reports
        )
        all_new_columns = {col for r in reports for col in r.columns_created}

        return DatasetBuildSummary(
            versions_processed=len(reports),
            tables_processed=total_tables_attempted,
            rows_processed=sum(r.rows_before for r in reports),
            successful_merges=sum(len(r.tables_joined) for r in reports),
            failed_merges=sum(1 for r in reports if not r.integrity_passed),
            new_columns=len(all_new_columns),
            memory_usage_mb=round(
                float(master_dataset.memory_usage(deep=True).sum()) / (1024 ** 2), 4
            ),
            execution_time_seconds=round(execution_time_seconds, 4),
            dataset_id=dataset_id,
            schema_hash=schema_hash,
        )

    # ------------------------------------------------------------------
    # Persistence (explicitly invoked -- see module docstring)
    # ------------------------------------------------------------------

    def save_result(
        self,
        result: DatasetBuildResult,
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Persist `result.master_dataset` plus metadata/report/schema JSON
        under `output_dir/latest/` (always overwritten) and, if
        `config.keep_archive_snapshots` is True, an additional timestamped
        copy under `output_dir/archive/<dataset_id>/` for reproducibility.

        Filenames written under `latest/` (and mirrored under
        `archive/<dataset_id>/`):
            - master_dataset.parquet
            - master_dataset_metadata.json
            - master_dataset_report.json
            - schema_fingerprint.json

        Returns a dict of the file paths written.
        """
        base_dir = Path(output_dir or self.config.output_dir)
        latest_dir = base_dir / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)

        written: Dict[str, str] = {}

        dataset_path = latest_dir / "master_dataset.parquet"
        result.master_dataset.to_parquet(dataset_path, index=False)
        written["dataset"] = str(dataset_path)

        metadata = {
            "dataset_id": result.summary.dataset_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "rows": len(result.master_dataset),
            "columns": list(result.master_dataset.columns),
            "versions": list(result.consolidated_data.keys()),
            "schema_hash": result.summary.schema_hash,
        }
        metadata_path = latest_dir / "master_dataset_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        written["metadata"] = str(metadata_path)

        schema_path = latest_dir / "schema_fingerprint.json"
        schema_path.write_text(json.dumps(result.schema_fingerprint, indent=2))
        written["schema_fingerprint"] = str(schema_path)

        report_payload = {
            "dataset_id": result.summary.dataset_id,
            "summary": result.summary.as_log_dict(),
            "reports": [r.as_log_dict() for r in result.reports],
        }
        report_path = latest_dir / "master_dataset_report.json"
        report_path.write_text(json.dumps(report_payload, indent=2, default=str))
        written["report"] = str(report_path)

        if self.config.keep_archive_snapshots:
            archive_dir = base_dir / "archive" / result.summary.dataset_id
            archive_dir.mkdir(parents=True, exist_ok=True)

            for name, path_str in list(written.items()):
                src_path = Path(path_str)
                dst_path = archive_dir / src_path.name
                dst_path.write_bytes(src_path.read_bytes())
                written[f"archive_{name}"] = str(dst_path)

        logging.info("Saved master dataset build output: %s", written)
        return written


# ----------------------------------------------------------------------
# Backward-compatible alias -- DEPRECATED, migrate to
# `MasterDatasetBuilder` and remove this once the pipeline no longer
# references the old name.
# ----------------------------------------------------------------------
class DatasetConsolidation(MasterDatasetBuilder):
    """
    Deprecated alias for `MasterDatasetBuilder`. Kept only so existing
    code (e.g. `src/pipeline.py`) keeps working during the migration;
    instantiating this emits a `DeprecationWarning`. Update call sites to
    `MasterDatasetBuilder` and remove this class once nothing references
    it.
    """

    def __init__(self, config: Optional[MasterDatasetBuilderConfig] = None) -> None:
        warnings.warn(
            "DatasetConsolidation is deprecated and will be removed; "
            "use MasterDatasetBuilder instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(config)


# Backward-compatible aliases for the pre-rename report/summary/result
# dataclass names, in case other modules reference them directly (e.g.
# for type hints or isinstance checks). These are plain aliases (not
# subclasses) since they're typically referenced, not instantiated
# directly -- remove once nothing references them.
ConsolidationReport = DatasetBuildReport
ConsolidationSummary = DatasetBuildSummary
ConsolidationResult = DatasetBuildResult
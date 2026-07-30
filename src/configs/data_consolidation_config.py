"""
Configuration for the Master Dataset Builder component
(formerly "Dataset Consolidation" -- renamed to better reflect what this
stage actually produces: the single analysis-ready dataset that Feature
Engineering consumes, not just a generic "consolidation" step).

This module contains ONLY configuration (data), never processing logic --
mirroring the split used for every prior component. `MasterDatasetBuilder`
(in src/components/dataset_consolidation.py) is the sole consumer of this
config: adding a new table to the master dataset means adding one spec
here, not touching the merge engine's code.

--------------------------------------------------------------------------
Why three different join types, instead of one generic "join"
--------------------------------------------------------------------------
The one hard invariant is: joining a table into the fact table must never
change the fact table's row count. Three genuinely different situations
satisfy that invariant in three different ways, so the config models them
as three distinct spec types rather than one "join" concept with optional
flags bolted on:

    1. DimensionJoinSpec  -- the joined table is *expected* to already
       have at most one row per key (e.g. one row per company_id in
       `companies`). We still defensively de-duplicate before joining
       (data can surprise you), but conceptually this is a simple lookup.

    2. SnapshotJoinSpec   -- the joined table legitimately has multiple
       rows per key by design (e.g. `employee_counts` stores one row per
       company per time it was recorded). Before joining, we must reduce
       it to exactly one row per key by picking the most recent snapshot
       (or, if there's no ordering column, the first row).

    3. BridgeAggregationSpec -- the joined table is a many-to-many bridge
       (e.g. `job_industries` links one job to many industries). It can
       never be joined directly without multiplying fact-table rows; it
       must first be aggregated per key into summary columns (a count, a
       capped list, and the single most common value).

Getting this distinction wrong (e.g. treating a bridge table as a
dimension table) is exactly the mistake that silently multiplies rows in
naive consolidation code -- modelling it explicitly in the config makes
that mistake structurally harder to make.

--------------------------------------------------------------------------
`enabled` and column-trimming knobs (added for Feature Store readiness)
--------------------------------------------------------------------------
Every spec type now carries an `enabled` flag so an individual join can be
switched off from configuration alone -- useful when a downstream Feature
Store already provides a table's signal and re-joining it here would just
be dead weight. Dimension/snapshot joins also accept `output_columns` to
trim which of the joined table's columns actually make it into the master
dataset; bridge aggregations expose `include_count` / `include_list` /
`include_top` for the same purpose, since each of those three generated
columns has a different cost (the list column in particular can be large).
None of this changes today's default behavior -- every default is the
same "keep everything, always run" behavior as before.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class DimensionJoinSpec:
    """
    A simple lookup join: at most one row per `right_key` in `table_name`
    is expected. Left-joined into the fact table on
    `left_key == right_key`.

    Attributes:
        table_name: canonical table to join in (e.g. "companies").
        left_key: column in the accumulating consolidated frame to join on.
        right_key: column in `table_name` to join on.
        rename_map: original_column -> renamed_column, applied to
            `table_name` before merging, so generic names like "name" or
            "description" don't silently collide with the fact table's
            own columns (e.g. "name" -> "company_name").
        dedupe_keep: if `right_key` turns out not to be unique in
            `table_name`, which duplicate to keep ("first" or "last")
            before joining -- a defensive fallback, not the expected path.
        enabled: if False, this join is skipped entirely (recorded in the
            report's `skipped_tables`, not counted as a failure).
        output_columns: if set, only these (post-rename) columns from
            `table_name` are kept; all others are dropped before the
            merge. `None` keeps every non-key column (today's default).
    """

    table_name: str
    left_key: str
    right_key: str
    rename_map: Dict[str, str] = field(default_factory=dict)
    dedupe_keep: str = "first"
    enabled: bool = True
    output_columns: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class SnapshotJoinSpec:
    """
    A join against a table with legitimately multiple rows per key, where
    exactly one "current" row per key must be selected before joining.

    Attributes:
        table_name: e.g. "employee_counts", "salaries".
        left_key / right_key: join columns.
        order_by_column: column used to pick the most recent snapshot per
            key (highest value kept, e.g. "time_recorded"). If None,
            simply keeps the first row encountered per key.
        rename_map: same purpose as DimensionJoinSpec's.
        enabled: if False, this join is skipped entirely (recorded in the
            report's `skipped_tables`, not counted as a failure).
        output_columns: if set, only these (post-rename) columns from
            `table_name` are kept. `None` keeps every non-key column.
    """

    table_name: str
    left_key: str
    right_key: str
    order_by_column: Optional[str] = None
    rename_map: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    output_columns: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class BridgeAggregationSpec:
    """
    A many-to-many bridge table, aggregated per key into up to three
    summary columns before joining: `{output_prefix}_count`,
    `{output_prefix}_list` (capped), and `top_{output_prefix}` (the most
    frequent value).

    Attributes:
        table_name: e.g. "job_industries", "job_skills", "benefits".
        group_by_key: column to aggregate by -- must match a column that
            will exist in the consolidated frame (typically "job_id" or
            "company_id").
        value_column: column in `table_name` holding the raw value being
            aggregated (e.g. "industry_id", "type").
        output_prefix: prefix for the generated columns.
        dimension_table / dimension_join_key / dimension_key_column /
            dimension_name_column: if the raw value is an id that needs
            resolving to a human-readable name first (e.g. industry_id ->
            industry_name via the `industries` table), specify all four;
            otherwise leave as None and `value_column`'s raw values are
            used directly (e.g. `benefits.type` is already a name).
        top_n_list: maximum number of distinct values kept in the
            generated list column, to keep cell sizes bounded.
        enabled: if False, this aggregation is skipped entirely (recorded
            in the report's `skipped_tables`, not counted as a failure).
        include_count / include_list / include_top: toggle which of the
            three generated columns are actually produced and joined.
            All default to True (today's behavior).
    """

    table_name: str
    group_by_key: str
    value_column: str
    output_prefix: str
    dimension_table: Optional[str] = None
    dimension_join_key: Optional[str] = None
    dimension_key_column: Optional[str] = None
    dimension_name_column: Optional[str] = None
    top_n_list: int = 10
    enabled: bool = True
    include_count: bool = True
    include_list: bool = True
    include_top: bool = True


def _default_dimension_joins() -> Tuple[DimensionJoinSpec, ...]:
    return (
        DimensionJoinSpec(
            table_name="companies",
            left_key="company_id",
            right_key="company_id",
            rename_map={
                "name": "company_name",
                "description": "company_description",
                "company_size": "company_size",
                "state": "company_state",
                "country": "company_country",
                "city": "company_city",
                "zip_code": "company_zip_code",
                "address": "company_address",
                "url": "company_url",
            },
        ),
    )


def _default_snapshot_joins() -> Tuple[SnapshotJoinSpec, ...]:
    return (
        SnapshotJoinSpec(
            table_name="employee_counts",
            left_key="company_id",
            right_key="company_id",
            order_by_column="time_recorded",
            rename_map={
                "employee_count": "company_employee_count",
                "follower_count": "company_follower_count",
                "time_recorded": "company_employee_count_recorded_at",
            },
        ),
        SnapshotJoinSpec(
            table_name="salaries",
            left_key="job_id",
            right_key="job_id",
            order_by_column=None,
            rename_map={
                "min_salary": "listed_min_salary",
                "max_salary": "listed_max_salary",
                "med_salary": "listed_med_salary",
                "pay_period": "listed_pay_period",
                "currency": "listed_currency",
                "compensation_type": "listed_compensation_type",
            },
        ),
    )


def _default_bridge_aggregations() -> Tuple[BridgeAggregationSpec, ...]:
    return (
        BridgeAggregationSpec(
            table_name="job_industries",
            group_by_key="job_id",
            value_column="industry_id",
            output_prefix="industry",
            dimension_table="industries",
            dimension_join_key="industry_id",
            dimension_key_column="industry_id",
            dimension_name_column="industry_name",
        ),
        BridgeAggregationSpec(
            table_name="job_skills",
            group_by_key="job_id",
            value_column="skill_abr",
            output_prefix="skill",
            dimension_table="skills",
            dimension_join_key="skill_abr",
            dimension_key_column="skill_abr",
            dimension_name_column="skill_name",
        ),
        BridgeAggregationSpec(
            table_name="benefits",
            group_by_key="job_id",
            value_column="type",
            output_prefix="benefit",
        ),
        BridgeAggregationSpec(
            table_name="company_specialities",
            group_by_key="company_id",
            value_column="speciality",
            output_prefix="speciality",
        ),
    )


@dataclass(frozen=True)
class MasterDatasetBuilderConfig:
    """
    Top-level configuration for the Master Dataset Builder component.

    Attributes:
        fact_table: the canonical table every other table joins into
            (one row per real-world entity of interest -- "postings").
        fact_primary_key: the fact table's unique identifier ("job_id").
            After the build, this column must have zero duplicates and
            zero nulls, or the run is considered broken (see
            `_enforce_integrity` in dataset_consolidation.py).
        dimension_joins / snapshot_joins / bridge_aggregations: see the
            respective spec dataclasses above. Applied in this order.
        output_dir: base directory for persisted build outputs (see
            `MasterDatasetBuilder.save_result`). Not used by the core
            merge logic itself, which is a pure function of its inputs
            and has no filesystem side effects.
        keep_archive_snapshots: if True, `save_result` writes a
            timestamped copy under `output_dir/archive/<timestamp>/` in
            addition to overwriting `output_dir/latest/` -- so you can
            always answer "what data trained model version X" without
            losing the convenience of a stable "latest" path.
    """

    fact_table: str = "postings"
    fact_primary_key: str = "job_id"

    dimension_joins: Tuple[DimensionJoinSpec, ...] = field(
        default_factory=_default_dimension_joins
    )
    snapshot_joins: Tuple[SnapshotJoinSpec, ...] = field(
        default_factory=_default_snapshot_joins
    )
    bridge_aggregations: Tuple[BridgeAggregationSpec, ...] = field(
        default_factory=_default_bridge_aggregations
    )

    output_dir: str = "artifacts/master_dataset"
    keep_archive_snapshots: bool = True


# ----------------------------------------------------------------------
# Backward-compatible alias -- DEPRECATED, migrate to
# `MasterDatasetBuilderConfig` and remove this once the pipeline no
# longer references the old name.
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetConsolidationConfig(MasterDatasetBuilderConfig):
    """
    Deprecated alias for `MasterDatasetBuilderConfig`. Kept only so
    existing imports (e.g. `src/pipeline.py`) keep working during the
    migration; instantiating this emits a `DeprecationWarning`. Update
    call sites to `MasterDatasetBuilderConfig` and remove this class once
    nothing references it.
    """

    def __post_init__(self) -> None:
        warnings.warn(
            "DatasetConsolidationConfig is deprecated and will be removed; "
            "use MasterDatasetBuilderConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )
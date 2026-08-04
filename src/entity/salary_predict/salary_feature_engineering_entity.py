"""
src/entity/salary_feature_engineering_entity.py

Entity dataclasses for the Salary Feature Engineering stage.

These are pure data containers — no logic lives here. They exist so that
`salary_feature_engineering.py` has a typed, structured object to return,
and so the report/summary written to disk has a single source of truth
for its schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class SalaryFeatureEngineeringSummary:
    """
    Numeric/statistical summary of a single run of the salary feature
    engineering stage. This is the payload that gets logged and written
    to `salary_feature_report.json`.
    """
    dataset_id: str
    input_row_count: int
    salary_candidate_count: int
    valid_usd_salary_count: int

    rows_removed_missing_salary: int
    rows_removed_currency: int
    rows_removed_unsupported_pay_period: int
    rows_removed_out_of_bounds: int

    final_row_count: int
    target_coverage_pct: float

    target_source_counts: Dict[str, int]

    annual_salary_stats: Dict[str, float]   # min/max/mean/median/std
    log_salary_stats: Dict[str, float]      # min/max/mean/median/std

    feature_columns: List[str]
    metadata_columns: List[str]
    leakage_columns_removed: List[str]

    schema_hash: str
    execution_time_seconds: float
    integrity_passed: bool

    def as_log_dict(self) -> dict:
        """Flat dict suitable for structured logging (no nested truncation)."""
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


@dataclass
class SalaryFeatureEngineeringReport:
    """
    Wraps the summary with run-level metadata (paths, timestamp, config
    snapshot). This is a slightly higher-level record than the summary —
    the summary is "what happened to the data", the report is "what
    happened to the data, in the context of this specific run".
    """

    dataset_id: str
    created_at: str
    input_feature_store_path: str
    output_dataset_path: str
    summary: SalaryFeatureEngineeringSummary

    def as_log_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.as_log_dict(), indent=2, default=str)


@dataclass
class SalaryFeatureEngineeringResult:
    """
    Return value of `SalaryFeatureEngineering.initiate_salary_feature_engineering()`.
    Exposes every artifact path produced, plus the summary, so that
    `pipeline.py` and downstream stages (dataset splitter) have everything
    they need without re-deriving paths themselves.
    """

    salary_modeling_dataset_path: Path
    metadata_path: Path
    report_path: Path
    schema_fingerprint_path: Path

    summary: SalaryFeatureEngineeringSummary

    archive_dataset_path: Optional[Path] = None
    archive_metadata_path: Optional[Path] = None
    archive_report_path: Optional[Path] = None
    archive_schema_fingerprint_path: Optional[Path] = None

    dataframe: Optional[pd.DataFrame] = field(default=None,repr=False,)
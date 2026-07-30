"""
Entity definitions for the Common Feature Engineering stage.
Includes custom exception handling and centralized logging.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd

from src.logger import logging
from src.exception import CustomException


@dataclass
class TransformationStepSummary:
    step_name: str
    columns_affected: List[str] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        try:
            return asdict(self)
        except Exception as e:
            logging.error(f"Error converting TransformationStepSummary to dict: {e}")
            raise CustomException(e, sys)


@dataclass
class FeatureEngineeringSummary:
    dataset_id: str
    rows_before: int = 0
    rows_after: int = 0
    rows_removed: int = 0
    columns_before: int = 0
    columns_after: int = 0
    identifiers_dropped: List[str] = field(default_factory=list)
    columns_removed: List[str] = field(default_factory=list)
    columns_created: List[str] = field(default_factory=list)
    dtype_conversions: Dict[str, str] = field(default_factory=dict)
    null_values_before: int = 0
    null_values_after: int = 0
    missing_values_filled: int = 0
    duplicates_removed: int = 0
    schema_hash: str = ""
    execution_time_seconds: float = 0.0
    integrity_passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        try:
            return asdict(self)
        except Exception as e:
            logging.error(f"Error converting FeatureEngineeringSummary to dict: {e}")
            raise CustomException(e, sys)


@dataclass
class FeatureEngineeringReport:
    summary: FeatureEngineeringSummary
    steps: List[TransformationStepSummary] = field(default_factory=list)
    final_schema: Dict[str, str] = field(default_factory=dict)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        try:
            return {
                "summary": self.summary.to_dict(),
                "steps": [s.to_dict() for s in self.steps],
                "final_schema": self.final_schema,
                "generated_at": self.generated_at,
            }
        except Exception as e:
            logging.error(f"Error converting FeatureEngineeringReport to dict: {e}")
            raise CustomException(e, sys)


@dataclass
class FeatureEngineeringResult:
    feature_store_path: str
    metadata_path: str | None = None
    report_path: str | None = None
    schema_fingerprint_path: str | None = None

    feature_store: pd.DataFrame | None = None
    archive_feature_store_path: str | None = None
    archive_metadata_path: str | None = None
    archive_report_path: str | None = None
    archive_schema_fingerprint_path: str | None = None
    summary: FeatureEngineeringSummary | None = None

    def to_dict(self) -> Dict[str, Any]:
        try:
            return {
                "feature_store_path": self.feature_store_path,
                "metadata_path": self.metadata_path,
                "report_path": self.report_path,
                "schema_fingerprint_path": self.schema_fingerprint_path,
                "archive_feature_store_path": self.archive_feature_store_path,
                "archive_metadata_path": self.archive_metadata_path,
                "archive_report_path": self.archive_report_path,
                "archive_schema_fingerprint_path": self.archive_schema_fingerprint_path,
                "summary": self.summary.to_dict() if self.summary is not None else None,
            }
        except Exception as e:
            logging.error(f"Error converting FeatureEngineeringResult to dict: {e}")
            raise CustomException(e, sys)
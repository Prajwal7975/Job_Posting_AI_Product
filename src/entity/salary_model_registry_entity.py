from __future__ import annotations

import json

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class SalaryModelRegistryResult:
    
    success: bool

    registered_model_name: str

    model_version: Optional[str] = None

    model_uri: Optional[str] = None

    source_run_id: Optional[str] = None

    production_alias: Optional[str] = None

    alias_updated: bool = False

    model_artifact_path: Optional[str] = None

    validation_passed: bool = False

    test_metrics: Dict[str, float] = field(default_factory=dict)

    validation_metrics: Dict[str, float] = field(default_factory=dict)

    error: Optional[str] = None

    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "success": self.success,
            "registered_model_name": (self.registered_model_name),
            "model_version": self.model_version,
            "model_uri": self.model_uri,
            "source_run_id": self.source_run_id,
            "production_alias": self.production_alias,
            "alias_updated": self.alias_updated,
            "model_artifact_path": (self.model_artifact_path),
            "validation_passed": (self.validation_passed),
            "test_metrics": dict(self.test_metrics),
            "validation_metrics": dict(self.validation_metrics),
            "error": self.error,
            "generated_at": self.generated_at,
        }

    def to_json(self, indent: int = 2) -> str:

        return json.dumps(
            self.to_dict(),
            indent=indent,
            default=str,
        )

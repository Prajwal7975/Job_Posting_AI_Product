"""
src/entity/salary_final_model_entity.py

Final Model Training Result Entity.

Represents the outcome of the final-model training stage.

This entity does not contain the fitted sklearn Pipeline itself.
The trained pipeline is persisted as a joblib artifact and referenced
through model_artifact_path.

The entity preserves feature, model, tuning, validation, artifact,
and MLflow lineage required by downstream stages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class SalaryFinalModelResult:
    """
    Structured result produced by SalaryFinalModelTrainer.
    """

    # --------------------------------------------------------------
    # Training status
    # --------------------------------------------------------------
    success: bool

    error: Optional[str] = None

    # --------------------------------------------------------------
    # Model identity
    # --------------------------------------------------------------
    model_name: str = ""
    model_class_name: Optional[str] = None

    # --------------------------------------------------------------
    # Feature lineage
    # --------------------------------------------------------------
    feature_experiment_id: str = ""
    feature_config_signature: Optional[str] = None

    # --------------------------------------------------------------
    # Model-family lineage
    # --------------------------------------------------------------
    model_experiment_id: Optional[str] = None
    model_config_signature: Optional[str] = None

    # --------------------------------------------------------------
    # Hyperparameter-tuning lineage
    # --------------------------------------------------------------
    tuning_config_signature: Optional[str] = None

    preferred_params: Dict[str, Any] = field(
        default_factory=dict
    )

    # --------------------------------------------------------------
    # Training performance
    # --------------------------------------------------------------
    training_seconds: float = 0.0

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------
    validation_metrics: Dict[str, float] = field(
        default_factory=dict
    )

    validation_metric: Optional[str] = None

    validation_direction: Optional[str] = None

    validation_threshold: Optional[float] = None

    validation_threshold_configured: bool = False

    validation_passed: bool = False

    # --------------------------------------------------------------
    # Model artifact
    # --------------------------------------------------------------
    model_artifact_path: Optional[str] = None

    artifact_directory: Optional[str] = None

    # --------------------------------------------------------------
    # MLflow lineage
    # --------------------------------------------------------------
    mlflow_run_id: Optional[str] = None

    # --------------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------------
    generated_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    # --------------------------------------------------------------
    # Serialization
    # --------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable representation of the result.
        """

        return {
            "success": self.success,
            "error": self.error,

            "model_name": self.model_name,
            "model_class_name": self.model_class_name,

            "feature_experiment_id": self.feature_experiment_id,
            "feature_config_signature": (
                self.feature_config_signature
            ),

            "model_experiment_id": self.model_experiment_id,
            "model_config_signature": (
                self.model_config_signature
            ),

            "tuning_config_signature": (
                self.tuning_config_signature
            ),

            "preferred_params": dict(
                self.preferred_params
            ),

            "training_seconds": self.training_seconds,

            "validation_metrics": dict(
                self.validation_metrics
            ),

            "validation_metric": self.validation_metric,
            "validation_direction": self.validation_direction,
            "validation_threshold": self.validation_threshold,
            "validation_threshold_configured": (
                self.validation_threshold_configured
            ),
            "validation_passed": self.validation_passed,

            "model_artifact_path": self.model_artifact_path,
            "artifact_directory": self.artifact_directory,

            "mlflow_run_id": self.mlflow_run_id,

            "generated_at": self.generated_at,
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Return the result as JSON.
        """

        return json.dumps(
            self.to_dict(),
            indent=indent,
            default=str,
        )
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class SalaryModelExperimentResult:
    """
    Structured result produced by exactly one salary model experiment.

    This is an entity/data contract only.

    It does not train models, save artifacts, or communicate with MLflow.
    """

    # ------------------------------------------------------------------
    # Experiment identity
    # ------------------------------------------------------------------
    experiment_id: str
    model_name: str

    # Feature configuration used by this model
    feature_experiment_id: str

    # Fingerprint of the model configuration
    config_signature: str

    # ------------------------------------------------------------------
    # Execution status
    # ------------------------------------------------------------------
    success: bool
    error: Optional[str] = None

    # ------------------------------------------------------------------
    # Model information
    # ------------------------------------------------------------------
    model_class_name: Optional[str] = None

    model_params: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Evaluation metrics
    # ------------------------------------------------------------------
    metrics: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Performance metadata
    # ------------------------------------------------------------------
    training_seconds: float = 0.0
    prediction_seconds: float = 0.0

    # ------------------------------------------------------------------
    # MLflow lineage
    # ------------------------------------------------------------------
    mlflow_run_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Local artifact lineage
    # ------------------------------------------------------------------
    artifact_directory: Optional[str] = None
    model_artifact_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------------------
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable representation of the result.
        """

        return {
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "feature_experiment_id": self.feature_experiment_id,
            "config_signature": self.config_signature,
            "success": self.success,
            "error": self.error,
            "model_class_name": self.model_class_name,
            "model_params": dict(self.model_params),
            "metrics": dict(self.metrics),
            "training_seconds": self.training_seconds,
            "prediction_seconds": self.prediction_seconds,
            "mlflow_run_id": self.mlflow_run_id,
            "artifact_directory": self.artifact_directory,
            "model_artifact_path": self.model_artifact_path,
            "generated_at": self.generated_at,
        }

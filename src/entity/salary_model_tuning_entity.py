"""
src/entity/salary_model_tuning_entity.py

Hyperparameter Tuning Entities.

Note on location: sits flat under src/entity/, matching the precedent set by 
salary_model_family_entity.py (imported as `from src.entity.salary_model_family_entity import ...`).

Two structured entities:

    SalaryModelTuningTrialResult
        One hyperparameter combination: fit once, evaluated once.
        Tracks individual trial execution, metrics, parameters, and MLflow lineage.

    SalaryModelTuningSummary
        The complete tuning run outcome: every trial, ranked trials, the best
        trial, and the explicit baseline-vs-tuned comparison.

Both are plain (mutable) @dataclasses, matching SalaryModelExperimentResult 
and SalaryModelFamilyExperimentSummary's convention — post-construction 
fields like `report_artifacts_dir`, `mlflow_summary_run_id`, and 
`best_model_artifact_path` are populated once reports are written or MLflow tracking completes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SalaryModelTuningTrialResult:
    """One hyperparameter trial combination: fit once, evaluated once."""

    trial_id: str
    model_name: str
    feature_experiment_id: str
    config_signature: str  # Tuning search space configuration signature

    success: bool
    error: Optional[str] = None

    model_class_name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    training_seconds: float = 0.0
    prediction_seconds: float = 0.0

    feature_config_signature: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    model_artifact_path: Optional[str] = None

    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "model_name": self.model_name,
            "model_class_name": self.model_class_name,
            "params": dict(self.params),
            "metrics": dict(self.metrics),
            "training_seconds": self.training_seconds,
            "prediction_seconds": self.prediction_seconds,
            "success": self.success,
            "error": self.error,
            "feature_experiment_id": self.feature_experiment_id,
            "feature_config_signature": self.feature_config_signature,
            "config_signature": self.config_signature,
            "mlflow_run_id": self.mlflow_run_id,
            "model_artifact_path": self.model_artifact_path,
            "generated_at": self.generated_at,
        }


@dataclass
class SalaryModelTuningSummary:
    """
    Complete outcome of tuning ONE winning model family, including the
    explicit baseline-vs-tuned decision — this is what lets a future
    Final Model Trainer consume `preferred_params` (or fall back to
    `baseline_params` when tuning didn't actually help) without ever
    re-running the tuning process itself.
    """

    feature_experiment_id: str
    feature_config_signature: Optional[str]

    winner_model_experiment_id: str  # e.g. "M4" — the model-family winner's id
    model_name: str
    model_class_name: Optional[str]

    baseline_score: float
    best_score: float

    ranking_metric: str
    ranking_direction: str  # "minimize" | "maximize"

    baseline_config_signature: str
    best_config_signature: str

    baseline_params: Dict[str, Any]
    best_params: Dict[str, Any]

    tuning_improved_baseline: bool
    improvement: float
    improvement_percentage: float

    total_trial_count: int
    successful_trial_count: int
    failed_trial_count: int

    best_trial_id: str

    successful_trials: Tuple[SalaryModelTuningTrialResult, ...]
    failed_trials: Tuple[SalaryModelTuningTrialResult, ...]
    ranked_trials: Tuple[SalaryModelTuningTrialResult, ...]

    execution_seconds: float
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Populated AFTER construction mirroring SalaryModelFamilyExperimentSummary
    report_artifacts_dir: Optional[str] = None
    mlflow_summary_run_id: Optional[str] = None
    best_model_artifact_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Dynamic Decision Properties
    # ------------------------------------------------------------------
    @property
    def preferred_params(self) -> Dict[str, Any]:
        """
        What the Final Model Trainer should actually use: the tuned
        params if tuning genuinely beat baseline, otherwise the baseline
        params. Never silently prefers "tuned" just because tuning ran.
        """
        return (
            dict(self.best_params)
            if self.tuning_improved_baseline
            else dict(self.baseline_params)
        )

    @property
    def preferred_config_signature(self) -> str:
        """
        Returns best_config_signature if tuning improved upon the baseline,
        otherwise falls back to baseline_config_signature.
        """
        return (
            self.best_config_signature
            if self.tuning_improved_baseline
            else self.baseline_config_signature
        )

    @property
    def preferred_model_artifact_path(self) -> Optional[str]:
        """
        Returns the best trial's model artifact path if tuning improved the baseline,
        otherwise None (signaling to downstream stages that a fresh baseline fit is preferred).
        """
        return self.best_model_artifact_path if self.tuning_improved_baseline else None

    # ------------------------------------------------------------------
    # Serialization Methods
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_experiment_id": self.feature_experiment_id,
            "feature_config_signature": self.feature_config_signature,
            "winner_model_experiment_id": self.winner_model_experiment_id,
            "model_name": self.model_name,
            "model_class_name": self.model_class_name,
            "ranking_metric": self.ranking_metric,
            "ranking_direction": self.ranking_direction,
            "baseline_score": self.baseline_score,
            "best_score": self.best_score,
            "baseline_config_signature": self.baseline_config_signature,
            "best_config_signature": self.best_config_signature,
            "baseline_params": dict(self.baseline_params),
            "best_params": dict(self.best_params),
            "preferred_params": dict(self.preferred_params),
            "preferred_config_signature": self.preferred_config_signature,
            "preferred_model_artifact_path": self.preferred_model_artifact_path,
            "tuning_improved_baseline": self.tuning_improved_baseline,
            "improvement": self.improvement,
            "improvement_percentage": self.improvement_percentage,
            "best_trial_id": self.best_trial_id,
            "total_trial_count": self.total_trial_count,
            "successful_trial_count": self.successful_trial_count,
            "failed_trial_count": self.failed_trial_count,
            "execution_seconds": self.execution_seconds,
            "generated_at": self.generated_at,
            "report_artifacts_dir": self.report_artifacts_dir,
            "mlflow_summary_run_id": self.mlflow_summary_run_id,
            "best_model_artifact_path": self.best_model_artifact_path,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
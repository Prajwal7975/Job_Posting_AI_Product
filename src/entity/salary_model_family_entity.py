from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.entity.single_salary_model_entity import (
    SalaryModelExperimentResult,
)


@dataclass
class SalaryModelFamilyExperimentSummary:
    """
    Structured summary of a complete model-family comparison run.

    Represents the outcome of comparing multiple model families while
    keeping the selected feature configuration fixed.
    """

    # ------------------------------------------------------------------
    # Required feature lineage
    # ------------------------------------------------------------------
    feature_experiment_id: str

    # ------------------------------------------------------------------
    # Required ranking configuration
    # ------------------------------------------------------------------
    ranking_metric: str
    ranking_direction: str

    # ------------------------------------------------------------------
    # Required experiment counts
    # ------------------------------------------------------------------
    experiment_count: int
    successful_experiment_count: int
    failed_experiment_count: int

    # ------------------------------------------------------------------
    # Optional feature lineage
    # ------------------------------------------------------------------
    feature_config_signature: Optional[str] = None

    # ------------------------------------------------------------------
    # Winner
    # ------------------------------------------------------------------
    winner_experiment_id: Optional[str] = None
    winner_model_name: Optional[str] = None
    winner_model_class: Optional[str] = None
    winner_score: Optional[float] = None
    winner_config_signature: Optional[str] = None

    # Direct lineage to the winning trained model
    winner_model_artifact_path: Optional[str] = None
    winner_mlflow_run_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Experiment results
    # ------------------------------------------------------------------
    successful_results: List[SalaryModelExperimentResult] = field(
        default_factory=list
    )

    failed_results: List[SalaryModelExperimentResult] = field(
        default_factory=list
    )

    ranked_results: List[SalaryModelExperimentResult] = field(
        default_factory=list
    )

    # ------------------------------------------------------------------
    # Execution metadata
    # ------------------------------------------------------------------
    execution_seconds: float = 0.0

    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ------------------------------------------------------------------
    # Reporting / MLflow lineage
    # ------------------------------------------------------------------
    report_artifacts_dir: Optional[str] = None
    mlflow_summary_run_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Return a compact JSON-serializable summary.

        Detailed experiment results are intentionally excluded because
        they are stored separately in individual experiment artifacts
        and comparison reports.
        """

        data = asdict(self)

        data.pop("successful_results", None)
        data.pop("failed_results", None)
        data.pop("ranked_results", None)

        return data
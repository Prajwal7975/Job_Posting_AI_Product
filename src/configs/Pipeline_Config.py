from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PipelineConfig:
    """
    Global filesystem configuration shared across the entire pipeline.

    Every pipeline stage should derive its input/output paths from this
    configuration rather than hard-coding directory names.
    """

    # Project root
    project_root: Path = PROJECT_ROOT

    # Base directories
    data_dir: Path = PROJECT_ROOT / "data"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    models_dir: Path = PROJECT_ROOT / "models"
    logs_dir: Path = PROJECT_ROOT / "logs"

    # Artifact directories
    summary_reports_dir: Path = PROJECT_ROOT / "artifacts" / "summary_reports"
    master_dataset_dir: Path = PROJECT_ROOT / "artifacts" / "master_dataset"

    # Future stages
    feature_store_dir: Path = PROJECT_ROOT / "artifacts" / "feature_store"
    mlflow_dir: Path = PROJECT_ROOT / "artifacts" / "mlruns"
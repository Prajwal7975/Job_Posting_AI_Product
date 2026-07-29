from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class PipelineConfig:
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    logs_dir: Path = PROJECT_ROOT / "logs"
    models_dir: Path = PROJECT_ROOT / "models"
    data_dir: Path = PROJECT_ROOT / "data"
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Environment variables this module knows how to read. Every one of them
# is optional -- local development works with zero environment setup.
ENV_TRACKING_URI = "MLFLOW_TRACKING_URI"
ENV_EXPERIMENT_NAME = "SALARY_MLFLOW_EXPERIMENT_NAME"
ENV_TRACKING_ENABLED = "SALARY_MLFLOW_TRACKING_ENABLED"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _local_path_to_tracking_uri(path: Path) -> str:
    return path.resolve().as_uri()


def _default_tracking_uri() -> str:
    env_value = os.environ.get(ENV_TRACKING_URI)
    if env_value:
        return env_value
    return _local_path_to_tracking_uri(_project_root() / "mlruns")


def _default_experiment_name() -> str:
    """Local default experiment name, overridable via env for CI/remote use."""
    return os.environ.get(ENV_EXPERIMENT_NAME, "salary_prediction")


def _parse_bool_env(env_var: str, default: bool) -> bool:
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"Invalid boolean value for environment variable {env_var!r}: {raw!r}. "
        f"Expected one of {_TRUE_VALUES | _FALSE_VALUES}."
    )


def _default_tracking_enabled() -> bool:
    return _parse_bool_env(ENV_TRACKING_ENABLED, default=True)


def _is_blank(value: str) -> bool:
    return not value or not value.strip()


def _is_malformed_tracking_uri(value: str) -> bool:
    if _is_blank(value):
        return True
    if "://" in value:
        parsed = urlparse(value)
        if not parsed.scheme or not (parsed.netloc or parsed.path):
            return True
    return False


@dataclass(frozen=True)
class SalaryMLflowConfig:

    tracking_uri: str = field(default_factory=_default_tracking_uri)
    experiment_name: str = field(default_factory=_default_experiment_name)
    project_name: str = "linkedin-job-intelligence"
    pipeline_name: str = "salary_predict"
    run_name_prefix: str = "salary"
    artifact_location: Optional[str] = None
    registered_model_name: str = "salary_prediction_model"
    max_parameter_length: int = 250
    tracking_enabled: bool = field(default_factory=_default_tracking_enabled)

    def __post_init__(self) -> None:
        # --- Validate tracking_uri type first ---
        if not isinstance(self.tracking_uri, str):
            raise TypeError(
                f"tracking_uri must be a string, got {type(self.tracking_uri).__name__}."
            )
            
        if _is_malformed_tracking_uri(self.tracking_uri):
            raise ValueError(
                f"tracking_uri is empty or malformed: {self.tracking_uri!r}"
            )
            
        if _is_blank(self.experiment_name):
            raise ValueError("experiment_name must not be empty.")
        if _is_blank(self.project_name):
            raise ValueError("project_name must not be empty.")
        if _is_blank(self.pipeline_name):
            raise ValueError("pipeline_name must not be empty.")
        if _is_blank(self.run_name_prefix):
            raise ValueError("run_name_prefix must not be empty.")
        if _is_blank(self.registered_model_name):
            raise ValueError("registered_model_name must not be empty.")
            
        # --- Validate artifact_location type first ---
        if self.artifact_location is not None:
            if not isinstance(self.artifact_location, str):
                raise TypeError(
                    f"artifact_location must be a string, got {type(self.artifact_location).__name__}."
                )
            if _is_blank(self.artifact_location):
                raise ValueError(
                    "artifact_location must be either None or a non-empty string."
                )
            
        if not isinstance(self.tracking_enabled, bool):
            raise TypeError(
                f"tracking_enabled must be a bool, got {type(self.tracking_enabled).__name__}."
            )
            
        # --- Validate max_parameter_length type first ---
        if not isinstance(self.max_parameter_length, int):
            raise TypeError(
                f"max_parameter_length must be an int, got {type(self.max_parameter_length).__name__}."
            )
        if self.max_parameter_length <= 0:
            raise ValueError("max_parameter_length must be greater than zero.")

    # -- read-only helpers ---------------------------------------------

    @property
    def is_tracking_enabled(self) -> bool:
        """Whether the future tracker should record runs at all."""
        return self.tracking_enabled

    @property
    def is_local_tracking(self) -> bool:
        """Whether ``tracking_uri`` points at a local filesystem store."""
        return self.tracking_uri.startswith("file://") or "://" not in self.tracking_uri

    @property
    def local_tracking_path(self) -> Optional[Path]:
        if not self.is_local_tracking:
            return None
        if self.tracking_uri.startswith("file://"):
            parsed = urlparse(self.tracking_uri)
            decoded_path = unquote(parsed.path)
            return Path(url2pathname(decoded_path))
        return Path(self.tracking_uri)

    def default_tags(self) -> Mapping[str, str]:
        return {
            "project": self.project_name,
            "pipeline": self.pipeline_name,
        }

    def build_run_name(self, suffix: str) -> str:
        if _is_blank(suffix):
            raise ValueError("suffix must not be empty.")
        return f"{self.run_name_prefix}-{suffix.strip()}"
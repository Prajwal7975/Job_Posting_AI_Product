"""
src/configs/salary_predict/salary_model_experiment_config.py

Model Family Experiment Configuration.

Separate and distinct from SalaryExperimentConfig (which bundles feature
selection AND a model choice together for the feature-engineering
ablation stage E0-E3B). This config represents ONLY: "which algorithm,
with which baseline parameters" — used by the model-family comparison
stage, which holds the winning feature configuration (e.g. E3B) FIXED and
varies only the estimator.

Like salary_experiment_config.py, this module is a pure standard-library
configuration contract: no sklearn/pandas/MLflow imports, nothing here
trains or instantiates a real estimator. That happens in
SalaryModelFactory, reading model_name/model_params from an instance of
this class exactly the same way it already reads them from
SalaryExperimentConfig.

Note: Model support/availability is explicitly NOT validated here. The
SalaryModelFactory is the single source of truth for the model registry.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

CONFIG_VERSION = "1.0"


@dataclass(frozen=True)
class SalaryModelExperimentConfig:
    model_experiment_id: str
    model_experiment_name: str
    model_name: str
    description: str

    model_params: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True
    random_state: Optional[int] = None
    config_version: str = CONFIG_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_experiment_id, str)
            or not self.model_experiment_id.strip()
        ):
            raise ValueError("model_experiment_id must be a non-empty string.")
            
        if (
            not isinstance(self.model_experiment_name, str)
            or not self.model_experiment_name.strip()
        ):
            raise ValueError("model_experiment_name must be a non-empty string.")
            
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError(f"model_name must be a non-empty string, got {self.model_name!r}.")

        normalized_model_name = self.model_name.strip().lower()

        if self.random_state is not None and not isinstance(self.random_state, int):
            raise ValueError(
                "random_state must be an integer or None, "
                f"got {type(self.random_state).__name__}."
            )

        if not isinstance(self.model_params, Mapping):
            raise ValueError(f"model_params must be a mapping, got {type(self.model_params).__name__}.")
            
        non_string_keys = [
            key for key in self.model_params.keys() if not isinstance(key, str)
        ]
        if non_string_keys:
            raise ValueError(
                "model_params keys must all be strings, "
                f"got non-string keys: {non_string_keys}"
            )

        # Normalize model_name to lowercase, and — only when random_state
        # was explicitly given as a constructor argument and isn't already
        # present in model_params — fold it in. This is a deliberate,
        # visible normalization of an argument the caller explicitly
        # supplied, not a hidden default the caller never asked for.
        merged_params: Dict[str, Any] = dict(self.model_params)
        if self.random_state is not None and "random_state" not in merged_params:
            merged_params["random_state"] = self.random_state

        object.__setattr__(self, "model_name", normalized_model_name)
        object.__setattr__(self, "model_params", MappingProxyType(merged_params))

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_version": self.config_version,
            "model_experiment_id": self.model_experiment_id,
            "model_experiment_name": self.model_experiment_name,
            "model_name": self.model_name,
            "description": self.description,
            "model_params": dict(self.model_params),
            "enabled": self.enabled,
        }

    @property
    def config_signature(self) -> str:
        """
        Deterministic fingerprint of everything that affects model
        behavior. Excludes model_experiment_id/name/description/enabled —
        identity/orchestration fields, not behavior.
        """
        payload = {
            "config_version": self.config_version,
            "model_name": self.model_name,
            "model_params": dict(self.model_params),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ======================================================================
# PREDEFINED MODEL EXPERIMENTS — M0..M5
# ======================================================================
#
# alpha/n_estimators etc. below are deliberately simple, stable baselines
# — this stage compares ALGORITHM FAMILIES, not tuned hyperparameters.
# Tuning is explicitly the NEXT stage's job, not this one's.


def build_m0_config() -> SalaryModelExperimentConfig:
    return SalaryModelExperimentConfig(
        model_experiment_id="M0",
        model_experiment_name="dummy_median_baseline",
        model_name="dummy",
        description="Naive median baseline — the floor every real model family must beat.",
        model_params={"strategy": "median"},
    )


def build_m1_config() -> SalaryModelExperimentConfig:
    return SalaryModelExperimentConfig(
        model_experiment_id="M1",
        model_experiment_name="ridge_baseline",
        model_name="ridge",
        description="Ridge regression, same alpha used throughout the feature-experiment stage.",
        model_params={"alpha": 1.0},
    )


def build_m2_config() -> SalaryModelExperimentConfig:
    return SalaryModelExperimentConfig(
        model_experiment_id="M2",
        model_experiment_name="random_forest_baseline",
        model_name="random_forest",
        description="RandomForestRegressor, modest baseline size (not tuned).",
        model_params={"n_estimators": 200, "max_depth": None, "n_jobs": -1},
        random_state=42,
    )


def build_m3_config() -> SalaryModelExperimentConfig:
    return SalaryModelExperimentConfig(
        model_experiment_id="M3",
        model_experiment_name="lightgbm_baseline",
        model_name="lightgbm",
        description="LightGBM baseline configuration for model-family comparison; not hyperparameter tuned.",
        model_params={"n_estimators": 300, "learning_rate": 0.05},
        random_state=42,
    )


def build_m4_config() -> SalaryModelExperimentConfig:
    return SalaryModelExperimentConfig(
        model_experiment_id="M4",
        model_experiment_name="xgboost_baseline",
        model_name="xgboost",
        description="XGBoost baseline configuration for model-family comparison; not hyperparameter tuned.",
        model_params={"n_estimators": 300, "learning_rate": 0.05},
        random_state=42,
    )


def build_m5_config() -> SalaryModelExperimentConfig:
    return SalaryModelExperimentConfig(
        model_experiment_id="M5",
        model_experiment_name="catboost_baseline",
        model_name="catboost",
        description="CatBoost baseline configuration for model-family comparison; not hyperparameter tuned.",
        model_params={"iterations": 300, "learning_rate": 0.05, "verbose": False,"allow_writing_files": False},
        random_state=42,
    )


_MODEL_EXPERIMENT_BUILDERS: Dict[str, Callable[[], SalaryModelExperimentConfig]] = {
    "M0": build_m0_config,
    "M1": build_m1_config,
    "M2": build_m2_config,
    "M3": build_m3_config,
    "M4": build_m4_config,
    "M5": build_m5_config,
}


def get_all_model_experiment_configs() -> Tuple[SalaryModelExperimentConfig, ...]:
    """
    Every registered model experiment, regardless of `enabled`.
    Returns experiments deterministically sorted by their ID (M0, M1, M2...).
    """
    return tuple(
        _MODEL_EXPERIMENT_BUILDERS[experiment_id]()
        for experiment_id in sorted(_MODEL_EXPERIMENT_BUILDERS)
    )


def get_model_experiment_configs() -> Tuple[SalaryModelExperimentConfig, ...]:
    """Only `enabled=True` model experiments — what the orchestrator actually runs."""
    return tuple(config for config in get_all_model_experiment_configs() if config.enabled)


def get_model_experiment_config(model_experiment_id: str) -> SalaryModelExperimentConfig:
    builder = _MODEL_EXPERIMENT_BUILDERS.get(model_experiment_id)
    if builder is None:
        raise KeyError(
            f"Unknown model_experiment_id '{model_experiment_id}'. "
            f"Available: {sorted(_MODEL_EXPERIMENT_BUILDERS)}"
        )
    return builder()
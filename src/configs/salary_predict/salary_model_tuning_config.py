"""
src/configs/salary_predict/salary_model_tuning_config.py

Hyperparameter Tuning Configuration.

Defines WHAT a tuning run should search: which model family, which
search strategy, which hyperparameter space, which trial budget. Pure
standard-library configuration contract — no pandas/sklearn/MLflow
imports, nothing here runs a trial or touches a dataset.

Model-specific search spaces live in DATA (_DEFAULT_SEARCH_SPACES below),
not in branching logic — this is what lets the tuning runner stay
model-agnostic (`if model_name == "xgboost": ...` never needs to exist
anywhere in the runner, only in this lookup table).
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

CONFIG_VERSION = "1.0"

# Matches exactly what SalarySingleModelExperimentRunner actually computes
# (MAE/RMSE/R2). Deliberately does NOT include "MAPE" — the model-family
# runner's _METRIC_DIRECTIONS lists it as supported, but nothing in the
# pipeline ever computes it, so listing it here would validate a metric
# that can never actually be ranked on.
SUPPORTED_RANKING_METRICS: Tuple[str, ...] = ("MAE", "RMSE", "R2")

_METRIC_DIRECTIONS: Dict[str, str] = {
    "MAE": "minimize",
    "RMSE": "minimize",
    "R2": "maximize",
}

SUPPORTED_SEARCH_STRATEGIES: Tuple[str, ...] = ("grid", "random")


def ranking_direction_for(ranking_metric: str) -> str:
    """ "minimize" or "maximize" for a supported ranking metric."""
    if not isinstance(ranking_metric, str):
        raise ValueError(
            f"ranking_metric must be a string, got {type(ranking_metric).__name__}."
        )
    metric = ranking_metric.strip().upper()
    if metric not in _METRIC_DIRECTIONS:
        raise ValueError(
            f"Unsupported ranking_metric '{ranking_metric}'. "
            f"Supported: {list(SUPPORTED_RANKING_METRICS)}"
        )
    return _METRIC_DIRECTIONS[metric]


@dataclass(frozen=True)
class SalaryModelTuningConfig:
    tuning_config_id: str
    model_name: str
    description: str

    search_strategy: str = "grid"
    search_space: Mapping[str, Sequence[Any]] = field(default_factory=dict)

    ranking_metric: str = "RMSE"
    max_trials: Optional[int] = None
    random_state: int = 42

    enabled: bool = True
    config_version: str = CONFIG_VERSION

    def __post_init__(self) -> None:
        # 1. Type and non-empty check for identity/description strings
        if (
            not isinstance(self.tuning_config_id, str)
            or not self.tuning_config_id.strip()
        ):
            raise ValueError("tuning_config_id must be a non-empty string.")

        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string.")

        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError(
                f"model_name must be a non-empty string, got {self.model_name!r}."
            )

        normalized_model_name = self.model_name.strip().lower()
        if normalized_model_name == "dummy":
            raise ValueError(
                "model_name cannot be 'dummy' — the median baseline has no "
                "meaningful hyperparameters to tune. If 'dummy' won the "
                "model-family comparison, something upstream needs "
                "investigation before tuning is the right next step."
            )

        # 2. Strategy validation
        if not isinstance(self.search_strategy, str):
            raise ValueError(
                f"search_strategy must be a string, got {type(self.search_strategy).__name__}."
            )

        strategy = self.search_strategy.strip().lower()
        if strategy not in SUPPORTED_SEARCH_STRATEGIES:
            raise ValueError(
                f"Unsupported search_strategy '{self.search_strategy}'. "
                f"Supported: {list(SUPPORTED_SEARCH_STRATEGIES)}"
            )

        # 3. Metric validation
        if not isinstance(self.ranking_metric, str):
            raise ValueError(
                "ranking_metric must be a string, "
                f"got {type(self.ranking_metric).__name__}."
            )

        ranking_metric = self.ranking_metric.strip().upper()

        if not ranking_metric:
            raise ValueError("ranking_metric must not be empty.")

        ranking_direction_for(ranking_metric)

        # 4. Search space validation
        if not isinstance(self.search_space, Mapping) or not self.search_space:
            raise ValueError(
                "search_space must be a non-empty mapping of hyperparameter -> candidate values."
            )

        normalized_space: Dict[str, Tuple[Any, ...]] = {}
        for param_name, candidates in self.search_space.items():
            if not isinstance(param_name, str) or not param_name.strip():
                raise ValueError(
                    f"search_space keys must be non-empty strings, got {param_name!r}."
                )
            if (
                not isinstance(candidates, Sequence)
                or isinstance(candidates, (str, bytes))
                or len(candidates) == 0
            ):
                raise ValueError(
                    f"search_space['{param_name}'] must be a non-empty sequence of candidate "
                    f"values, got {candidates!r}."
                )
            normalized_space[param_name] = tuple(candidates)

        # 5. Search strategy vs. trial budget semantics
        if strategy == "grid":
            if self.max_trials is not None:
                raise ValueError(
                    "max_trials should not be set when search_strategy='grid'. "
                    "Grid search evaluates the full hyperparameter space. "
                    "Use search_strategy='random' if you want to cap the trial budget with max_trials."
                )
        elif strategy == "random":
            if (
                self.max_trials is None
                or not isinstance(self.max_trials, int)
                or isinstance(self.max_trials, bool)
                or self.max_trials <= 0
            ):
                raise ValueError(
                    "max_trials must be a positive integer when search_strategy='random', "
                    f"got {self.max_trials!r}."
                )

        # 6. Reject bools for random_state
        if not isinstance(self.random_state, int) or isinstance(
            self.random_state, bool
        ):
            raise ValueError(
                f"random_state must be an int, got {type(self.random_state).__name__}."
            )

        object.__setattr__(self, "model_name", normalized_model_name)
        object.__setattr__(self, "search_strategy", strategy)
        object.__setattr__(self, "ranking_metric", ranking_metric)
        object.__setattr__(self, "search_space", MappingProxyType(normalized_space))

    # ------------------------------------------------------------------
    @property
    def ranking_direction(self) -> str:
        return ranking_direction_for(self.ranking_metric)

    @property
    def search_space_size(self) -> int:
        """Total number of distinct hyperparameter combinations (full grid)."""
        size = 1
        for candidates in self.search_space.values():
            size *= len(candidates)
        return size

    def generate_param_grid(self) -> Tuple[Dict[str, Any], ...]:
        """
        Deterministic, ordered list of hyperparameter combinations described
        by this config. For "grid", this evaluates the entire Cartesian product.
        For "random", this returns a deterministic pseudo-random sample of size
        max_trials, seeded by random_state.
        """
        keys = tuple(sorted(self.search_space.keys()))
        all_combos = tuple(
            dict(zip(keys, values))
            for values in itertools.product(*(self.search_space[k] for k in keys))
        )

        if self.search_strategy == "grid":
            return all_combos

        # search_strategy == "random"
        import random as _random

        rng = _random.Random(self.random_state)
        indices = list(range(len(all_combos)))
        rng.shuffle(indices)
        sample_size = min(self.max_trials, len(all_combos))
        return tuple(all_combos[i] for i in indices[:sample_size])

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_version": self.config_version,
            "tuning_config_id": self.tuning_config_id,
            "model_name": self.model_name,
            "description": self.description,
            "search_strategy": self.search_strategy,
            "search_space": {k: list(v) for k, v in self.search_space.items()},
            "search_space_size": self.search_space_size,
            "ranking_metric": self.ranking_metric,
            "ranking_direction": self.ranking_direction,
            "max_trials": self.max_trials,
            "random_state": self.random_state,
            "enabled": self.enabled,
        }

    @property
    def config_signature(self) -> str:
        """
        Deterministic fingerprint of everything that affects which trials
        get run and how they're ranked. Excludes tuning_config_id and
        description — identity/display fields, not behavior.
        """
        payload = {
            "config_version": self.config_version,
            "model_name": self.model_name,
            "search_strategy": self.search_strategy,
            "search_space": {k: list(v) for k, v in self.search_space.items()},
            "ranking_metric": self.ranking_metric,
            "max_trials": self.max_trials,
            "random_state": self.random_state,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ======================================================================
# DEFAULT SEARCH SPACES — one entry per tunable model family
# ======================================================================

_DEFAULT_SEARCH_SPACES: Dict[str, Dict[str, Tuple[Any, ...]]] = {
    "ridge": {
        "alpha": (0.01, 0.1, 1.0, 10.0, 100.0),
    },
    "random_forest": {
        "n_estimators": (100, 200, 400),
        "max_depth": (None, 10, 20),
        "min_samples_leaf": (1, 2, 4),
    },
    "lightgbm": {
        "n_estimators": (200, 300, 500, 800),
        "learning_rate": (0.01, 0.03, 0.05, 0.1),
        "num_leaves": (31, 63, 127),
    },
    "xgboost": {
        "n_estimators": (200, 300, 500, 800),
        "learning_rate": (0.01, 0.03, 0.05, 0.1),
        "max_depth": (3, 5, 7),
        "subsample": (0.7, 0.8, 1.0),
    },
    "catboost": {
        "iterations": (200, 300, 500),
        "learning_rate": (0.01, 0.03, 0.05, 0.1),
        "depth": (4, 6, 8),
    },
}


def build_default_tuning_config(
    model_name: str,
    tuning_config_id: Optional[str] = None,
    ranking_metric: str = "RMSE",
    search_strategy: str = "grid",
    max_trials: Optional[int] = None,
    random_state: int = 42,
) -> SalaryModelTuningConfig:
    """
    Build a tuning config for `model_name` using its registered default
    search space. Raises KeyError with a clear message if the model has no
    default space registered.
    """
    normalized = model_name.strip().lower()
    search_space = _DEFAULT_SEARCH_SPACES.get(normalized)
    if search_space is None:
        raise KeyError(
            f"No default tuning search space registered for model_name='{model_name}'. "
            f"Registered: {sorted(_DEFAULT_SEARCH_SPACES)}. "
            "Add an entry to _DEFAULT_SEARCH_SPACES, or construct "
            "SalaryModelTuningConfig directly with an explicit search_space."
        )

    return SalaryModelTuningConfig(
        tuning_config_id=tuning_config_id or f"TUNE_{normalized.upper()}",
        model_name=normalized,
        description=f"Default hyperparameter search space for {normalized}.",
        search_strategy=search_strategy,
        search_space=search_space,
        ranking_metric=ranking_metric,
        max_trials=max_trials,
        random_state=random_state,
    )


def get_tuning_config_for_winner(
    model_family_summary: Any, **overrides: Any
) -> SalaryModelTuningConfig:
    """
    Build the default tuning config for whichever model actually won the
    model-family comparison. Accepts any object exposing `winner_model_name`
    (duck-typed to avoid circular imports).
    """
    winner_model_name = getattr(model_family_summary, "winner_model_name", None)
    if not winner_model_name:
        raise ValueError(
            "model_family_summary has no winner_model_name — cannot determine "
            "which model family to tune."
        )
    return build_default_tuning_config(model_name=winner_model_name, **overrides)


def supported_tunable_models() -> Tuple[str, ...]:
    return tuple(sorted(_DEFAULT_SEARCH_SPACES))

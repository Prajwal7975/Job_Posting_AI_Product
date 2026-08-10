"""
src/configs/salary_predict/salary_final_model_config.py

Final Model Training Configuration.

Controls only orchestration behavior for the final-model training stage:

- artifact locations
- validation metric
- validation quality threshold
- MLflow experiment intent
- random-state configuration
- overwrite behavior

It does NOT contain model hyperparameters, feature-selection settings,
or tuning search spaces.

Model hyperparameters come from:
    SalaryModelTuningSummary.preferred_params

Feature configuration comes from:
    the winning SalaryExperimentConfig

Model construction is delegated to:
    SalaryModelFactory

MLflow tracking is delegated to:
    SalaryMLflowTracker
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.configs.salary_predict.salary_model_tuning_config import (
    SUPPORTED_RANKING_METRICS,
    ranking_direction_for,
)

CONFIG_VERSION = "1.0"


@dataclass(frozen=True)
class SalaryFinalModelConfig:
    """
    Configuration contract for the final model-training stage.

    This class controls final-training orchestration only. It does not
    decide which model or hyperparameters should be used.
    """

    config_version: str = CONFIG_VERSION

    # --------------------------------------------------------------
    # Artifact configuration
    # --------------------------------------------------------------
    artifact_dir: str = "artifacts/salary_final_model"

    model_filename: str = "final_model.joblib"

    # --------------------------------------------------------------
    # Validation / quality gate
    # --------------------------------------------------------------
    validation_metric: str = "RMSE"

    validation_threshold: Optional[float] = None

    # --------------------------------------------------------------
    # MLflow intent
    # --------------------------------------------------------------
    mlflow_experiment_name: str = "salary_final_model"

    # --------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------
    random_state: Optional[int] = None

    # --------------------------------------------------------------
    # Artifact overwrite policy
    # --------------------------------------------------------------
    overwrite_existing: bool = False

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------
    def __post_init__(self) -> None:

        # ----------------------------------------------------------
        # validation_metric
        # ----------------------------------------------------------
        metric = (
            self.validation_metric.strip().upper()
            if isinstance(self.validation_metric, str)
            else None
        )

        if metric not in SUPPORTED_RANKING_METRICS:
            raise ValueError(
                f"Unsupported validation_metric "
                f"'{self.validation_metric}'. "
                f"Supported: {list(SUPPORTED_RANKING_METRICS)}"
            )

        object.__setattr__(self, "validation_metric", metric)

        # ----------------------------------------------------------
        # artifact_dir
        # ----------------------------------------------------------
        if (
            not isinstance(self.artifact_dir, str)
            or not self.artifact_dir.strip()
        ):
            raise ValueError(
                "artifact_dir must be a non-empty string."
            )

        # ----------------------------------------------------------
        # model_filename
        # ----------------------------------------------------------
        if (
            not isinstance(self.model_filename, str)
            or not self.model_filename.strip()
        ):
            raise ValueError(
                "model_filename must be a non-empty string."
            )

        # ----------------------------------------------------------
        # validation_threshold
        # ----------------------------------------------------------
        if self.validation_threshold is not None:

            if isinstance(self.validation_threshold, bool):
                raise ValueError(
                    "validation_threshold must not be a bool."
                )

            if not isinstance(
                self.validation_threshold,
                (int, float),
            ):
                raise ValueError(
                    "validation_threshold must be numeric or None, "
                    f"got {type(self.validation_threshold).__name__}."
                )

        # ----------------------------------------------------------
        # mlflow_experiment_name
        # ----------------------------------------------------------
        if (
            not isinstance(self.mlflow_experiment_name, str)
            or not self.mlflow_experiment_name.strip()
        ):
            raise ValueError(
                "mlflow_experiment_name must be a "
                "non-empty string."
            )

        # ----------------------------------------------------------
        # random_state
        # ----------------------------------------------------------
        if self.random_state is not None:

            if (
                not isinstance(self.random_state, int)
                or isinstance(self.random_state, bool)
            ):
                raise ValueError(
                    "random_state must be an int or None, "
                    f"got {type(self.random_state).__name__}."
                )

        # ----------------------------------------------------------
        # overwrite_existing
        # ----------------------------------------------------------
        if not isinstance(self.overwrite_existing, bool):
            raise ValueError(
                "overwrite_existing must be a bool, "
                f"got {type(self.overwrite_existing).__name__}."
            )

    # --------------------------------------------------------------
    # Derived validation direction
    # --------------------------------------------------------------
    @property
    def validation_direction(self) -> str:
        """
        Return the optimization direction for validation_metric.

        Examples:
            RMSE → minimize
            MAE  → minimize
            R2   → maximize
        """
        return ranking_direction_for(self.validation_metric)
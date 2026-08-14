from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn

# These custom transformers are required when reconstructing
# the registered sklearn pipeline.
from src.components.salary_predict.salary_preprocessor_builder import (
    SafeCategoricalTransformer,
    SafeTextTransformer,
)

from src.logger import logging

from .salary_model_config import SalaryServingConfig


class SalaryModelLoader:
    """
    Loads and serves the exact production sklearn model
    registered in MLflow.

    The loaded artifact contains:

        fitted preprocessing pipeline
                    +
        fitted Ridge estimator

    Therefore inference does not recreate preprocessing
    or model parameters.
    """

    def __init__(
        self,
        config: SalaryServingConfig | None = None,
    ) -> None:

        self.config = config or SalaryServingConfig()

        self._model: Any = None

    # ==========================================================
    # LOAD
    # ==========================================================

    def load(self) -> Any:

        if self._model is not None:
            return self._model

        logging.info(
            "Loading production salary model from MLflow."
        )

        logging.info(
            "Tracking URI: %s",
            self.config.tracking_uri,
        )

        logging.info(
            "Model URI: %s",
            self.config.model_uri,
        )

        mlflow.set_tracking_uri(
            self.config.tracking_uri
        )

        try:
            self._model = mlflow.sklearn.load_model(
                self.config.model_uri
            )

        except Exception as exc:
            logging.exception(
                "Failed to load salary model from MLflow."
            )
            raise RuntimeError(
                "Unable to load the registered salary model "
                f"from URI: {self.config.model_uri}"
            ) from exc

        logging.info(
            "Production salary model loaded successfully."
        )

        return self._model

    # ==========================================================
    # PREDICT
    # ==========================================================

    def predict(
        self,
        features: Any,
    ) -> Any:

        model = self.load()

        logging.info(
            "Running prediction using registered salary model."
        )

        return model.predict(features)

    # ==========================================================
    # HEALTH
    # ==========================================================

    @property
    def is_loaded(self) -> bool:

        return self._model is not None
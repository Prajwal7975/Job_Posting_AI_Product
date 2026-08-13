from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn

# IMPORTANT:
# The registered model contains custom transformers.
# Importing the module makes those classes available when MLflow/skops
# reconstructs the pipeline.
from src.components.salary_predict.salary_preprocessor_builder import (
    SafeCategoricalTransformer,
    SafeTextTransformer,
)

from src.configs.salary_predict.salary_ML_flow_config import (
    SalaryMLflowConfig,
)

from src.logger import logging

from .salary_model_config import SalaryServingConfig


class SalaryModelLoader:

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

        logging.info("Loading production salary model from MLflow.")

        logging.info(
            "Tracking URI: %s",
            self.config.tracking_uri,
        )

        logging.info(
            "Model URI: %s",
            self.config.model_uri,
        )

        mlflow.set_tracking_uri(self.config.tracking_uri)

        self._model = mlflow.sklearn.load_model(self.config.model_uri)

        logging.info("Production salary model loaded successfully.")

        return self._model

    # ==========================================================
    # PREDICT
    # ==========================================================

    def predict(
        self,
        features: Any,
    ) -> Any:

        model = self.load()

        return model.predict(features)

    # ==========================================================
    # HEALTH
    # ==========================================================

    @property
    def is_loaded(self) -> bool:

        return self._model is not None

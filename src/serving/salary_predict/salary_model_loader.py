from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# These custom transformers are required when reconstructing
# the registered sklearn pipeline.
from src.components.salary_predict.salary_preprocessor_builder import (
    SafeCategoricalTransformer,
    SafeTextTransformer,
)

from src.logger import logging

from .salary_model_config import SalaryServingConfig

# Tag keys checked, in priority order, for the underlying algorithm
# family (e.g. "ridge", "random_forest"). These match the tagging
# conventions already used elsewhere in the project's MLflow tracking
# (SalaryMLflowTracker logs "model_family"/"model_class" tags on the
# training run; SalaryModelRegistry persists caller-supplied metadata as
# tags on the model version). Registry/training code is NOT modified by
# this change -- if none of these tags are present on a given deployment,
# model_name safely falls back to "unknown" rather than fabricating a
# value.
_MODEL_NAME_TAG_KEYS: tuple[str, ...] = ("model_family", "model_name", "model_class")

_UNKNOWN_MODEL_NAME = "unknown"


@dataclass(frozen=True)
class SalaryModelMetadata:
    """
    Metadata resolved alongside the loaded model. Every field here is
    either a config value the API already had, or something read
    directly from the MLflow registry at load time -- nothing is
    fabricated. `model_name` falls back to `_UNKNOWN_MODEL_NAME` when it
    genuinely cannot be determined (see _MODEL_NAME_TAG_KEYS above).
    """

    registered_model_name: str
    model_alias: str
    model_name: str
    model_version: Optional[str] = None
    run_id: Optional[str] = None


class SalaryModelLoader:
    """
    Loads and serves the exact production sklearn model
    registered in MLflow.

    The loaded artifact contains:

        fitted preprocessing pipeline
                    +
        fitted estimator

    Therefore inference does not recreate preprocessing
    or model parameters.
    """

    def __init__(
        self,
        config: SalaryServingConfig | None = None,
    ) -> None:

        self.config = config or SalaryServingConfig()

        self._model: Any = None
        self._metadata: Optional[SalaryModelMetadata] = None

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
            self.config.resolved_tracking_uri,
        )

        logging.info(
            "Model URI: %s",
            self.config.model_uri,
        )

        mlflow.set_tracking_uri(
            self.config.resolved_tracking_uri
        )

        try:
            self._model = mlflow.sklearn.load_model(
                self.config.model_uri
            )
            self._metadata = self._resolve_metadata()

        except Exception as exc:
            logging.exception(
                "Failed to load salary model from MLflow."
            )
            raise RuntimeError(
                "Unable to load the registered salary model "
                f"from URI: {self.config.model_uri}"
            ) from exc

        logging.info(
            "Production salary model loaded successfully "
            "(model_name=%s, version=%s).",
            self._metadata.model_name,
            self._metadata.model_version,
        )

        return self._model

    def _resolve_metadata(self) -> SalaryModelMetadata:
        """
        Best-effort metadata lookup via MlflowClient. Failures here must
        never fail model loading itself -- a model that loaded correctly
        but whose descriptive metadata couldn't be fetched is still a
        servable model, just with less observability.
        """
        model_name = _UNKNOWN_MODEL_NAME
        model_version: Optional[str] = None
        run_id: Optional[str] = None

        try:
            client = MlflowClient()
            version_info = client.get_model_version_by_alias(
                self.config.registered_model_name,
                self.config.model_alias,
            )
            model_version = str(version_info.version)
            run_id = version_info.run_id

            tags = version_info.tags or {}
            for tag_key in _MODEL_NAME_TAG_KEYS:
                if tags.get(tag_key):
                    model_name = tags[tag_key]
                    break

        except Exception as exc:
            logging.warning(
                "Could not resolve full model metadata from MLflow "
                "(model will still be served): %s",
                exc,
            )

        return SalaryModelMetadata(
            registered_model_name=self.config.registered_model_name,
            model_alias=self.config.model_alias,
            model_name=model_name,
            model_version=model_version,
            run_id=run_id,
        )

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
    # HEALTH / METADATA
    # ==========================================================

    @property
    def is_loaded(self) -> bool:

        return self._model is not None

    @property
    def metadata(self) -> Optional[SalaryModelMetadata]:
        """
        None until load() has succeeded at least once. Callers that need
        metadata (the prediction response, /readiness) should call
        load() first -- which the FastAPI lifespan already does at
        startup, and which predict() does defensively via its own
        load() call.
        """
        return self._metadata
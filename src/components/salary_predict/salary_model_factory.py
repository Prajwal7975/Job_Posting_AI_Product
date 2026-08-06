from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Tuple

from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

from src.logger import logging
from src.exception import CustomException
import sys
from src.configs.salary_predict.salary_experiment_config import SalaryExperimentConfig


class SalaryModelFactory:

    _MODEL_REGISTRY: Dict[str, Callable[..., RegressorMixin]] = {
        "dummy": DummyRegressor,
        "ridge": Ridge,
    }

    def build(self, config: SalaryExperimentConfig) -> RegressorMixin:

        try:
            self._validate_config_type(config)

            model_name = self._normalize_model_name(config.model_name)
            self._validate_supported(model_name)

            params = self._validate_and_copy_params(config.model_params)

            logging.info(
                f"Building salary estimator for "
                f"experiment_id='{config.experiment_id}'..."
            )

            logging.info(f"Model family: {model_name}")

            logging.info(f"Model parameters: {params}")

            estimator = self._construct(
                model_name=model_name,
                params=params,
            )

            logging.info(
                f"Successfully constructed unfitted "
                f"{type(estimator).__name__} estimator."
            )

            return estimator

        except Exception as e:

            logging.error(
                f"Failed to build salary estimator: {e}",
                exc_info=True,
            )

            raise CustomException(e, sys) from e

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_config_type(config: Any) -> None:
        if not isinstance(config, SalaryExperimentConfig):
            raise TypeError(
                f"SalaryModelFactory.build() expects a SalaryExperimentConfig, "
                f"got {type(config).__name__}."
            )

    @staticmethod
    def _normalize_model_name(model_name: Any) -> str:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                f"model_name must be a non-empty string, got {model_name!r}."
            )
        return model_name.strip().lower()

    def _validate_supported(self, model_name: str) -> None:
        if model_name not in self._MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported salary model family '{model_name}'. "
                f"Supported families: {sorted(self._MODEL_REGISTRY)}"
            )

    @staticmethod
    def _validate_and_copy_params(model_params: Any) -> Dict[str, Any]:
        if not isinstance(model_params, Mapping):
            raise ValueError(
                f"model_params must be a mapping (dict-like), got {type(model_params).__name__}."
            )
        non_string_keys = [k for k in model_params.keys() if not isinstance(k, str)]
        if non_string_keys:
            raise ValueError(
                f"model_params keys must all be strings, got non-string keys: {non_string_keys}"
            )
        return dict(model_params)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _construct(self, model_name: str, params: Dict[str, Any]) -> RegressorMixin:
        model_class = self._MODEL_REGISTRY[model_name]
        try:
            return model_class(**params)
        except TypeError as e:
            raise ValueError(
                f"Failed to construct model family '{model_name}' "
                f"with parameters {params}: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Read-only helpers
    # ------------------------------------------------------------------
    def supported_model_families(self) -> Tuple[str, ...]:
        """Immutable, sorted tuple of currently registered model_name values."""
        return tuple(sorted(self._MODEL_REGISTRY))

    def is_supported_model_family(self, model_name: Any) -> bool:
        try:
            normalized = self._normalize_model_name(model_name)
        except ValueError:
            return False
        return normalized in self._MODEL_REGISTRY

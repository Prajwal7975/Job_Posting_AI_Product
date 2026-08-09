from __future__ import annotations

import sys
from typing import (
    Any,
    ClassVar,
    Callable,
    Dict,
    Mapping,
    Tuple,
)

from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.exception import CustomException
from src.logger import logging


# ======================================================================
# Optional / third-party model builders
# ======================================================================


def _build_lightgbm(**params: Any) -> RegressorMixin:
    """Construct an unfitted LightGBM regressor lazily."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError as e:
        raise ImportError(
            "lightgbm is not installed. "
            "Install it with `pip install lightgbm` "
            "to use model_name='lightgbm'."
        ) from e

    return LGBMRegressor(**params)


def _build_xgboost(**params: Any) -> RegressorMixin:
    """Construct an unfitted XGBoost regressor lazily."""
    try:
        from xgboost import XGBRegressor
    except ImportError as e:
        raise ImportError(
            "xgboost is not installed. "
            "Install it with `pip install xgboost` "
            "to use model_name='xgboost'."
        ) from e

    return XGBRegressor(**params)


def _build_catboost(**params: Any) -> RegressorMixin:
    """Construct an unfitted CatBoost regressor lazily."""
    try:
        from catboost import CatBoostRegressor
    except ImportError as e:
        raise ImportError(
            "catboost is not installed. "
            "Install it with `pip install catboost` "
            "to use model_name='catboost'."
        ) from e

    return CatBoostRegressor(**params)


# ======================================================================
# Salary Model Factory
# ======================================================================


class SalaryModelFactory:
    """
    Factory for constructing salary regression estimators.

    The registry is the single source of truth for model availability.

    Configuration classes do not need to import this factory. They only
    need to expose:

        model_name
        model_params

    This keeps the configuration layer independent from ML libraries.
    """

    _MODEL_REGISTRY: ClassVar[
        Dict[str, Callable[..., RegressorMixin]]
    ] = {
        "dummy": DummyRegressor,
        "ridge": Ridge,
        "random_forest": RandomForestRegressor,
        "lightgbm": _build_lightgbm,
        "xgboost": _build_xgboost,
        "catboost": _build_catboost,
    }

    # ==================================================================
    # Public API
    # ==================================================================

    def build(self, config: Any) -> RegressorMixin:
        """
        Construct and return an unfitted salary regression estimator.

        Parameters
        ----------
        config:
            Any configuration object exposing:

                - model_name
                - model_params

        Returns
        -------
        RegressorMixin
            Unfitted estimator.

        Raises
        ------
        CustomException
            If validation or model construction fails.
        """
        try:
            self._validate_config_type(config)

            model_name = self._normalize_model_name(config.model_name)
            self._validate_supported(model_name)

            params = self._validate_and_copy_params(config.model_params)

            config_identifier = self._resolve_config_identifier(config)

            logging.info(
                "Building salary estimator for experiment_id='%s'...",
                config_identifier,
            )
            logging.info("Model family: %s", model_name)
            logging.info("Model parameters: %s", params)

            estimator = self._construct(
                model_name=model_name,
                params=params,
            )

            logging.info(
                "Successfully constructed unfitted %s estimator.",
                type(estimator).__name__,
            )

            return estimator

        except CustomException:
            # Prevents double wrapping if an internal method raises CustomException
            raise

        except Exception as e:
            logging.error(
                "Failed to build salary estimator: %s",
                e,
                exc_info=True,
            )
            raise CustomException(e, sys) from e

    # ==================================================================
    # Validation
    # ==================================================================

    @staticmethod
    def _resolve_config_identifier(config: Any) -> str:
        """
        Resolve the experiment identifier without depending on a concrete
        configuration class.

        Feature experiments use:
            experiment_id

        Model-family experiments use:
            model_experiment_id
        """
        return (
            getattr(config, "experiment_id", None)
            or getattr(config, "model_experiment_id", "?")
        )

    @staticmethod
    def _validate_config_type(config: Any) -> None:
        """
        Validate the structural configuration contract.
        """
        required_attributes = ("model_name", "model_params")

        missing = [
            attribute
            for attribute in required_attributes
            if not hasattr(config, attribute)
        ]

        if missing:
            raise TypeError(
                "SalaryModelFactory.build() requires configuration "
                f"attributes {required_attributes}; missing: {missing}. "
                f"Got {type(config).__name__}."
            )

    @staticmethod
    def _normalize_model_name(model_name: Any) -> str:
        """Normalize and validate a model family name."""
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                "model_name must be a non-empty string, "
                f"got {model_name!r}."
            )
        return model_name.strip().lower()

    @classmethod
    def _validate_supported(cls, model_name: str) -> None:
        """Validate that the requested model is registered."""
        if model_name not in cls._MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported salary model family '{model_name}'. "
                f"Supported families: "
                f"{sorted(cls._MODEL_REGISTRY)}"
            )

    @staticmethod
    def _validate_and_copy_params(model_params: Any) -> Dict[str, Any]:
        """Validate model parameters and return a mutable copy."""
        if not isinstance(model_params, Mapping):
            raise ValueError(
                "model_params must be a mapping (dict-like), "
                f"got {type(model_params).__name__}."
            )

        non_string_keys = [
            key
            for key in model_params.keys()
            if not isinstance(key, str)
        ]

        if non_string_keys:
            raise ValueError(
                "model_params keys must all be strings, "
                f"got non-string keys: {non_string_keys}"
            )

        return dict(model_params)

    # ==================================================================
    # Construction
    # ==================================================================

    @classmethod
    def _construct(
        cls,
        model_name: str,
        params: Dict[str, Any],
    ) -> RegressorMixin:
        """
        Construct an unfitted estimator from the registry.
        """
        model_builder = cls._MODEL_REGISTRY[model_name]

        try:
            estimator = model_builder(**params)

        except TypeError as e:
            raise ValueError(
                f"Failed to construct model family "
                f"'{model_name}' with parameters {params}: {e}"
            ) from e

        if not hasattr(estimator, "fit"):
            raise TypeError(
                f"Registered model family '{model_name}' "
                "did not produce an estimator with a fit() method."
            )

        if not hasattr(estimator, "predict"):
            raise TypeError(
                f"Registered model family '{model_name}' "
                "did not produce an estimator with a predict() method."
            )

        return estimator

    # ==================================================================
    # Read-only helpers
    # ==================================================================

    @classmethod
    def list_supported_models(cls) -> Tuple[str, ...]:
        """
        Return an immutable sorted tuple of registered model families.
        """
        return tuple(sorted(cls._MODEL_REGISTRY))

    @classmethod
    def is_supported_model_family(cls, model_name: Any) -> bool:
        """Return True when the model family is registered."""
        try:
            normalized = cls._normalize_model_name(model_name)
        except ValueError:
            return False

        return normalized in cls._MODEL_REGISTRY
"""
tests/smoke/test_salary_model_factory_smoke.py

Smoke test for SalaryModelFactory.

Verifies model CONSTRUCTION only:
    - the correct estimator class is built per experiment
    - config parameters actually propagate onto the estimator
    - the estimator is unfitted
    - the estimator survives sklearn.base.clone()
    - an unsupported/corrupted model_name is rejected, without ever
      calling .fit() or touching the salary dataset.

Run with either:
    python -m tests.smoke.test_salary_model_factory_smoke
    pytest tests/smoke/test_salary_model_factory_smoke.py
"""

from __future__ import annotations

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

from src.logger import logging
from src.exception import CustomException
from src.configs.salary_predict.salary_experiment_config import get_experiment_config
from src.components.salary_predict.salary_model_factory import SalaryModelFactory

_EXPECTED_ESTIMATOR_TYPE = {
    "E0": DummyRegressor,
    "E1": Ridge,
    "E2": Ridge,
    "E3A": Ridge,
    "E3B": Ridge,
}


def _assert_unfitted(model) -> None:
    # Neither DummyRegressor nor Ridge sets these attributes until fit() is
    # called, so their absence is a reliable "never fitted" signal.
    fitted_markers = ("coef_", "constant_")
    already_fitted = [m for m in fitted_markers if hasattr(model, m)]
    assert (
        not already_fitted
    ), f"Estimator appears fitted, has attributes: {already_fitted}"


def test_build_all_initial_experiments() -> None:
    factory = SalaryModelFactory()

    for experiment_id, expected_type in _EXPECTED_ESTIMATOR_TYPE.items():
        config = get_experiment_config(experiment_id)
        model = factory.build(config)

        assert model is not None, f"{experiment_id}: factory returned None"
        assert isinstance(
            model, expected_type
        ), f"{experiment_id}: expected {expected_type.__name__}, got {type(model).__name__}"
        _assert_unfitted(model)

        actual_params = model.get_params(deep=False)

        for param_name, expected_value in config.model_params.items():

            assert param_name in actual_params, (
                f"{experiment_id}: parameter '{param_name}' "
                f"not found on {type(model).__name__}"
            )

            actual_value = actual_params[param_name]

            assert actual_value == expected_value, (
                f"{experiment_id}: expected "
                f"{param_name}={expected_value!r}, "
                f"got {actual_value!r}"
            )

        # Must remain a well-behaved sklearn estimator.

        cloned = clone(model)

        assert isinstance(
            cloned, expected_type
        ), f"{experiment_id}: clone() changed estimator type"

        assert cloned.get_params(deep=False) == model.get_params(
            deep=False
        ), f"{experiment_id}: clone() did not preserve estimator parameters"

        _assert_unfitted(cloned)

        logging.info(
            f"[SMOKE] {experiment_id}: built {type(model).__name__} "
            f"with params={config.model_params} - OK"
        )


def test_build_rejects_wrong_config_type() -> None:
    factory = SalaryModelFactory()

    try:
        factory.build("not a config")  # type: ignore[arg-type]

    except CustomException as e:
        logging.info(
            f"[SMOKE] wrong config type correctly rejected "
            f"through CustomException: {e}"
        )

    else:
        raise AssertionError(
            "Expected CustomException for a non-SalaryExperimentConfig argument."
        )


def test_build_rejects_unsupported_model_name() -> None:
    factory = SalaryModelFactory()
    config = get_experiment_config("E1")
    object.__setattr__(config, "model_name", "xgboost")

    try:
        factory.build(config)

    except CustomException as e:
        logging.info(
            f"[SMOKE] unsupported model_name correctly rejected "
            f"through CustomException: {e}"
        )

    else:
        raise AssertionError("Expected CustomException for an unsupported model_name.")


def test_supported_model_family_helpers() -> None:
    factory = SalaryModelFactory()

    assert factory.supported_model_families() == (
        "dummy",
        "ridge",
    )

    assert factory.is_supported_model_family("dummy") is True
    assert factory.is_supported_model_family("ridge") is True

    # Normalization should work
    assert factory.is_supported_model_family("RIDGE") is True
    assert factory.is_supported_model_family(" Ridge ") is True

    # Unsupported/invalid values
    assert factory.is_supported_model_family("xgboost") is False
    assert factory.is_supported_model_family("") is False
    assert factory.is_supported_model_family(None) is False

    logging.info("[SMOKE] supported-model helper methods - OK")


def run_all() -> None:
    logging.info("=" * 70)
    logging.info("SALARY MODEL FACTORY SMOKE TEST STARTED")
    logging.info("=" * 70)

    test_build_all_initial_experiments()
    test_supported_model_family_helpers()
    test_build_rejects_wrong_config_type()
    test_build_rejects_unsupported_model_name()

    logging.info("=" * 70)
    logging.info("SALARY MODEL FACTORY SMOKE TEST PASSED")
    logging.info("=" * 70)
    print("SALARY MODEL FACTORY SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    run_all()

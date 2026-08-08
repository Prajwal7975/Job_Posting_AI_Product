"""
tests/smoke/test_salary_training_runner_smoke.py

Smoke test for SalaryTrainingRunner.

Purpose
-------
Verify that the real salary training stack works end-to-end for every
configured initial experiment:

    ExperimentConfig
          ↓
    PreprocessorBuilder
          ↓
    ModelFactory
          ↓
    SalaryTrainingRunner
          ↓
    fit TRAIN
          ↓
    predict VALIDATION
          ↓
    validation metrics

Important
---------
- Uses train.parquet and validation.parquet.
- NEVER uses test.parquet.
- Does NOT perform model selection.
- Does NOT use MLflow.
- Does NOT save joblib artifacts.
- Does NOT perform hyperparameter tuning.

Run:
    python -m tests.smoke.test_salary_training_runner_smoke

or:
    pytest tests/smoke/test_salary_training_runner_smoke.py -s
"""

from __future__ import annotations

import math
from typing import Dict

from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from src.logger import logging

from src.configs.salary_predict.salary_experiment_config import (
    get_experiment_config,
)

from components.salary_predict.salary_feature_experiment_runner import (
    SalaryTrainingRunner,
    SalaryTrainingResult,
)

# ======================================================================
# EXPERIMENTS
# ======================================================================

EXPERIMENT_IDS = (
    "E0",
    "E1",
    "E2",
    "E3A",
    "E3B",
)


# ======================================================================
# HELPERS
# ======================================================================


def _assert_finite_number(
    value: float,
    label: str,
) -> None:
    """
    Ensure a returned numeric value is finite.
    """

    assert isinstance(
        value,
        (int, float),
    ), (
        f"{label}: expected numeric value, " f"got {type(value).__name__}"
    )

    assert math.isfinite(float(value)), f"{label}: expected finite value, got {value}"


def _validate_metrics(
    metrics: Dict[str, float],
    experiment_id: str,
    metric_space: str,
) -> None:
    """
    Validate metric dictionary returned by SalaryTrainingRunner.
    """

    required_metrics = {
        "mae",
        "rmse",
        "r2",
    }

    missing = required_metrics - set(metrics)

    assert not missing, (
        f"{experiment_id}: " f"{metric_space} metrics missing " f"{sorted(missing)}"
    )

    for metric_name in required_metrics:

        _assert_finite_number(
            metrics[metric_name],
            f"{experiment_id}.{metric_space}.{metric_name}",
        )

    # MAE and RMSE must never be negative.

    assert metrics["mae"] >= 0, (
        f"{experiment_id}: " f"{metric_space} MAE cannot be negative."
    )

    assert metrics["rmse"] >= 0, (
        f"{experiment_id}: " f"{metric_space} RMSE cannot be negative."
    )


# ======================================================================
# RESULT VALIDATION
# ======================================================================


def _validate_training_result(
    experiment_id: str,
    result: SalaryTrainingResult,
) -> None:

    config = get_experiment_config(experiment_id)

    # ------------------------------------------------------------------
    # Result contract
    # ------------------------------------------------------------------

    assert isinstance(
        result,
        SalaryTrainingResult,
    ), (
        f"{experiment_id}: expected SalaryTrainingResult, "
        f"got {type(result).__name__}"
    )

    # ------------------------------------------------------------------
    # Experiment identity
    # ------------------------------------------------------------------

    assert result.experiment_id == config.experiment_id

    assert result.experiment_name == config.experiment_name

    assert result.model_name == config.model_name

    assert result.config_signature == config.config_signature

    assert (
        result.config_signature.strip()
    ), f"{experiment_id}: config signature is empty."

    # ------------------------------------------------------------------
    # Dataset counts
    # ------------------------------------------------------------------

    assert result.train_row_count > 0, f"{experiment_id}: train dataset is empty."

    assert (
        result.validation_row_count > 0
    ), f"{experiment_id}: validation dataset is empty."

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    _assert_finite_number(
        result.training_seconds,
        f"{experiment_id}.training_seconds",
    )

    _assert_finite_number(
        result.validation_prediction_seconds,
        f"{experiment_id}.validation_prediction_seconds",
    )

    assert result.training_seconds >= 0

    assert result.validation_prediction_seconds >= 0

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    _validate_metrics(
        result.log_metrics,
        experiment_id,
        "log",
    )

    _validate_metrics(
        result.annual_metrics,
        experiment_id,
        "annual",
    )

    # Annual metric dictionary should additionally contain Median APE.

    assert "median_ape" in result.annual_metrics, (
        f"{experiment_id}: " "annual metrics missing median_ape."
    )

    median_ape = result.annual_metrics["median_ape"]

    _assert_finite_number(
        median_ape,
        f"{experiment_id}.annual.median_ape",
    )

    assert median_ape >= 0

    # ------------------------------------------------------------------
    # Fitted object
    # ------------------------------------------------------------------

    assert (
        result.fitted_workflow is not None
    ), f"{experiment_id}: fitted_pipeline is None."

    # ==================================================================
    # E0 — DUMMY BASELINE
    # ==================================================================

    if experiment_id == "E0":

        assert isinstance(
            result.fitted_workflow,
            DummyRegressor,
        ), "E0 must return a fitted DummyRegressor."

        assert result.raw_feature_columns == (), "E0 should not use predictor features."

        assert result.raw_feature_count == 0

        assert result.transformed_feature_count is None

        # DummyRegressor gets constant_ after fitting.

        assert hasattr(
            result.fitted_workflow,
            "constant_",
        ), "E0 DummyRegressor does not appear fitted."

        return

    # ==================================================================
    # FEATURE-BASED EXPERIMENTS
    # ==================================================================

    assert isinstance(
        result.fitted_workflow,
        Pipeline,
    ), (
        f"{experiment_id}: expected sklearn Pipeline, "
        f"got {type(result.fitted_workflow).__name__}"
    )

    # ------------------------------------------------------------------
    # Pipeline structure
    # ------------------------------------------------------------------

    assert "preprocessor" in result.fitted_workflow.named_steps, (
        f"{experiment_id}: " "pipeline missing preprocessor."
    )

    assert "model" in result.fitted_workflow.named_steps, (
        f"{experiment_id}: " "pipeline missing model."
    )

    # ------------------------------------------------------------------
    # Raw feature contract
    # ------------------------------------------------------------------

    assert result.raw_feature_count > 0

    assert result.raw_feature_count == len(result.raw_feature_columns)

    assert result.raw_feature_columns == tuple(config.active_predictor_features), (
        f"{experiment_id}: runner did not use "
        "the exact feature contract from config."
    )

    # ------------------------------------------------------------------
    # Transformed features
    # ------------------------------------------------------------------

    assert result.transformed_feature_count is not None, (
        f"{experiment_id}: " "transformed_feature_count is None."
    )

    assert result.transformed_feature_count > 0

    fitted_preprocessor = result.fitted_workflow.named_steps["preprocessor"]

    feature_names = fitted_preprocessor.get_feature_names_out()

    assert len(feature_names) == result.transformed_feature_count, (
        f"{experiment_id}: transformed feature "
        "count does not match fitted preprocessor."
    )

    # ------------------------------------------------------------------
    # Model must actually be fitted
    # ------------------------------------------------------------------

    fitted_model = result.fitted_workflow.named_steps["model"]

    assert hasattr(
        fitted_model,
        "n_features_in_",
    ), f"{experiment_id}: model does not appear fitted."


# ======================================================================
# RUN ONE EXPERIMENT
# ======================================================================


def _run_experiment(
    runner: SalaryTrainingRunner,
    experiment_id: str,
) -> SalaryTrainingResult:

    logging.info("")
    logging.info("=" * 70)
    logging.info(
        "SMOKE TESTING EXPERIMENT %s",
        experiment_id,
    )
    logging.info("=" * 70)

    config = get_experiment_config(experiment_id)

    result = runner.run(config)

    _validate_training_result(
        experiment_id,
        result,
    )

    logging.info(
        "[SMOKE] %s PASS",
        experiment_id,
    )

    logging.info(
        "[SMOKE] Raw features         : %s",
        result.raw_feature_count,
    )

    logging.info(
        "[SMOKE] Transformed features : %s",
        result.transformed_feature_count,
    )

    logging.info(
        "[SMOKE] Training time        : %.4f sec",
        result.training_seconds,
    )

    logging.info(
        "[SMOKE] Prediction time      : %.4f sec",
        result.validation_prediction_seconds,
    )

    logging.info(
        "[SMOKE] Log metrics          : %s",
        result.log_metrics,
    )

    logging.info(
        "[SMOKE] Annual metrics       : %s",
        result.annual_metrics,
    )

    return result


# ======================================================================
# PYTEST ENTRY POINTS
# ======================================================================


def test_salary_training_e0() -> None:

    runner = SalaryTrainingRunner()

    _run_experiment(
        runner,
        "E0",
    )


def test_salary_training_e1() -> None:

    runner = SalaryTrainingRunner()

    _run_experiment(
        runner,
        "E1",
    )


def test_salary_training_e2() -> None:

    runner = SalaryTrainingRunner()

    _run_experiment(
        runner,
        "E2",
    )


def test_salary_training_e3a() -> None:

    runner = SalaryTrainingRunner()

    _run_experiment(
        runner,
        "E3A",
    )


def test_salary_training_e3b() -> None:

    runner = SalaryTrainingRunner()

    _run_experiment(
        runner,
        "E3B",
    )


# ======================================================================
# MANUAL SMOKE TEST
# ======================================================================


def run_all() -> None:

    logging.info("")
    logging.info("#" * 70)
    logging.info("SALARY TRAINING RUNNER SMOKE TEST STARTED")
    logging.info("#" * 70)

    runner = SalaryTrainingRunner()

    results: Dict[
        str,
        SalaryTrainingResult,
    ] = {}

    for experiment_id in EXPERIMENT_IDS:

        result = _run_experiment(
            runner,
            experiment_id,
        )

        results[experiment_id] = result

    # ==================================================================
    # SUMMARY
    # ==================================================================

    logging.info("")
    logging.info("=" * 70)
    logging.info("SALARY TRAINING SMOKE TEST SUMMARY")
    logging.info("=" * 70)

    logging.info(
        "%-6s %-14s %-14s %-10s %-12s",
        "EXP",
        "ANNUAL MAE",
        "ANNUAL RMSE",
        "R2",
        "TRAIN SEC",
    )

    logging.info("-" * 70)

    for experiment_id in EXPERIMENT_IDS:

        result = results[experiment_id]

        logging.info(
            "%-6s %-14.2f %-14.2f %-10.4f %-12.4f",
            experiment_id,
            result.annual_metrics["mae"],
            result.annual_metrics["rmse"],
            result.annual_metrics["r2"],
            result.training_seconds,
        )

    logging.info("=" * 70)

    logging.info("ALL SALARY TRAINING RUNNER " "SMOKE TESTS PASSED")

    logging.info("=" * 70)

    print("\nSALARY TRAINING RUNNER SMOKE TEST: " "ALL CHECKS PASSED")


if __name__ == "__main__":
    run_all()

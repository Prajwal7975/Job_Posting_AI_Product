"""
tests/smoke/test_salary_mlflow_tracker_smoke.py

Smoke test for SalaryMLflowTracker.

Uses a temporary, isolated sqlite-backed MLflow tracking store under
pytest's tmp_path fixture — never the developer's real mlflow.db / mlruns/
/ http://127.0.0.1:5000. Suitable for CI: no manually running MLflow
server is required.

Asserts against ACTUAL persisted state via MlflowClient, not merely that
calls returned without raising.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from src.configs.salary_predict.salary_ML_flow_config import SalaryMLflowConfig
from src.configs.salary_predict.salary_experiment_config import (
    SalaryExperimentConfig,
    get_experiment_config,
)
from src.components.salary_predict.salary_mlflow_tracker import SalaryMLflowTracker
from src.components.salary_predict.salary_training_runner import SalaryTrainingResult


def _make_result(
    exp_config: SalaryExperimentConfig, 
    median_ape: float = 1234.5
) -> SalaryTrainingResult:
    return SalaryTrainingResult(
        experiment_id=exp_config.experiment_id,
        experiment_name=exp_config.experiment_name,
        model_name=exp_config.model_name,
        config_signature=exp_config.config_signature,
        train_row_count=1000,
        validation_row_count=200,
        raw_feature_columns=exp_config.active_predictor_features,
        raw_feature_count=len(exp_config.active_predictor_features),
        transformed_feature_count=42,
        training_seconds=1.234,
        validation_prediction_seconds=0.05,
        log_metrics={"mae": 0.31, "rmse": 0.42, "r2": 0.55},
        annual_metrics={"mae": 15000.0, "rmse": 21000.0, "r2": 0.55, "median_ape": median_ape},
        fitted_workflow=None,
    )


@pytest.fixture
def tracking_uri(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'mlflow_test.db'}"


def test_tracker_records_full_run(tracking_uri: str) -> None:
    config = SalaryMLflowConfig(tracking_uri=tracking_uri, experiment_name="salary_prediction_test")
    tracker = SalaryMLflowTracker(config)
    exp_config = get_experiment_config("E1")
    result = _make_result(exp_config)

    run_info = tracker.track_training_result(exp_config, result)

    # run terminates cleanly, status FINISHED, run_id available
    assert run_info.status == "FINISHED"
    assert run_info.run_id is not None

    client = MlflowClient(tracking_uri=tracking_uri)

    # experiment created/resolved, run created
    experiment = client.get_experiment_by_name("salary_prediction_test")
    assert experiment is not None
    run = client.get_run(run_info.run_id)
    assert run.info.experiment_id == experiment.experiment_id
    assert run.info.status == "FINISHED"

    # timestamps captured
    assert run.info.start_time is not None
    assert run.info.end_time is not None

    # expected run name
    assert run.data.tags.get("mlflow.runName") == "salary-E1"

    # baseline + experiment-specific tags stored
    assert run.data.tags.get("project") == config.project_name
    assert run.data.tags.get("pipeline") == config.pipeline_name
    assert run.data.tags.get("experiment_id") == "E1"
    assert run.data.tags.get("model_family") == "ridge"
    assert run.data.tags.get("config_signature") == exp_config.config_signature

    # model parameters stored
    assert run.data.params.get("model.name") == "ridge"
    assert run.data.params.get("model.alpha") == str(exp_config.model_params["alpha"])

    # validation metrics + timing metrics stored
    assert run.data.metrics.get("validation.log_mae") == pytest.approx(0.31)
    assert run.data.metrics.get("validation.annual_mae") == pytest.approx(15000.0)
    assert run.data.metrics.get("timing.training_seconds") == pytest.approx(1.234)
    assert run.data.metrics.get("timing.prediction_seconds") == pytest.approx(0.05)


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_tracker_skips_non_finite_metrics(tracking_uri: str, bad_val: float) -> None:
    config = SalaryMLflowConfig(tracking_uri=tracking_uri, experiment_name="salary_prediction_test_nan")
    tracker = SalaryMLflowTracker(config)
    exp_config = get_experiment_config("E1")
    result = _make_result(exp_config, median_ape=bad_val)

    run_info = tracker.track_training_result(exp_config, result)

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_info.run_id)

    # Non-finite metric rejected (skipped), not logged as a value at all
    assert "validation.median_ape" not in run.data.metrics
    # everything else still logged despite the one bad value
    assert "validation.annual_mae" in run.data.metrics


def test_tracking_disabled_creates_no_run(tracking_uri: str) -> None:
    config = SalaryMLflowConfig(tracking_uri=tracking_uri, tracking_enabled=False)
    tracker = SalaryMLflowTracker(config)
    exp_config = get_experiment_config("E0")
    result = _make_result(exp_config)

    run_info = tracker.track_training_result(exp_config, result)

    # tracking_enabled=False causes no MLflow run to be created
    assert run_info.status == "DISABLED"
    assert run_info.run_id is None

    client = MlflowClient(tracking_uri=tracking_uri)
    experiments = client.search_experiments()
    names = [e.name for e in experiments]
    assert config.experiment_name not in names


def test_repeated_calls_reuse_same_experiment(tracking_uri: str) -> None:
    config = SalaryMLflowConfig(tracking_uri=tracking_uri, experiment_name="salary_prediction_reuse")
    tracker = SalaryMLflowTracker(config)
    exp_config = get_experiment_config("E1")
    result = _make_result(exp_config)

    run_info_a = tracker.track_training_result(exp_config, result)
    run_info_b = tracker.track_training_result(exp_config, result)

    assert run_info_a.mlflow_experiment_id == run_info_b.mlflow_experiment_id
    assert run_info_a.run_id != run_info_b.run_id

    client = MlflowClient(tracking_uri=tracking_uri)
    experiments = [e for e in client.search_experiments() if e.name == "salary_prediction_reuse"]
    # Idempotent: exactly one experiment, not one per call.
    assert len(experiments) == 1


def test_no_model_registered_by_default(tracking_uri: str) -> None:
    config = SalaryMLflowConfig(tracking_uri=tracking_uri, experiment_name="salary_prediction_no_registry")
    tracker = SalaryMLflowTracker(config)
    exp_config = get_experiment_config("E1")
    result = _make_result(exp_config)

    tracker.track_training_result(exp_config, result)  # log_model=False default

    client = MlflowClient(tracking_uri=tracking_uri)
    registered_models = client.search_registered_models()
    assert len(registered_models) == 0
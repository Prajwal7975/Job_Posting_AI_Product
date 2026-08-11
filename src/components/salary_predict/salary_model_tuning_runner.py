"""
src/components/salary_predict/salary_model_tuning_runner.py

Hyperparameter Tuning Runner.

Tunes ONLY the model family that won the model-family comparison stage.
Never selects a different model family, never touches X_test/y_test (only
train/validation), and never mutates or reuses a previously-fitted
estimator — every trial constructs and fits a completely fresh one via
SalaryModelFactory.

DATASET CONTRACT EXPECTATION:
- Both Model Family Comparison and Hyperparameter Tuning MUST evaluate on
  X_validation / y_validation.
- Baseline winner_score from model_family_summary MUST be derived from
  X_validation / y_validation so that tuning score comparison is mathematically
  apples-to-apples.
- X_test / y_test is strictly held out for final model evaluation downstream.

Pipeline Lifecycle:
    SalaryModelFamilyExperimentSummary (winner_score on X_val)
                    |
                    v
    SalaryModelTuningConfig (search space for that winner)
                    |
                    v
    generate_param_grid() -> N trials
                    |
          for each trial:
                    v
    [MLflow start_run] -> SalaryModelFactory.build(trial_params)
                    |
                    v
    fit(X_train, y_train) -> predict(X_val) -> MAE/RMSE/R2 -> log -> [MLflow end_run]
                    |
                    v
    rank trials -> best trial -> compare vs baseline (X_val) -> SalaryModelTuningSummary

Responsibility boundary: this file does NOT build the model-family
comparison, does NOT train the final production model (it leaves artifact
persistence to the downstream final trainer), and does NOT implement a second
model registry — SalaryModelFactory remains the single source of truth for
constructing estimators.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.logger import logging
from src.exception import CustomException
from src.components.salary_predict.salary_model_factory import SalaryModelFactory
from src.configs.salary_predict.salary_model_tuning_config import (
    SalaryModelTuningConfig,
    get_tuning_config_for_winner,
)
from src.entity.salary_model_tuning_entity import (
    SalaryModelTuningTrialResult,
    SalaryModelTuningSummary,
)


@dataclass(frozen=True)
class _TrialConfigView:
    """
    Minimal adapter satisfying SalaryModelFactory.build()'s contract
    (model_name + model_params, optionally model_experiment_id for log
    identification) for one trial's parameter combination. Not a second model
    registry — just a throwaway view over a dict of params so the factory can
    be called uniformly for every trial.
    """

    model_experiment_id: str
    model_name: str
    model_params: Dict[str, Any]


def _config_signature(model_name: str, params: Dict[str, Any]) -> str:
    """
    Computes a deterministic SHA-256 signature for a specific trial's model configuration
    (model family + exact hyperparameter combination).
    """
    payload = {"model_name": model_name, "model_params": dict(params)}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SalaryModelTuningRunner:
    """
    Runs a full hyperparameter search over the winning model family from
    a SalaryModelFamilyExperimentSummary, using only train/validation
    data. Returns a SalaryModelTuningSummary that explicitly records
    whether tuning actually beat the baseline — never silently assumes
    tuned is better just because tuning ran.
    """

    def __init__(
        self,
        model_factory: Optional[SalaryModelFactory] = None,
        mlflow_tracker: Any = None,
        base_artifacts_dir: str = "artifacts/salary_model_tuning",
    ) -> None:
        self.model_factory = model_factory or SalaryModelFactory()
        self.mlflow_tracker = mlflow_tracker
        self.base_artifacts_dir = Path(base_artifacts_dir)

    # ==================================================================
    # PUBLIC API
    # ==================================================================
    def run_tuning(
        self,
        model_family_summary: Any,
        X_train: Any,
        y_train: Any,
        X_validation: Any,
        y_validation: Any,
        tuning_config: Optional[SalaryModelTuningConfig] = None,
    ) -> SalaryModelTuningSummary:
        try:
            logging.info("=" * 70)
            logging.info("HYPERPARAMETER TUNING STARTED")
            logging.info("=" * 70)

            start_time = perf_counter()

            self._validate_winner(model_family_summary)
            self._validate_data(X_train, y_train, X_validation, y_validation)

            baseline_result = self._get_winner_result(model_family_summary)
            feature_experiment_id = model_family_summary.feature_experiment_id
            feature_config_signature = getattr(
                model_family_summary, "feature_config_signature", None
            )

            if tuning_config is None:
                tuning_config = get_tuning_config_for_winner(
                    model_family_summary,
                    ranking_metric=getattr(
                        model_family_summary, "ranking_metric", "RMSE"
                    ),
                )

            self._validate_tuning_config(tuning_config, model_family_summary)

            logging.info(f"Winner model          : {tuning_config.model_name}")
            logging.info(f"Feature configuration : {feature_experiment_id}")
            logging.info(
                f"Baseline {tuning_config.ranking_metric:<5}       : {model_family_summary.winner_score}"
            )
            logging.info(f"Search strategy       : {tuning_config.search_strategy}")

            param_grid = tuning_config.generate_param_grid()
            logging.info(f"Total trials          : {len(param_grid)}")

            all_trials = self._run_all_trials(
                tuning_config=tuning_config,
                param_grid=param_grid,
                X_train=X_train,
                y_train=y_train,
                X_validation=X_validation,
                y_validation=y_validation,
                feature_experiment_id=feature_experiment_id,
                feature_config_signature=feature_config_signature,
            )

            successful_trials = tuple(t for t in all_trials if t.success)
            failed_trials = tuple(t for t in all_trials if not t.success)

            if not successful_trials:
                raise RuntimeError(
                    f"All {len(all_trials)} tuning trials failed. "
                    f"Failures: {[(t.trial_id, t.error) for t in failed_trials]}"
                )
            if failed_trials:
                logging.warning(
                    f"{len(failed_trials)}/{len(all_trials)} trial(s) failed: "
                    f"{[(t.trial_id, t.error) for t in failed_trials]}"
                )

            ranked_trials = self._rank_trials(
                successful_trials,
                tuning_config.ranking_metric,
                tuning_config.ranking_direction,
            )
            best_trial = ranked_trials[0]
            best_score = best_trial.metrics[tuning_config.ranking_metric]

            baseline_score = float(model_family_summary.winner_score)
            baseline_params = dict(
                getattr(baseline_result, "model_params", None)
                or getattr(baseline_result, "params", {})
                or {}
            )
            if not baseline_params:
                logging.warning(
                    "Winner baseline parameters are empty. "
                    "Verify that the model-family runner stored model_params."
                )

            baseline_config_signature = (
                getattr(baseline_result, "config_signature", None)
                or model_family_summary.winner_config_signature
            )
            if not baseline_config_signature:
                raise ValueError(
                    "Unable to resolve baseline_config_signature from "
                    "the model-family winner."
                )

            min_threshold = getattr(tuning_config, "min_improvement_threshold", 0.0)

            improvement, improvement_percentage, improved = self._compare_to_baseline(
                baseline_score=baseline_score,
                best_score=best_score,
                direction=tuning_config.ranking_direction,
                min_improvement_threshold=min_threshold,
            )

            execution_seconds = round(perf_counter() - start_time, 4)

            summary = SalaryModelTuningSummary(
                feature_experiment_id=feature_experiment_id,
                feature_config_signature=feature_config_signature,
                winner_model_experiment_id=model_family_summary.winner_experiment_id,
                model_name=tuning_config.model_name,
                model_class_name=getattr(
                    model_family_summary, "winner_model_class", None
                ),
                baseline_score=baseline_score,
                best_score=best_score,
                ranking_metric=tuning_config.ranking_metric,
                ranking_direction=tuning_config.ranking_direction,
                baseline_config_signature=baseline_config_signature,
                best_config_signature=best_trial.config_signature,
                baseline_params=baseline_params,
                best_params=dict(best_trial.params),
                tuning_improved_baseline=improved,
                improvement=improvement,
                improvement_percentage=improvement_percentage,
                total_trial_count=len(all_trials),
                successful_trial_count=len(successful_trials),
                failed_trial_count=len(failed_trials),
                best_trial_id=best_trial.trial_id,
                successful_trials=successful_trials,
                failed_trials=failed_trials,
                ranked_trials=ranked_trials,
                execution_seconds=execution_seconds,
                best_model_artifact_path=None,  # Final model training occurs downstream
            )

            try:
                self._generate_reports(summary=summary, tuning_config=tuning_config)
            except Exception as e:
                logging.error("Failed to generate tuning reports.", exc_info=True)
                raise CustomException(
                    RuntimeError(f"Failed to generate tuning reports: {e}"), sys
                ) from e

            try:
                summary.mlflow_summary_run_id = self._track_tuning_summary(
                    summary, tuning_config
                )
            except Exception as e:
                logging.warning(
                    f"Tuning summary MLflow tracking failed: {e}", exc_info=True
                )

            self._log_console_summary(summary)

            logging.info("=" * 70)
            logging.info("HYPERPARAMETER TUNING COMPLETED")
            logging.info("=" * 70)

            return summary

        except CustomException:
            raise
        except Exception as e:
            logging.error("Hyperparameter tuning failed: %s", e, exc_info=True)
            raise CustomException(e, sys) from e

    # ==================================================================
    # HELPER & VALIDATION METHODS
    # ==================================================================
    @staticmethod
    def _get_winner_result(model_family_summary: Any) -> Any:
        winner_id = getattr(model_family_summary, "winner_experiment_id", None)
        if not winner_id:
            raise ValueError(
                "model_family_summary does not contain a valid 'winner_experiment_id'."
            )

        candidates = []
        if (
            hasattr(model_family_summary, "ranked_results")
            and model_family_summary.ranked_results
        ):
            candidates.extend(model_family_summary.ranked_results)
        if (
            hasattr(model_family_summary, "successful_results")
            and model_family_summary.successful_results
        ):
            candidates.extend(model_family_summary.successful_results)

        for result in candidates:
            exp_id = getattr(result, "experiment_id", None) or getattr(
                result, "model_experiment_id", None
            )
            if exp_id == winner_id:
                return result

        raise ValueError(
            f"Winner experiment ID '{winner_id}' was not found in "
            "model_family_summary's ranked_results or successful_results."
        )

    @staticmethod
    def _validate_winner(model_family_summary: Any) -> None:
        required_attrs = (
            "winner_model_name",
            "winner_experiment_id",
            "winner_config_signature",
            "winner_score",
            "ranking_metric",
            "ranking_direction",
            "feature_experiment_id",
            "ranked_results",
        )
        missing = [a for a in required_attrs if not hasattr(model_family_summary, a)]
        if missing:
            raise ValueError(
                f"model_family_summary is missing required attributes: {missing}"
            )
        if not model_family_summary.ranked_results:
            raise ValueError(
                "model_family_summary.ranked_results is empty — no winner to tune."
            )
        if not model_family_summary.winner_model_name:
            raise ValueError("model_family_summary.winner_model_name is empty.")

        winner_score = getattr(model_family_summary, "winner_score", None)
        if winner_score is None:
            raise ValueError("model_family_summary.winner_score must not be None.")
        try:
            winner_score_val = float(winner_score)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"winner_score must be numeric, got {winner_score!r}."
            ) from e
        if not math.isfinite(winner_score_val):
            raise ValueError(f"winner_score must be finite, got {winner_score!r}.")

    @staticmethod
    def _validate_data(
        X_train: Any, y_train: Any, X_validation: Any, y_validation: Any
    ) -> None:
        # -------------------------------------------------------------
        # Data Validation
        # Supports pandas, NumPy and SciPy sparse matrices.
        # -------------------------------------------------------------

        if X_train is None or y_train is None:
            raise ValueError("Training data cannot be None.")

        if X_validation is None or y_validation is None:
            raise ValueError("Validation data cannot be None.")

        def _n_rows(data: Any) -> int:

            if not hasattr(data, "shape"):
                raise TypeError(f"Unsupported data object: {type(data).__name__}")

            if len(data.shape) == 0:
                raise ValueError("Input data must have at least one dimension.")

            return int(data.shape[0])

        train_rows = _n_rows(X_train)
        train_target_rows = _n_rows(y_train)

        validation_rows = _n_rows(X_validation)
        validation_target_rows = _n_rows(y_validation)

        if train_rows != train_target_rows:
            raise ValueError(
                "X_train/y_train length mismatch: "
                f"{train_rows} != {train_target_rows}"
            )

        if validation_rows != validation_target_rows:
            raise ValueError(
                "X_validation/y_validation length mismatch: "
                f"{validation_rows} != {validation_target_rows}"
            )

        if train_rows == 0:
            raise ValueError("X_train must not be empty.")

        if validation_rows == 0:
            raise ValueError("X_validation must not be empty.")

    @staticmethod
    def _validate_tuning_config(
        tuning_config: SalaryModelTuningConfig, model_family_summary: Any
    ) -> None:
        if tuning_config.model_name != model_family_summary.winner_model_name:
            raise ValueError(
                f"tuning_config.model_name ('{tuning_config.model_name}') does not match "
                f"the model-family winner ('{model_family_summary.winner_model_name}'). "
                "The tuning stage must never select a different model family than the "
                "one the comparison stage already chose."
            )

        summary_metric = getattr(model_family_summary, "ranking_metric", None)
        if summary_metric and tuning_config.ranking_metric != summary_metric:
            raise ValueError(
                f"tuning_config.ranking_metric ('{tuning_config.ranking_metric}') does not match "
                f"model_family_summary.ranking_metric ('{summary_metric}'). "
                "The tuning stage must use the exact same evaluation metric as model family comparison."
            )

        summary_direction = getattr(model_family_summary, "ranking_direction", None)
        if summary_direction and tuning_config.ranking_direction != summary_direction:
            raise ValueError(
                f"tuning_config.ranking_direction ('{tuning_config.ranking_direction}') does not match "
                f"model_family_summary.ranking_direction ('{summary_direction}'). "
                "The tuning stage must use the exact same ranking direction as model family comparison."
            )

        allowed_metrics = {"MAE", "RMSE", "R2"}
        if tuning_config.ranking_metric not in allowed_metrics:
            raise ValueError(
                f"Unsupported ranking_metric: '{tuning_config.ranking_metric}'. "
                f"Must be one of {sorted(allowed_metrics)}."
            )

        allowed_directions = {"minimize", "maximize"}
        if tuning_config.ranking_direction not in allowed_directions:
            raise ValueError(
                f"Unsupported ranking_direction: '{tuning_config.ranking_direction}'. "
                f"Must be one of {sorted(allowed_directions)}."
            )

    # ==================================================================
    # TRIAL EXECUTION WITH DECOUPLED MLFLOW LIFECYCLE
    # ==================================================================
    def _run_all_trials(
        self,
        tuning_config: SalaryModelTuningConfig,
        param_grid: Tuple[Dict[str, Any], ...],
        X_train: Any,
        y_train: Any,
        X_validation: Any,
        y_validation: Any,
        feature_experiment_id: str,
        feature_config_signature: Optional[str],
    ) -> Tuple[SalaryModelTuningTrialResult, ...]:
        trials = []
        total = len(param_grid)
        id_width = max(3, len(str(total)))

        for index, params in enumerate(param_grid, start=1):
            trial_id = f"T{index:0{id_width}d}"
            logging.info("")
            logging.info(f"Trial {index}/{total}")
            logging.info(f"Parameters: {params}")

            trial = self._run_one_trial_with_lifecycle(
                trial_id=trial_id,
                tuning_config=tuning_config,
                params=params,
                X_train=X_train,
                y_train=y_train,
                X_validation=X_validation,
                y_validation=y_validation,
                feature_experiment_id=feature_experiment_id,
                feature_config_signature=feature_config_signature,
            )
            trials.append(trial)

            if trial.success:
                logging.info(
                    f"{tuning_config.ranking_metric}: {trial.metrics.get(tuning_config.ranking_metric)}"
                )
                logging.info("Status: SUCCESS")
            else:
                logging.warning(f"Status: FAILED ({trial.error})")

        return tuple(trials)

    def _run_one_trial_with_lifecycle(
        self,
        trial_id: str,
        tuning_config: SalaryModelTuningConfig,
        params: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_validation: Any,
        y_validation: Any,
        feature_experiment_id: str,
        feature_config_signature: Optional[str],
    ) -> SalaryModelTuningTrialResult:
        """
        Executes model creation, fitting, prediction, and evaluation exactly once.
        MLflow logging wraps execution without overriding model trial status on failure.
        """
        model_name = tuning_config.model_name
        config_signature = _config_signature(model_name, params)

        trial = SalaryModelTuningTrialResult(
            trial_id=trial_id,
            model_name=model_name,
            feature_experiment_id=feature_experiment_id,
            feature_config_signature=feature_config_signature,
            config_signature=config_signature,
            success=False,
            params=dict(params),
        )

        def _execute_model_trial() -> None:
            trial_config = _TrialConfigView(
                model_experiment_id=trial_id,
                model_name=model_name,
                model_params=dict(params),
            )
            model = self.model_factory.build(trial_config)
            trial.model_class_name = type(model).__name__

            start = perf_counter()
            model.fit(X_train, y_train)
            trial.training_seconds = round(perf_counter() - start, 4)

            start = perf_counter()
            predictions = model.predict(X_validation)
            trial.prediction_seconds = round(perf_counter() - start, 4)

            rmse = float(np.sqrt(mean_squared_error(y_validation, predictions)))
            trial.metrics = {
                "MAE": float(mean_absolute_error(y_validation, predictions)),
                "RMSE": rmse,
                "R2": float(r2_score(y_validation, predictions)),
            }
            trial.success = True

        run_name = f"{getattr(tuning_config, 'tuning_config_id', 'tuning')}_{trial_id}"

        if (
            self.mlflow_tracker is not None
            and self.mlflow_tracker.config.is_tracking_enabled
        ):
            try:
                with self.mlflow_tracker.start_run(run_name=run_name) as active_run:
                    if hasattr(active_run, "info") and hasattr(
                        active_run.info, "run_id"
                    ):
                        trial.mlflow_run_id = active_run.info.run_id

                    try:
                        if trial.params:
                            self.mlflow_tracker.log_params(dict(trial.params))
                    except Exception as log_err:
                        logging.warning(
                            f"Failed to log params to MLflow for trial '{trial_id}': {log_err}"
                        )

                    # Execute model trial exactly once
                    try:
                        _execute_model_trial()
                    except Exception as trial_err:
                        trial.success = False
                        trial.error = str(trial_err)
                        logging.error(
                            f"Trial '{trial_id}' execution failed: {trial_err}",
                            exc_info=True,
                        )

                    # Post-execution MLflow tracking — failure here will NOT mark trial as failed
                    try:
                        tags = {
                            "stage": "hyperparameter_tuning",
                            "feature_experiment_id": feature_experiment_id,
                            "feature_config_signature": feature_config_signature or "",
                            "model_name": trial.model_name,
                            "model_class_name": trial.model_class_name or "",
                            "trial_id": trial.trial_id,
                            "tuning_config_id": getattr(
                                tuning_config, "tuning_config_id", ""
                            ),
                            "tuning_config_signature": getattr(
                                tuning_config, "config_signature", ""
                            ),
                            "ranking_metric": tuning_config.ranking_metric,
                            "ranking_direction": tuning_config.ranking_direction,
                            "status": "success" if trial.success else "failed",
                        }
                        if trial.error:
                            tags["error"] = str(trial.error)[:250]
                        self.mlflow_tracker.log_tags(tags)

                        if trial.success and trial.metrics:
                            metrics_to_log = {
                                **trial.metrics,
                                "training_time_seconds": trial.training_seconds,
                                "prediction_time_seconds": trial.prediction_seconds,
                            }
                            self.mlflow_tracker.log_metrics(metrics_to_log)
                    except Exception as mlflow_log_err:
                        logging.warning(
                            f"MLflow logging failed post-execution for trial '{trial_id}': {mlflow_log_err}"
                        )

            except Exception as mlflow_start_err:
                logging.warning(
                    f"Failed to start MLflow run for trial '{trial_id}': {mlflow_start_err}. "
                    "Executing trial without MLflow tracking."
                )
                try:
                    _execute_model_trial()
                except Exception as trial_err:
                    trial.success = False
                    trial.error = str(trial_err)
                    logging.error(
                        f"Trial '{trial_id}' execution failed: {trial_err}",
                        exc_info=True,
                    )
        else:
            try:
                _execute_model_trial()
            except Exception as trial_err:
                trial.success = False
                trial.error = str(trial_err)
                logging.error(
                    f"Trial '{trial_id}' execution failed: {trial_err}", exc_info=True
                )

        return trial

    # ==================================================================
    # RANKING & COMPARISON
    # ==================================================================
    @staticmethod
    def _directional_value(
        trial: SalaryModelTuningTrialResult, metric: str, direction: str
    ) -> float:
        value = trial.metrics.get(metric) if trial.metrics else None
        try:
            numeric = float(value)
            if not math.isfinite(numeric):
                return math.inf
        except (TypeError, ValueError):
            return math.inf
        return numeric if direction == "minimize" else -numeric

    def _rank_trials(
        self,
        trials: Tuple[SalaryModelTuningTrialResult, ...],
        metric: str,
        direction: str,
    ) -> Tuple[SalaryModelTuningTrialResult, ...]:
        valid = [
            t
            for t in trials
            if self._directional_value(t, metric, direction) != math.inf
        ]
        if not valid:
            raise ValueError(
                f"No successful trials produced a valid finite '{metric}' value."
            )
        return tuple(
            sorted(
                valid,
                key=lambda t: (
                    self._directional_value(t, metric, direction),
                    t.trial_id,
                ),
            )
        )

    @staticmethod
    def _compare_to_baseline(
        baseline_score: float,
        best_score: float,
        direction: str,
        min_improvement_threshold: float = 0.0,
    ) -> Tuple[float, float, bool]:
        if direction == "minimize":
            improvement = baseline_score - best_score
        else:
            improvement = best_score - baseline_score

        improved = improvement > min_improvement_threshold

        if baseline_score == 0:
            improvement_percentage = float("nan")
        else:
            improvement_percentage = (improvement / abs(baseline_score)) * 100.0

        return (
            round(improvement, 6),
            (
                round(improvement_percentage, 4)
                if math.isfinite(improvement_percentage)
                else improvement_percentage
            ),
            improved,
        )

    # ==================================================================
    # REPORTING
    # ==================================================================
    def _generate_reports(
        self, summary: SalaryModelTuningSummary, tuning_config: SalaryModelTuningConfig
    ) -> None:
        model_dir = self.base_artifacts_dir / summary.model_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = model_dir / f"run_{timestamp}"
        latest_dir = model_dir / "latest"
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "tuning_summary.json").write_text(
            summary.to_json(), encoding="utf-8"
        )

        with (run_dir / "tuning_trials.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(
                (
                    "trial_id",
                    "model_name",
                    "status",
                    f"ranking_metric_{summary.ranking_metric}",
                    "MAE",
                    "RMSE",
                    "R2",
                    "training_seconds",
                    "prediction_seconds",
                    "hyperparameters",
                    "error",
                )
            )
            for trial in list(summary.successful_trials) + list(summary.failed_trials):
                writer.writerow(
                    (
                        trial.trial_id,
                        trial.model_name,
                        "SUCCESS" if trial.success else "FAILED",
                        (
                            trial.metrics.get(summary.ranking_metric)
                            if trial.metrics
                            else None
                        ),
                        trial.metrics.get("MAE") if trial.metrics else None,
                        trial.metrics.get("RMSE") if trial.metrics else None,
                        trial.metrics.get("R2") if trial.metrics else None,
                        trial.training_seconds,
                        trial.prediction_seconds,
                        json.dumps(trial.params),
                        trial.error,
                    )
                )

        baseline_config = {
            "winner_model_experiment_id": summary.winner_model_experiment_id,
            "model_name": summary.model_name,
            "model_class_name": summary.model_class_name,
            "baseline_params": summary.baseline_params,
            "baseline_config_signature": summary.baseline_config_signature,
            "baseline_score": summary.baseline_score,
            "ranking_metric": summary.ranking_metric,
        }
        (run_dir / "baseline_config.json").write_text(
            json.dumps(baseline_config, indent=2, default=str), encoding="utf-8"
        )

        best_config = {
            "trial_id": summary.best_trial_id,
            "model_name": summary.model_name,
            "best_params": summary.best_params,
            "best_config_signature": summary.best_config_signature,
            "best_score": summary.best_score,
            "ranking_metric": summary.ranking_metric,
            "tuning_improved_baseline": summary.tuning_improved_baseline,
            "preferred_params": summary.preferred_params,
            "preferred_config_signature": summary.preferred_config_signature,
        }
        (run_dir / "best_config.json").write_text(
            json.dumps(best_config, indent=2, default=str), encoding="utf-8"
        )

        run_metadata = {
            "timestamp": timestamp,
            "feature_experiment_id": summary.feature_experiment_id,
            "feature_config_signature": summary.feature_config_signature,
            "model_name": summary.model_name,
            "tuning_config_id": getattr(tuning_config, "tuning_config_id", ""),
            "tuning_config_signature": getattr(tuning_config, "config_signature", ""),
            "ranking_metric": summary.ranking_metric,
            "ranking_direction": summary.ranking_direction,
            "total_trial_count": summary.total_trial_count,
            "successful_trial_count": summary.successful_trial_count,
            "failed_trial_count": summary.failed_trial_count,
            "execution_seconds": summary.execution_seconds,
            "best_trial_id": summary.best_trial_id,
            "baseline_score": summary.baseline_score,
            "best_score": summary.best_score,
            "tuning_improved_baseline": summary.tuning_improved_baseline,
        }
        (run_dir / "run_metadata.json").write_text(
            json.dumps(run_metadata, indent=2), encoding="utf-8"
        )

        temp_latest = model_dir / "_latest_tmp"
        if temp_latest.exists():
            shutil.rmtree(temp_latest)
        shutil.copytree(run_dir, temp_latest)
        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        temp_latest.rename(latest_dir)

        summary.report_artifacts_dir = str(run_dir)

        logging.info(f"Tuning reports saved at: {run_dir.resolve()}")
        logging.info(f"Latest reports available at: {latest_dir.resolve()}")

    # ==================================================================
    # MLFLOW TRACKING SUMMARY
    # ==================================================================
    def _track_tuning_summary(
        self, summary: SalaryModelTuningSummary, tuning_config: SalaryModelTuningConfig
    ) -> Optional[str]:
        if self.mlflow_tracker is None:
            logging.info(
                "MLflow tracker not provided. Skipping tuning-summary MLflow tracking."
            )
            return None

        run_name = (
            f"tuning_summary_{summary.model_name}_{summary.feature_experiment_id}"
        )
        try:
            with self.mlflow_tracker.start_run(run_name=run_name) as active_run:
                self.mlflow_tracker.log_tags(
                    {
                        "stage": "hyperparameter_tuning_summary",
                        "feature_experiment_id": summary.feature_experiment_id,
                        "feature_config_signature": summary.feature_config_signature
                        or "",
                        "winner_model_experiment_id": summary.winner_model_experiment_id,
                        "model_name": summary.model_name,
                        "best_trial_id": summary.best_trial_id,
                        "tuning_improved_baseline": str(
                            summary.tuning_improved_baseline
                        ),
                    }
                )
                self.mlflow_tracker.log_params(
                    {
                        "ranking_metric": summary.ranking_metric,
                        "ranking_direction": summary.ranking_direction,
                        "search_strategy": getattr(
                            tuning_config, "search_strategy", ""
                        ),
                        "tuning_config_id": getattr(
                            tuning_config, "tuning_config_id", ""
                        ),
                        "tuning_config_signature": getattr(
                            tuning_config, "config_signature", ""
                        ),
                        "total_trial_count": summary.total_trial_count,
                        "successful_trial_count": summary.successful_trial_count,
                        "failed_trial_count": summary.failed_trial_count,
                    }
                )
                self.mlflow_tracker.log_metrics(
                    {
                        "baseline_score": float(summary.baseline_score),
                        "best_score": float(summary.best_score),
                        "improvement": float(summary.improvement),
                        "improvement_percentage": (
                            float(summary.improvement_percentage)
                            if math.isfinite(summary.improvement_percentage)
                            else 0.0
                        ),
                    }
                )
                if summary.report_artifacts_dir:
                    self.mlflow_tracker.log_artifacts(
                        summary.report_artifacts_dir,
                        artifact_folder="hyperparameter_tuning",
                    )

                run_id = (
                    active_run.info.run_id
                    if hasattr(active_run, "info")
                    and hasattr(active_run.info, "run_id")
                    else None
                )
                logging.info(f"Tuning-summary MLflow run created: {run_id}")
                return run_id
        except Exception as e:
            logging.warning(
                f"Tuning summary MLflow tracking failed: {e}", exc_info=True
            )
            return None

    # ==================================================================
    # CONSOLE SUMMARY
    # ==================================================================
    @staticmethod
    def _log_console_summary(summary: SalaryModelTuningSummary) -> None:
        logging.info("")
        logging.info("Best trial:")
        logging.info(summary.best_trial_id)
        logging.info("")
        logging.info(f"Best {summary.ranking_metric}:")
        logging.info(str(summary.best_score))
        logging.info("")
        logging.info(f"Baseline {summary.ranking_metric}:")
        logging.info(str(summary.baseline_score))
        logging.info("")
        if math.isfinite(summary.improvement_percentage):
            logging.info(f"Improvement: {summary.improvement_percentage:.2f}%")
        else:
            logging.info("Improvement: n/a (baseline score was 0)")
        logging.info(f"Tuning improved baseline: {summary.tuning_improved_baseline}")
        logging.info("")
        logging.info("Best parameters:")
        logging.info(str(summary.best_params))

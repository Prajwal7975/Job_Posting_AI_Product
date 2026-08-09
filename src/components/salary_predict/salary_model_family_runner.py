"""
src/components/salary_predict/salary_model_family_experiment_runner.py

Model Family Experiment Orchestrator.

Responsibilities
----------------
This component orchestrates the comparison of multiple model-family
experiments while keeping the selected feature configuration fixed.

It does NOT:
    - build models
    - train models
    - predict with models
    - calculate model metrics

Those responsibilities belong to:

    SalaryModelFactory
    SalarySingleModelExperimentRunner

It DOES:
    - retrieve enabled model configurations
    - execute every configured model independently
    - isolate individual model failures
    - collect experiment results
    - rank successful experiments
    - select the winner
    - generate comparison artifacts
    - maintain a latest/ report directory
    - create an optional family-level MLflow summary run

MLflow responsibilities
-----------------------
Individual model MLflow runs are handled by:

    SalarySingleModelExperimentRunner
        -> SalaryMLflowTracker

This orchestrator does not directly import MLflow.

For the family-level summary, it also uses the same
SalaryMLflowTracker abstraction through:

    tracker.start_run()
    tracker.log_tags()
    tracker.log_params()
    tracker.log_metrics()
    tracker.log_artifacts()

This keeps MLflow infrastructure centralized in SalaryMLflowTracker.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, List, Optional, Tuple

import pandas as pd

from src.configs.salary_predict.salary_model_factory_config import (
    get_model_experiment_configs,
)

from src.components.salary_predict.single_salary_model_runner import (
    SalaryModelExperimentResult,
    SalarySingleModelExperimentRunner,
)

from src.entity.salary_model_family_entity import (
    SalaryModelFamilyExperimentSummary,
)

from src.exception import CustomException
from src.logger import logging


class SalaryModelFamilyExperimentRunner:
    """
    Orchestrates the complete model-family comparison.

    The feature configuration remains fixed while every enabled
    model-family configuration is executed independently.

    Adding a new model should only require updating:

        1. SalaryModelExperimentConfig registry
        2. SalaryModelFactory registry

    This orchestrator automatically receives the new model through
    get_model_experiment_configs().
    """

    _METRIC_DIRECTIONS = {
        "MAE": "minimize",
        "RMSE": "minimize",
        "MAPE": "minimize",
        "R2": "maximize",
    }

    def __init__(
        self,
        single_model_runner: SalarySingleModelExperimentRunner,
        mlflow_tracker=None,
        config_provider: Callable = get_model_experiment_configs,
        base_artifacts_dir: str = ("artifacts/salary_model_family_experiments"),
    ) -> None:

        if single_model_runner is None:
            raise ValueError("single_model_runner must be provided.")

        if not callable(config_provider):
            raise ValueError("config_provider must be callable.")

        self.single_model_runner = single_model_runner
        self.mlflow_tracker = mlflow_tracker
        self.config_provider = config_provider
        self.base_artifacts_dir = Path(base_artifacts_dir)

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def run_experiments(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_experiment_id: str,
        feature_config_signature: Optional[str] = None,
        ranking_metric: str = "RMSE",
    ) -> SalaryModelFamilyExperimentSummary:

        logging.info("=" * 70)
        logging.info("MODEL FAMILY EXPERIMENT RUN STARTED")
        logging.info("=" * 70)

        start_time = perf_counter()

        # --------------------------------------------------------------
        # 1. Validate ranking metric
        # --------------------------------------------------------------

        metric = self._normalize_ranking_metric(ranking_metric)

        ranking_direction = self._METRIC_DIRECTIONS[metric]

        # --------------------------------------------------------------
        # 2. Validate feature lineage
        # --------------------------------------------------------------

        if (
            not isinstance(feature_experiment_id, str)
            or not feature_experiment_id.strip()
        ):
            raise CustomException(
                ValueError("feature_experiment_id must be a non-empty string."),
                sys,
            )

        # --------------------------------------------------------------
        # 3. Load model experiment configurations
        # --------------------------------------------------------------

        try:
            configs = tuple(self.config_provider())

        except Exception as e:

            logging.error(
                "Failed to retrieve model experiment configurations.",
                exc_info=True,
            )

            raise CustomException(
                RuntimeError(
                    "Failed to retrieve model experiment configurations: " f"{e}"
                ),
                sys,
            ) from e

        if not configs:

            raise CustomException(
                ValueError(
                    "No enabled model experiment configurations " "were returned."
                ),
                sys,
            )

        logging.info(
            "Feature experiment ID    : %s",
            feature_experiment_id,
        )

        logging.info(
            "Feature config signature : %s",
            feature_config_signature,
        )

        logging.info(
            "Enabled model count      : %d",
            len(configs),
        )

        logging.info(
            "Ranking metric           : %s (%s)",
            metric,
            ranking_direction,
        )

        # --------------------------------------------------------------
        # 4. Run every model independently
        # --------------------------------------------------------------

        successful_results: List[SalaryModelExperimentResult] = []

        failed_results: List[SalaryModelExperimentResult] = []

        for config in configs:

            experiment_id = getattr(
                config,
                "model_experiment_id",
                "unknown",
            )

            model_name = getattr(
                config,
                "model_name",
                "unknown",
            )

            logging.info("-" * 70)
            logging.info(
                "Running model experiment: %s (%s)",
                experiment_id,
                model_name,
            )

            try:

                result = self.single_model_runner.run(
                    config=config,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    feature_experiment_id=feature_experiment_id,
                )

                if not isinstance(
                    result,
                    SalaryModelExperimentResult,
                ):
                    raise TypeError(
                        "Single model runner returned "
                        f"{type(result).__name__}; expected "
                        "SalaryModelExperimentResult."
                    )

                if result.success:

                    successful_results.append(result)

                    logging.info(
                        "Model experiment '%s' completed successfully.",
                        experiment_id,
                    )

                else:

                    failed_results.append(result)

                    logging.warning(
                        "Model experiment '%s' returned failed result: %s",
                        experiment_id,
                        result.error,
                    )

            except Exception as e:

                # ------------------------------------------------------
                # IMPORTANT:
                # A single model failure must not stop M0-M5.
                # ------------------------------------------------------

                logging.error(
                    "Model experiment '%s' failed: %s",
                    experiment_id,
                    e,
                    exc_info=True,
                )

                failed_results.append(
                    SalaryModelExperimentResult(
                        experiment_id=experiment_id,
                        model_name=model_name,
                        feature_experiment_id=feature_experiment_id,
                        config_signature=getattr(
                            config,
                            "config_signature",
                            "unknown",
                        ),
                        success=False,
                        error=str(e),
                        model_class_name=None,
                        model_params=dict(
                            getattr(
                                config,
                                "model_params",
                                {},
                            )
                        ),
                    )
                )

        # --------------------------------------------------------------
        # 5. At least one model must succeed
        # --------------------------------------------------------------

        if not successful_results:

            failed_ids = [result.experiment_id for result in failed_results]

            raise CustomException(
                RuntimeError(
                    f"All {len(configs)} model-family experiments failed. "
                    f"Failed experiments: {failed_ids}"
                ),
                sys,
            )

        # --------------------------------------------------------------
        # 6. Rank models
        # --------------------------------------------------------------

        try:

            ranked_results = self._rank_results(
                results=successful_results,
                metric=metric,
                direction=ranking_direction,
            )

        except Exception as e:

            logging.error(
                "Model-family ranking failed.",
                exc_info=True,
            )

            raise CustomException(
                RuntimeError(f"Failed to rank model-family experiments: {e}"),
                sys,
            ) from e

        if not ranked_results:

            raise CustomException(
                RuntimeError("No valid model results remained after ranking."),
                sys,
            )

        # --------------------------------------------------------------
        # 7. Select winner
        # --------------------------------------------------------------

        winner = ranked_results[0]

        execution_seconds = round(
            perf_counter() - start_time,
            4,
        )

        logging.info("=" * 70)
        logging.info("MODEL FAMILY WINNER SELECTED")
        logging.info("=" * 70)

        logging.info(
            "Winner ID      : %s",
            winner.experiment_id,
        )

        logging.info(
            "Winner Model   : %s",
            winner.model_name,
        )

        logging.info(
            "Winner Class   : %s",
            winner.model_class_name,
        )

        logging.info(
            "Winner %s     : %.6f",
            metric,
            winner.metrics[metric],
        )

        # --------------------------------------------------------------
        # 8. Build summary entity
        # --------------------------------------------------------------

        summary = SalaryModelFamilyExperimentSummary(
            feature_experiment_id=feature_experiment_id,
            feature_config_signature=feature_config_signature,
            ranking_metric=metric,
            ranking_direction=ranking_direction,
            experiment_count=len(configs),
            successful_experiment_count=len(successful_results),
            failed_experiment_count=len(failed_results),
            winner_experiment_id=winner.experiment_id,
            winner_model_name=winner.model_name,
            winner_model_class=winner.model_class_name,
            winner_score=winner.metrics.get(metric),
            winner_config_signature=winner.config_signature,
            winner_model_artifact_path=(winner.model_artifact_path),
            winner_mlflow_run_id=winner.mlflow_run_id,
            successful_results=successful_results,
            failed_results=failed_results,
            ranked_results=ranked_results,
            execution_seconds=execution_seconds,
        )

        # --------------------------------------------------------------
        # 9. Generate local artifacts
        # --------------------------------------------------------------

        try:

            self._generate_reports(
                summary=summary,
                winner=winner,
                configs=configs,
            )

        except Exception as e:

            logging.error(
                "Failed to generate model-family reports.",
                exc_info=True,
            )

            raise CustomException(
                RuntimeError(f"Failed to generate model-family reports: {e}"),
                sys,
            ) from e

        # --------------------------------------------------------------
        # 10. Track family-level comparison in MLflow
        # --------------------------------------------------------------

        try:

            summary.mlflow_summary_run_id = self._track_family_summary(
                summary=summary,
            )

        except Exception as e:

            # Local artifacts and model results are already valid.
            # MLflow summary failure should not destroy the comparison.

            logging.warning(
                "Family-level MLflow tracking failed: %s",
                e,
                exc_info=True,
            )

        # --------------------------------------------------------------
        # 11. Log final summary
        # --------------------------------------------------------------

        self._log_console_summary(summary)

        logging.info("=" * 70)
        logging.info("MODEL FAMILY EXPERIMENT RUN COMPLETED")
        logging.info("=" * 70)

        return summary

    # ==================================================================
    # RANKING
    # ==================================================================

    def _normalize_ranking_metric(
        self,
        ranking_metric: str,
    ) -> str:

        if not isinstance(ranking_metric, str):

            raise CustomException(
                TypeError("ranking_metric must be a string."),
                sys,
            )

        metric = ranking_metric.strip().upper()

        if metric not in self._METRIC_DIRECTIONS:

            raise CustomException(
                ValueError(
                    f"Unsupported ranking metric '{ranking_metric}'. "
                    f"Supported metrics: "
                    f"{sorted(self._METRIC_DIRECTIONS)}"
                ),
                sys,
            )

        return metric

    def _rank_results(
        self,
        results: List[SalaryModelExperimentResult],
        metric: str,
        direction: str,
    ) -> List[SalaryModelExperimentResult]:

        valid_results = []

        for result in results:

            value = result.metrics.get(metric)

            try:

                numeric_value = float(value)

                if not math.isfinite(numeric_value):
                    raise ValueError

            except (TypeError, ValueError):

                logging.warning(
                    "Experiment '%s' has invalid %s value: %r",
                    result.experiment_id,
                    metric,
                    value,
                )

                continue

            valid_results.append(result)

        if not valid_results:

            raise CustomException(
                ValueError(
                    f"No successful model experiments produced "
                    f"a valid '{metric}' value."
                ),
                sys,
            )

        descending = direction == "maximize"

        return sorted(
            valid_results,
            key=lambda result: (
                (
                    -float(result.metrics[metric])
                    if descending
                    else float(result.metrics[metric])
                ),
                result.experiment_id,
            ),
        )

    # ==================================================================
    # REPORT GENERATION
    # ==================================================================

    def _generate_reports(
        self,
        summary: SalaryModelFamilyExperimentSummary,
        winner: SalaryModelExperimentResult,
        configs: tuple,
    ) -> None:

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

        self.base_artifacts_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        run_dir = self.base_artifacts_dir / f"run_{timestamp}"

        run_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        summary.report_artifacts_dir = str(run_dir.resolve())

        # --------------------------------------------------------------
        # Summary
        # --------------------------------------------------------------

        (run_dir / "model_family_summary.json").write_text(
            json.dumps(
                summary.to_dict(),
                indent=4,
                default=str,
            ),
            encoding="utf-8",
        )

        # --------------------------------------------------------------
        # Winner configuration
        # --------------------------------------------------------------

        winner_config = next(
            (
                config
                for config in configs
                if config.model_experiment_id == winner.experiment_id
            ),
            None,
        )

        if winner_config is None:

            raise CustomException(
                ValueError(
                    f"Winner configuration "
                    f"'{winner.experiment_id}' "
                    "could not be found."
                ),
                sys,
            )

        winner_dict = winner_config.to_dict()

        winner_dict.update(
            {
                "feature_experiment_id": summary.feature_experiment_id,
                "feature_config_signature": summary.feature_config_signature,
                "winner_score": summary.winner_score,
                "winner_mlflow_run_id": summary.winner_mlflow_run_id,
                "winner_model_artifact_path": summary.winner_model_artifact_path,
            }
        )

        (run_dir / "winner_model_config.json").write_text(
            json.dumps(
                winner_dict,
                indent=4,
                default=str,
            ),
            encoding="utf-8",
        )

        # --------------------------------------------------------------
        # Comparison CSV
        # --------------------------------------------------------------

        csv_path = run_dir / "model_family_comparison.csv"

        all_results = (
            list(summary.ranked_results)
            + [
                result
                for result in summary.successful_results
                if result.experiment_id
                not in {r.experiment_id for r in summary.ranked_results}
            ]
            + list(summary.failed_results)
        )

        metric_names = sorted(
            {metric for result in all_results for metric in result.metrics}
        )

        fieldnames = [
            "rank",
            "experiment_id",
            "model_name",
            "model_class_name",
            "feature_experiment_id",
            "feature_config_signature",
            "config_signature",
            "success",
            "error",
            "model_params",
            "training_seconds",
            "prediction_seconds",
            "mlflow_run_id",
            "artifact_directory",
            "model_artifact_path",
        ] + metric_names

        ranking_map = {
            result.experiment_id: rank
            for rank, result in enumerate(
                summary.ranked_results,
                start=1,
            )
        }

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for result in all_results:

                row = {
                    "rank": ranking_map.get(
                        result.experiment_id,
                        "",
                    ),
                    "experiment_id": result.experiment_id,
                    "model_name": result.model_name,
                    "model_class_name": result.model_class_name or "",
                    "feature_experiment_id": result.feature_experiment_id,
                    "feature_config_signature": summary.feature_config_signature or "",
                    "config_signature": result.config_signature,
                    "success": result.success,
                    "error": result.error or "",
                    "model_params": json.dumps(
                        result.model_params,
                        sort_keys=True,
                        default=str,
                    ),
                    "training_seconds": result.training_seconds,
                    "prediction_seconds": result.prediction_seconds,
                    "mlflow_run_id": result.mlflow_run_id or "",
                    "artifact_directory": result.artifact_directory or "",
                    "model_artifact_path": result.model_artifact_path or "",
                }

                for metric_name in metric_names:

                    row[metric_name] = result.metrics.get(
                        metric_name,
                        "",
                    )

                writer.writerow(row)

        # --------------------------------------------------------------
        # Model selection decision
        # --------------------------------------------------------------

        selection_report = {
            "selection_strategy": {
                "metric": summary.ranking_metric,
                "direction": summary.ranking_direction,
            },
            "winner": {
                "experiment_id": summary.winner_experiment_id,
                "model_name": summary.winner_model_name,
                "model_class": summary.winner_model_class,
                "score": summary.winner_score,
                "config_signature": summary.winner_config_signature,
                "model_artifact_path": summary.winner_model_artifact_path,
                "mlflow_run_id": summary.winner_mlflow_run_id,
            },
            "feature_lineage": {
                "feature_experiment_id": summary.feature_experiment_id,
                "feature_config_signature": summary.feature_config_signature,
            },
        }

        (run_dir / "model_selection_decision.json").write_text(
            json.dumps(
                selection_report,
                indent=4,
                default=str,
            ),
            encoding="utf-8",
        )

        # --------------------------------------------------------------
        # Run metadata
        # --------------------------------------------------------------

        metadata = {
            "artifact_schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "feature_experiment_id": summary.feature_experiment_id,
            "feature_config_signature": summary.feature_config_signature,
            "ranking_metric": summary.ranking_metric,
            "ranking_direction": summary.ranking_direction,
            "experiment_count": summary.experiment_count,
            "successful_experiment_count": summary.successful_experiment_count,
            "failed_experiment_count": summary.failed_experiment_count,
            "winner_experiment_id": summary.winner_experiment_id,
            "winner_model_name": summary.winner_model_name,
            "winner_model_class": summary.winner_model_class,
            "winner_score": summary.winner_score,
            "winner_config_signature": summary.winner_config_signature,
            "winner_model_artifact_path": summary.winner_model_artifact_path,
            "winner_mlflow_run_id": summary.winner_mlflow_run_id,
            "execution_seconds": summary.execution_seconds,
        }

        (run_dir / "run_metadata.json").write_text(
            json.dumps(
                metadata,
                indent=4,
                default=str,
            ),
            encoding="utf-8",
        )

        # --------------------------------------------------------------
        # latest/
        # --------------------------------------------------------------

        latest_dir = self.base_artifacts_dir / "latest"

        temp_latest = self.base_artifacts_dir / "_latest_tmp"

        if temp_latest.exists():
            shutil.rmtree(temp_latest)

        shutil.copytree(
            run_dir,
            temp_latest,
        )

        if latest_dir.exists():
            shutil.rmtree(latest_dir)

        temp_latest.rename(latest_dir)

        logging.info(
            "Model-family reports saved at: %s",
            run_dir.resolve(),
        )

        logging.info(
            "Latest reports available at: %s",
            latest_dir.resolve(),
        )

    # ==================================================================
    # FAMILY-LEVEL MLFLOW TRACKING
    # ==================================================================

    def _track_family_summary(
        self,
        summary: SalaryModelFamilyExperimentSummary,
    ) -> Optional[str]:
        """
        Create one MLflow run representing the model-family
        comparison decision.

        This uses SalaryMLflowTracker's existing start_run()
        API. No direct MLflow dependency exists here.
        """

        if self.mlflow_tracker is None:

            logging.info(
                "MLflow tracker not provided. " "Skipping family-level MLflow tracking."
            )

            return None

        try:

            run_name = (
                f"model_family_"
                f"{summary.feature_experiment_id}_"
                f"{summary.ranking_metric}"
            )

            with self.mlflow_tracker.start_run(run_name=run_name) as active_run:

                # ------------------------------------------------------
                # Tags
                # ------------------------------------------------------

                self.mlflow_tracker.log_tags(
                    {
                        "stage": "model_family_comparison",
                        "feature_experiment_id": summary.feature_experiment_id,
                        "feature_config_signature": summary.feature_config_signature,
                        "ranking_metric": summary.ranking_metric,
                        "ranking_direction": summary.ranking_direction,
                        "winner_experiment_id": summary.winner_experiment_id,
                        "winner_model": summary.winner_model_name,
                        "winner_model_class": summary.winner_model_class,
                        "winner_config_signature": summary.winner_config_signature,
                        "winner_mlflow_run_id": summary.winner_mlflow_run_id,
                    }
                )

                # ------------------------------------------------------
                # Parameters
                # ------------------------------------------------------

                self.mlflow_tracker.log_params(
                    {
                        "feature_experiment_id": summary.feature_experiment_id,
                        "ranking_metric": summary.ranking_metric,
                        "ranking_direction": summary.ranking_direction,
                        "experiment_count": summary.experiment_count,
                        "successful_experiments": summary.successful_experiment_count,
                        "failed_experiments": summary.failed_experiment_count,
                    }
                )

                # ------------------------------------------------------
                # Ranking metrics
                # ------------------------------------------------------

                metrics = {}

                if summary.winner_score is not None:

                    metrics["winner_score"] = float(summary.winner_score)

                for result in summary.ranked_results:

                    value = result.metrics.get(summary.ranking_metric)

                    if value is None:
                        continue

                    try:

                        numeric_value = float(value)

                        if math.isfinite(numeric_value):

                            metrics[
                                f"{result.experiment_id}_" f"{summary.ranking_metric}"
                            ] = numeric_value

                    except (
                        TypeError,
                        ValueError,
                    ):
                        continue

                # Winner timing
                if summary.ranked_results:

                    winner_result = summary.ranked_results[0]

                    metrics["winner_training_seconds"] = float(
                        winner_result.training_seconds
                    )

                    metrics["winner_prediction_seconds"] = float(
                        winner_result.prediction_seconds
                    )

                self.mlflow_tracker.log_metrics(metrics)

                # ------------------------------------------------------
                # Upload family comparison artifacts
                # ------------------------------------------------------

                if summary.report_artifacts_dir:

                    self.mlflow_tracker.log_artifacts(
                        summary.report_artifacts_dir,
                        artifact_folder=("model_family_comparison"),
                    )

                # ------------------------------------------------------
                # Extract MLflow run ID
                # ------------------------------------------------------

                run_id = active_run.info.run_id

                logging.info(
                    "Family-level MLflow run created: %s",
                    run_id,
                )

                return run_id

        except Exception as e:

            logging.warning(
                "Family-level MLflow tracking failed: %s",
                e,
                exc_info=True,
            )

            return None

    # ==================================================================
    # CONSOLE SUMMARY
    # ==================================================================

    def _log_console_summary(
        self,
        summary: SalaryModelFamilyExperimentSummary,
    ) -> None:

        logging.info("")
        logging.info("MODEL FAMILY RANKING SUMMARY")
        logging.info("-" * 60)

        for rank, result in enumerate(
            summary.ranked_results,
            start=1,
        ):

            score = result.metrics.get(
                summary.ranking_metric,
                "N/A",
            )

            logging.info(
                "%d. [%s] %s -> %s: %s",
                rank,
                result.experiment_id,
                result.model_name,
                summary.ranking_metric,
                score,
            )

        logging.info("-" * 60)

        logging.info(
            "Winner Model ID : %s",
            summary.winner_experiment_id,
        )

        logging.info(
            "Winner Name     : %s",
            summary.winner_model_name,
        )

        logging.info(
            "Winner Class    : %s",
            summary.winner_model_class,
        )

        logging.info(
            "Winner Score    : %s",
            summary.winner_score,
        )

        logging.info(
            "Winner Artifact : %s",
            summary.winner_model_artifact_path,
        )

        logging.info(
            "Winner MLflow   : %s",
            summary.winner_mlflow_run_id,
        )

        logging.info(
            "Summary MLflow  : %s",
            summary.mlflow_summary_run_id,
        )

        logging.info(
            "Reports         : %s",
            summary.report_artifacts_dir,
        )

        logging.info(
            "Successful      : %d/%d",
            summary.successful_experiment_count,
            summary.experiment_count,
        )

        if summary.failed_results:

            failed_ids = [result.experiment_id for result in summary.failed_results]

            logging.warning(
                "Failed Models (%d): %s",
                len(summary.failed_results),
                failed_ids,
            )

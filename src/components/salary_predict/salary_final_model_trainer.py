"""
src/components/salary_predict/salary_final_model_trainer.py

Final Model Trainer.

Consumes already-decided inputs:

    1. Winning feature configuration
    2. Winning model-family summary
    3. Hyperparameter tuning summary

Produces one fitted production-candidate sklearn Pipeline:

    Pipeline(
        preprocessor -> trained model
    )

The fitted pipeline is persisted as:

    final_model.joblib

This component DOES NOT:

    - select the model family
    - perform hyperparameter tuning
    - modify the test set
    - orchestrate the complete ML pipeline
    - deploy the model

Those responsibilities belong to other stages / the future
master pipeline orchestrator.

The trainer integrates existing project components:

    SalaryPreprocessorBuilder
    SalaryModelFactory
    SalaryMLflowTracker

through dependency injection.

The final artifact contains BOTH:

    preprocessing
    +
    trained model

inside one sklearn Pipeline.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import CustomException

from src.components.salary_predict.salary_model_factory import (
    SalaryModelFactory,
)

from src.configs.salary_predict.salary_final_model_trainer_config import (
    SalaryFinalModelConfig,
)

from src.entity.salary_predict.salary_final_model_trainer_entity import (
    SalaryFinalModelResult,
)

# ======================================================================
# INTERNAL MODEL CONFIG ADAPTER
# ======================================================================


@dataclass(frozen=True)
class _FinalModelConfigView:
    """
    Minimal configuration contract required by SalaryModelFactory.

    This is NOT another model configuration registry.

    It simply adapts the already-decided tuning result to the
    SalaryModelFactory.build() interface.
    """

    model_experiment_id: str
    model_name: str
    model_params: Dict[str, Any]


# ======================================================================
# FINAL MODEL TRAINER
# ======================================================================


class SalaryFinalModelTrainer:
    """
    Train and package the final production-candidate salary model.

    Responsibilities
    ----------------
    1. Validate upstream lineage.
    2. Build the existing preprocessor.
    3. Build the already-selected model using preferred parameters.
    4. Combine them into one sklearn Pipeline.
    5. Train on training data.
    6. Evaluate on validation data.
    7. Apply configured quality gate.
    8. Save the complete pipeline as joblib.
    9. Save metadata/artifacts.
    10. Track the run through SalaryMLflowTracker.

    This is NOT the master pipeline orchestrator.
    """

    def __init__(
        self,
        preprocessor_builder: Any,
        model_factory: Optional[SalaryModelFactory] = None,
        mlflow_tracker: Any = None,
        config: Optional[SalaryFinalModelConfig] = None,
    ) -> None:

        if preprocessor_builder is None:
            raise ValueError("preprocessor_builder must be provided.")

        self.preprocessor_builder = preprocessor_builder

        self.model_factory = (
            model_factory if model_factory is not None else SalaryModelFactory()
        )

        self.mlflow_tracker = mlflow_tracker

        self.config = config if config is not None else SalaryFinalModelConfig()

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def run(
        self,
        X_train: Any,
        y_train: Any,
        X_validation: Any,
        y_validation: Any,
        feature_config: Any,
        model_family_summary: Any,
        tuning_summary: Any,
    ) -> SalaryFinalModelResult:

        try:

            logging.info("=" * 70)
            logging.info("FINAL MODEL TRAINING STARTED")
            logging.info("=" * 70)

            # ----------------------------------------------------------
            # Validate input data
            # ----------------------------------------------------------

            self._validate_inputs(
                X_train,
                y_train,
                X_validation,
                y_validation,
            )

            # ----------------------------------------------------------
            # Validate upstream experiment lineage
            # ----------------------------------------------------------

            self._validate_lineage(
                model_family_summary,
                tuning_summary,
            )

            # ----------------------------------------------------------
            # Extract authoritative decisions
            # ----------------------------------------------------------

            model_name = tuning_summary.model_name

            preferred_params = dict(tuning_summary.preferred_params)

            feature_experiment_id = model_family_summary.feature_experiment_id

            feature_config_signature = getattr(
                model_family_summary,
                "feature_config_signature",
                None,
            )

            model_experiment_id = getattr(
                model_family_summary,
                "winner_experiment_id",
                None,
            )

            model_config_signature = getattr(
                model_family_summary,
                "winner_config_signature",
                None,
            )

            tuning_config_signature = getattr(
                tuning_summary,
                "preferred_config_signature",
                None,
            )

            # ----------------------------------------------------------
            # Create result entity
            # ----------------------------------------------------------

            result = SalaryFinalModelResult(
                success=False,
                model_name=model_name,
                feature_experiment_id=feature_experiment_id,
                feature_config_signature=feature_config_signature,
                model_experiment_id=model_experiment_id,
                model_config_signature=model_config_signature,
                tuning_config_signature=tuning_config_signature,
                preferred_params=preferred_params,
                validation_metric=self.config.validation_metric,
                validation_direction=self.config.validation_direction,
                validation_threshold=self.config.validation_threshold,
                validation_threshold_configured=(
                    self.config.validation_threshold is not None
                ),
            )

            # ----------------------------------------------------------
            # Prepare artifact directory
            # ----------------------------------------------------------

            run_dir = self._prepare_run_dir(model_name)

            result.artifact_directory = str(run_dir)

            run_name = f"final_salary_model_{model_name}"

            # ----------------------------------------------------------
            # MLflow tracked execution
            # ----------------------------------------------------------

            if self.mlflow_tracker is not None:

                logging.info("MLflow tracking enabled.")

                with self.mlflow_tracker.start_run(run_name=run_name) as active_run:

                    result.mlflow_run_id = active_run.info.run_id

                    self._track_start(
                        model_name=model_name,
                        preferred_params=preferred_params,
                        feature_experiment_id=(feature_experiment_id),
                        feature_config_signature=(feature_config_signature),
                        model_experiment_id=(model_experiment_id),
                        tuning_config_signature=(tuning_config_signature),
                    )

                    self._execute_training(
                        result=result,
                        feature_config=feature_config,
                        model_name=model_name,
                        preferred_params=preferred_params,
                        X_train=X_train,
                        y_train=y_train,
                        X_validation=X_validation,
                        y_validation=y_validation,
                        run_dir=run_dir,
                        model_family_summary=model_family_summary,
                        tuning_summary=tuning_summary,
                    )

                    self._track_completion(result)

            # ----------------------------------------------------------
            # Non-MLflow execution
            # ----------------------------------------------------------

            else:

                logging.info(
                    "No MLflow tracker supplied. "
                    "Running final training without MLflow."
                )

                self._execute_training(
                    result=result,
                    feature_config=feature_config,
                    model_name=model_name,
                    preferred_params=preferred_params,
                    X_train=X_train,
                    y_train=y_train,
                    X_validation=X_validation,
                    y_validation=y_validation,
                    run_dir=run_dir,
                    model_family_summary=model_family_summary,
                    tuning_summary=tuning_summary,
                )

            # ----------------------------------------------------------
            # Completion
            # ----------------------------------------------------------

            logging.info("=" * 70)
            logging.info(
                "FINAL MODEL TRAINING COMPLETED | "
                f"success={result.success} | "
                f"validation_passed={result.validation_passed}"
            )
            logging.info("=" * 70)

            return result

        except CustomException:
            raise

        except Exception as e:

            logging.error(
                "Final model training failed: %s",
                e,
                exc_info=True,
            )

            raise CustomException(
                e,
                sys,
            ) from e

    # ==================================================================
    # INPUT VALIDATION
    # ==================================================================

    @staticmethod
    def _validate_inputs(
        X_train: Any,
        y_train: Any,
        X_validation: Any,
        y_validation: Any,
    ) -> None:

        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must not be None.")

        if X_validation is None or y_validation is None:
            raise ValueError("X_validation and y_validation " "must not be None.")

        if len(X_train) != len(y_train):
            raise ValueError(
                "Training data length mismatch: " f"{len(X_train)} != {len(y_train)}"
            )

        if len(X_validation) != len(y_validation):
            raise ValueError(
                "Validation data length mismatch: "
                f"{len(X_validation)} != "
                f"{len(y_validation)}"
            )

        if len(X_train) == 0:
            raise ValueError("X_train must not be empty.")

        if len(X_validation) == 0:
            raise ValueError("X_validation must not be empty.")

    # ==================================================================
    # LINEAGE VALIDATION
    # ==================================================================

    @staticmethod
    def _validate_lineage(
        model_family_summary: Any,
        tuning_summary: Any,
    ) -> None:

        family_required = (
            "feature_experiment_id",
            "winner_model_name",
            "winner_experiment_id",
        )

        missing_family = [
            attribute
            for attribute in family_required
            if not hasattr(
                model_family_summary,
                attribute,
            )
        ]

        if missing_family:
            raise ValueError(
                "model_family_summary is missing "
                f"required attributes: {missing_family}"
            )

        tuning_required = (
            "model_name",
            "preferred_params",
            "preferred_config_signature",
        )

        missing_tuning = [
            attribute
            for attribute in tuning_required
            if not hasattr(
                tuning_summary,
                attribute,
            )
        ]

        if missing_tuning:
            raise ValueError(
                "tuning_summary is missing " f"required attributes: {missing_tuning}"
            )

        if tuning_summary.model_name != model_family_summary.winner_model_name:
            raise ValueError(
                "Model lineage mismatch. "
                f"Tuning model='{tuning_summary.model_name}' "
                f"but model-family winner="
                f"'{model_family_summary.winner_model_name}'."
            )

    # ==================================================================
    # CORE TRAINING
    # ==================================================================

    def _execute_training(
        self,
        result: SalaryFinalModelResult,
        feature_config: Any,
        model_name: str,
        preferred_params: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_validation: Any,
        y_validation: Any,
        run_dir: Path,
        model_family_summary: Any,
        tuning_summary: Any,
    ) -> None:

        # --------------------------------------------------------------
        # Build preprocessing
        # --------------------------------------------------------------

        logging.info("BUILDING PREPROCESSOR")

        preprocessor = self.preprocessor_builder.build(feature_config)

        # --------------------------------------------------------------
        # Build selected model
        # --------------------------------------------------------------

        logging.info("BUILDING FINAL MODEL")

        model_view = _FinalModelConfigView(
            model_experiment_id=(result.model_experiment_id or "FINAL"),
            model_name=model_name,
            model_params=preferred_params,
        )

        model = self.model_factory.build(model_view)

        result.model_class_name = type(model).__name__

        # --------------------------------------------------------------
        # Create ONE production pipeline
        # --------------------------------------------------------------

        if preprocessor is None:
            pipeline = model
        else:
            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor,
                    ),
                    (
                        "model",
                        model,
                    ),
                ]
            )

        # --------------------------------------------------------------
        # Train
        # --------------------------------------------------------------

        logging.info("FINAL MODEL TRAINING START")

        training_start = perf_counter()

        pipeline.fit(
            X_train,
            y_train,
        )

        result.training_seconds = round(
            perf_counter() - training_start,
            4,
        )

        logging.info("FINAL MODEL TRAINING COMPLETE | " f"{result.training_seconds}s")

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        logging.info("FINAL MODEL VALIDATION START")

        predictions = pipeline.predict(X_validation)

        validation_metrics = self._calculate_validation_metrics(
            y_validation,
            predictions,
        )

        result.validation_metrics = validation_metrics

        logging.info(
            "VALIDATION METRICS: %s",
            validation_metrics,
        )

        # --------------------------------------------------------------
        # Quality gate
        # --------------------------------------------------------------

        result.validation_passed = self._apply_quality_gate(validation_metrics)

        # --------------------------------------------------------------
        # Persist complete pipeline
        # --------------------------------------------------------------

        model_path = run_dir / self.config.model_filename

        joblib.dump(
            pipeline,
            model_path,
        )

        if not model_path.exists():
            raise RuntimeError("Final model artifact was not created: " f"{model_path}")

        result.model_artifact_path = str(model_path)

        logging.info(
            "FINAL MODEL ARTIFACT SAVED: %s",
            model_path,
        )

        # --------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------

        self._write_metadata(
            run_dir=run_dir,
            result=result,
            feature_config=feature_config,
            model_family_summary=model_family_summary,
            tuning_summary=tuning_summary,
        )

        # --------------------------------------------------------------
        # MLflow
        # --------------------------------------------------------------

        if self.mlflow_tracker is not None:

            self.mlflow_tracker.log_metrics(
                {
                    "validation_MAE": (validation_metrics["MAE"]),
                    "validation_RMSE": (validation_metrics["RMSE"]),
                    "validation_R2": (validation_metrics["R2"]),
                    "training_time_seconds": (result.training_seconds),
                    "validation_passed": (int(result.validation_passed)),
                }
            )

            self.mlflow_tracker.log_artifacts(
                str(run_dir),
                artifact_folder="final_model",
            )

            logging.info("FINAL MODEL ARTIFACTS LOGGED TO MLFLOW")

        # --------------------------------------------------------------
        # Technical training succeeded
        # --------------------------------------------------------------

        result.success = True

        logging.info("FINAL MODEL TRAINING COMPONENT COMPLETED")

    # ==================================================================
    # VALIDATION METRICS
    # ==================================================================

    @staticmethod
    def _calculate_validation_metrics(
        y_true: Any,
        predictions: Any,
    ) -> Dict[str, float]:

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    predictions,
                )
            )
        )

        return {
            "MAE": float(
                mean_absolute_error(
                    y_true,
                    predictions,
                )
            ),
            "RMSE": rmse,
            "R2": float(
                r2_score(
                    y_true,
                    predictions,
                )
            ),
        }

    # ==================================================================
    # QUALITY GATE
    # ==================================================================

    def _apply_quality_gate(
        self,
        validation_metrics: Dict[str, float],
    ) -> bool:

        metric = self.config.validation_metric

        score = validation_metrics[metric]

        threshold = self.config.validation_threshold

        # --------------------------------------------------------------
        # No explicit threshold
        # --------------------------------------------------------------

        if threshold is None:

            logging.info(
                "No explicit quality threshold "
                f"configured for {metric}. "
                "Validation considered successful "
                "because metric calculation succeeded."
            )

            return True

        # --------------------------------------------------------------
        # Minimize metrics
        # --------------------------------------------------------------

        if self.config.validation_direction == "minimize":

            passed = score <= threshold

        # --------------------------------------------------------------
        # Maximize metrics
        # --------------------------------------------------------------

        else:

            passed = score >= threshold

        logging.info(
            "QUALITY GATE | metric=%s | score=%.6f | "
            "threshold=%.6f | direction=%s | result=%s",
            metric,
            score,
            threshold,
            self.config.validation_direction,
            "PASSED" if passed else "FAILED",
        )

        return passed

    # ==================================================================
    # ARTIFACT DIRECTORY
    # ==================================================================

    def _prepare_run_dir(
        self,
        model_name: str,
    ) -> Path:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        run_dir = Path(self.config.artifact_dir) / f"run_{timestamp}_{model_name}"

        # Historical experiment runs should not overwrite
        # each other.

        run_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        return run_dir

    # ==================================================================
    # METADATA
    # ==================================================================

    def _write_metadata(
        self,
        run_dir: Path,
        result: SalaryFinalModelResult,
        feature_config: Any,
        model_family_summary: Any,
        tuning_summary: Any,
    ) -> None:

        # --------------------------------------------------------------
        # Main model metadata
        # --------------------------------------------------------------

        model_metadata = {
            "stage": "final_model_training",
            "model_name": result.model_name,
            "model_class_name": result.model_class_name,
            "preferred_params": (result.preferred_params),
            "feature_experiment_id": (result.feature_experiment_id),
            "feature_config_signature": (result.feature_config_signature),
            "model_experiment_id": (result.model_experiment_id),
            "model_config_signature": (result.model_config_signature),
            "tuning_config_signature": (result.tuning_config_signature),
            "training_seconds": (result.training_seconds),
            "validation_metrics": (result.validation_metrics),
            "validation_metric": (result.validation_metric),
            "validation_direction": (result.validation_direction),
            "validation_threshold": (result.validation_threshold),
            "validation_threshold_configured": (result.validation_threshold_configured),
            "validation_passed": (result.validation_passed),
            "production_approved": (result.validation_passed),
            "model_artifact_path": (result.model_artifact_path),
            "mlflow_run_id": (result.mlflow_run_id),
            "generated_at": result.generated_at,
        }

        self._write_json(
            run_dir / "model_metadata.json",
            model_metadata,
        )

        # --------------------------------------------------------------
        # Feature configuration
        # --------------------------------------------------------------

        if hasattr(
            feature_config,
            "to_dict",
        ):

            feature_config_dict = feature_config.to_dict()

        else:

            feature_config_dict = {
                "feature_experiment_id": (result.feature_experiment_id),
                "feature_config_signature": (result.feature_config_signature),
            }

        self._write_json(
            run_dir / "feature_config.json",
            feature_config_dict,
        )

        # --------------------------------------------------------------
        # Model configuration
        # --------------------------------------------------------------

        model_config_dict = {
            "model_name": result.model_name,
            "model_class_name": result.model_class_name,
            "model_params": result.preferred_params,
            "model_experiment_id": result.model_experiment_id,
            "model_config_signature": result.model_config_signature,
        }

        self._write_json(
            run_dir / "model_config.json",
            model_config_dict,
        )

        # --------------------------------------------------------------
        # Tuning summary
        # --------------------------------------------------------------

        if hasattr(
            tuning_summary,
            "to_json",
        ):

            tuning_json = tuning_summary.to_json()

        elif hasattr(
            tuning_summary,
            "to_dict",
        ):

            tuning_json = json.dumps(
                tuning_summary.to_dict(),
                indent=2,
                default=str,
            )

        else:

            tuning_json = json.dumps(
                {
                    "model_name": result.model_name,
                    "preferred_params": (result.preferred_params),
                    "preferred_config_signature": (result.tuning_config_signature),
                },
                indent=2,
                default=str,
            )

        (run_dir / "tuning_summary.json").write_text(
            tuning_json,
            encoding="utf-8",
        )

        # --------------------------------------------------------------
        # Validation metrics
        # --------------------------------------------------------------

        self._write_json(
            run_dir / "validation_metrics.json",
            result.validation_metrics,
        )

        # --------------------------------------------------------------
        # Run metadata
        # --------------------------------------------------------------

        run_metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": "final_model_training",
            "artifact_directory": (str(run_dir)),
            "model_artifact_path": (result.model_artifact_path),
            "mlflow_run_id": (result.mlflow_run_id),
            "feature_experiment_id": (result.feature_experiment_id),
            "feature_config_signature": (result.feature_config_signature),
            "model_experiment_id": (result.model_experiment_id),
            "model_config_signature": (result.model_config_signature),
            "tuning_config_signature": (result.tuning_config_signature),
            "validation_metric": (result.validation_metric),
            "validation_direction": (result.validation_direction),
            "validation_threshold": (result.validation_threshold),
            "validation_passed": (result.validation_passed),
            "production_approved": (result.validation_passed),
            "training_seconds": (result.training_seconds),
        }

        self._write_json(
            run_dir / "run_metadata.json",
            run_metadata,
        )

    # ==================================================================
    # JSON HELPER
    # ==================================================================

    @staticmethod
    def _write_json(
        path: Path,
        data: Any,
    ) -> None:

        path.write_text(
            json.dumps(
                data,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    # ==================================================================
    # MLFLOW
    # ==================================================================

    def _track_start(
        self,
        model_name: str,
        preferred_params: Dict[str, Any],
        feature_experiment_id: str,
        feature_config_signature: Optional[str],
        model_experiment_id: Optional[str],
        tuning_config_signature: Optional[str],
    ) -> None:

        tags = {
            "stage": "final_model_training",
            "model_name": model_name,
            "feature_experiment_id": (feature_experiment_id),
            "model_experiment_id": (model_experiment_id),
            "tuning_stage": "completed",
        }

        self.mlflow_tracker.log_tags(tags)

        params = {
            "model_name": model_name,
            "feature_experiment_id": (feature_experiment_id),
            "feature_config_signature": (feature_config_signature),
            "model_experiment_id": (model_experiment_id),
            "tuning_config_signature": (tuning_config_signature),
        }

        self.mlflow_tracker.log_params(params)

        if preferred_params:

            self.mlflow_tracker.log_params(preferred_params)

    # ==================================================================
    # MLFLOW COMPLETION
    # ==================================================================

    def _track_completion(
        self,
        result: SalaryFinalModelResult,
    ) -> None:

        self.mlflow_tracker.log_tags(
            {
                "status": ("success" if result.success else "failed"),
                "validation_passed": str(result.validation_passed),
                "production_approved": str(result.validation_passed),
            }
        )

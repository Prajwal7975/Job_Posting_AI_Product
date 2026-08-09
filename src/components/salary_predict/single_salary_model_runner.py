import json
import joblib
import numpy as np
import pandas as pd
from time import perf_counter
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from src.entity.single_salary_model_entity import (
    SalaryModelExperimentResult,
)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.logger import logging


class SalarySingleModelExperimentRunner:
    """
    Executes exactly ONE model-family baseline training experiment.
    Expects pre-engineered, fixed feature data. Returns a result object
    (success or failure) rather than raising exceptions, allowing the
    parent orchestrator to continue testing remaining models.
    """

    def __init__(
        self,
        model_factory,
        mlflow_tracker,
        base_artifact_dir: str = "artifacts/salary_model_experiments",
    ):
        self.factory = model_factory
        self.mlflow_tracker = mlflow_tracker
        self.base_artifact_dir = Path(base_artifact_dir)

    def run(
        self,
        config: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_experiment_id: str,
    ) -> SalaryModelExperimentResult:

        logging.info("=" * 60)
        logging.info("MODEL EXPERIMENT STARTED")
        logging.info("=" * 60)
        logging.info(f"Experiment ID : {config.model_experiment_id}")
        logging.info(f"Model         : {config.model_name}")
        logging.info(f"Features      : {feature_experiment_id}")

        # 1. Setup Versioned Artifact Directory (Stable Identity)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = (
            self.base_artifact_dir
            / f"{config.model_experiment_id}_{config.model_name}"
            / f"run_{timestamp}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the result object immediately
        result = SalaryModelExperimentResult(
            experiment_id=config.model_experiment_id,
            model_name=config.model_name,
            feature_experiment_id=feature_experiment_id,
            config_signature=getattr(config, "config_signature", "unknown"),
            model_params=dict(getattr(config, "model_params", {})),
            success=False,
            artifact_directory=str(run_dir),
        )

        run_name = f"{config.model_experiment_id}_{config.model_name}"

        try:
            # 2. Use MLflow Context Manager
            with self.mlflow_tracker.start_run(run_name=run_name) as active_run:
                result.mlflow_run_id = active_run.info.run_id

                try:
                    # -------------------------------------------------------------
                    # INNER EXECUTION BLOCK (Tracks failures inside MLflow run)
                    # -------------------------------------------------------------

                    # Log initial MLflow tags and parameters
                    tags = {
                        "stage": "model_family_comparison",
                        "model_experiment_id": config.model_experiment_id,
                        "model_family": config.model_name,
                        "feature_experiment_id": feature_experiment_id,
                        "config_signature": result.config_signature,
                    }
                    if hasattr(config, "config_version"):
                        tags["config_version"] = config.config_version

                    self.mlflow_tracker.log_tags(tags)

                    self.mlflow_tracker.log_params(
                        {
                            "model_experiment_id": config.model_experiment_id,
                            "model_name": config.model_name,
                            "feature_experiment_id": feature_experiment_id,
                        }
                    )

                    if result.model_params:
                        self.mlflow_tracker.log_params(result.model_params)

                    # Data Validation Guardrails
                    if X_train is None or y_train is None:
                        raise ValueError("Training data cannot be None.")
                    if X_test is None or y_test is None:
                        raise ValueError("Test data cannot be None.")
                    if len(X_train) != len(y_train):
                        raise ValueError(
                            f"Training feature/target length mismatch: {len(X_train)} != {len(y_train)}"
                        )
                    if len(X_test) != len(y_test):
                        raise ValueError(
                            f"Test feature/target length mismatch: {len(X_test)} != {len(y_test)}"
                        )

                    # 3. Build Estimator & Log Class Name
                    model = self.factory.build(config)
                    result.model_class_name = type(model).__name__
                    self.mlflow_tracker.log_tags(
                        {
                            "model_class": result.model_class_name,
                        }
                    )

                    # 4. Train Model
                    logging.info("Training started...")
                    start_time = perf_counter()
                    model.fit(X_train, y_train)
                    result.training_seconds = round(perf_counter() - start_time, 4)
                    logging.info(
                        f"Training completed. Training time : {result.training_seconds} seconds"
                    )

                    # 5. Predict
                    start_time = perf_counter()
                    predictions = model.predict(X_test)
                    result.prediction_seconds = round(perf_counter() - start_time, 4)

                    # 6. Evaluate (Explicit Float casting for JSON Safety & Version compat)
                    rmse = np.sqrt(mean_squared_error(y_test, predictions))
                    metrics = {
                        "MAE": float(mean_absolute_error(y_test, predictions)),
                        "RMSE": float(rmse),
                        "R2": float(r2_score(y_test, predictions)),
                    }
                    result.metrics = metrics

                    logging.info("Evaluation completed")
                    for m_name, m_val in metrics.items():
                        logging.info(f"{m_name} : {m_val:.4f}")

                    self.mlflow_tracker.log_metrics(
                        {
                            **metrics,
                            "training_time_seconds": result.training_seconds,
                            "prediction_time_seconds": result.prediction_seconds,
                        }
                    )

                    # 7. Local Artifact Serialization & Verification
                    model_path = run_dir / "model.joblib"
                    joblib.dump(model, model_path)

                    if not model_path.exists():
                        raise RuntimeError(
                            f"Model artifact was not created: {model_path}"
                        )

                    result.model_artifact_path = str(model_path)

                    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=4))

                    # Config serialization using single source of truth
                    model_config_dict = (
                        config.to_dict() if hasattr(config, "to_dict") else {}
                    )
                    model_config_dict["config_signature"] = result.config_signature
                    (run_dir / "model_config.json").write_text(
                        json.dumps(model_config_dict, indent=4)
                    )

                    feature_config_dict = {
                        "feature_experiment_id": feature_experiment_id
                    }
                    (run_dir / "feature_config.json").write_text(
                        json.dumps(feature_config_dict, indent=4)
                    )

                    run_metadata = {
                        "artifact_schema_version": "1.0",
                        "timestamp": result.generated_at,
                        "experiment_id": config.model_experiment_id,
                        "model_name": config.model_name,
                        "model_class_name": result.model_class_name,
                        "feature_experiment_id": feature_experiment_id,
                        "training_seconds": result.training_seconds,
                        "prediction_seconds": result.prediction_seconds,
                    }
                    (run_dir / "run_metadata.json").write_text(
                        json.dumps(run_metadata, indent=4)
                    )

                    logging.info(f"Model artifact saved at: {run_dir}")

                    # 8. MLflow Artifact Logging
                    self.mlflow_tracker.log_artifacts(
                        str(run_dir), artifact_folder="model_artifacts"
                    )

                    # Mark Success
                    result.success = True
                    self.mlflow_tracker.log_tags({"status": "success"})

                except Exception as inner_e:
                    # Capture failure state WITHOUT letting MLflow communication issues obscure root cause
                    result.success = False
                    result.error = str(inner_e)

                    logging.error(
                        "Model experiment '%s' failed during training/evaluation: %s",
                        config.model_experiment_id,
                        inner_e,
                        exc_info=True,
                    )

                    try:
                        self.mlflow_tracker.log_tags({"status": "failed"})
                    except Exception as mlflow_error:
                        logging.warning(
                            "Failed to log MLflow failure status for experiment '%s': %s",
                            config.model_experiment_id,
                            mlflow_error,
                        )

        except Exception as e:
            # Catches catastrophic tracking infrastructure failures (e.g., MLflow server down)
            result.success = False
            result.error = str(e)
            logging.error(
                "Model experiment '%s' failed at the tracking/orchestration level: %s",
                config.model_experiment_id,
                e,
                exc_info=True,
            )

        finally:
            logging.info("=" * 60)
            logging.info(f"MODEL EXPERIMENT COMPLETED - SUCCESS: {result.success}")
            logging.info("=" * 60)

        return result

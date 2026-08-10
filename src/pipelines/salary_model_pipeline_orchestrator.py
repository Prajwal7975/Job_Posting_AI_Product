"""
src/components/salary_predict/salary_model_pipeline_orchestrator.py

MASTER SALARY MODEL PIPELINE ORCHESTRATOR
=========================================

Complete production-oriented salary ML lifecycle:

    Common Feature Store
            |
            v
    1. Salary Feature Engineering
            |
            v
    2. Group-Aware Dataset Splitting
       ├── train
       ├── validation
       └── test
            |
            v
    3. Feature Experiments
       └── E0 ... EN
            |
            v
    4. Winning Feature Configuration
            |
            v
    5. Winning Feature Preprocessing
       └── fit ONLY on train
            |
            v
    6. Model Family Comparison
       └── M0 ... M5
       └── validation only
            |
            v
    7. Winning Model Family
            |
            v
    8. Hyperparameter Tuning
       └── train + validation only
       └── test untouched
            |
            v
    9. Final Model Training
       └── winning features
       └── preferred hyperparameters
       └── preprocessor + model
            |
            v
    10. Validation Quality Gate
            |
       PASS |
            v
    11. Final Holdout Test Evaluation
            |
            v
    12. Final Model Promotion
            |
            v
        model.joblib

IMPORTANT DATA LEAKAGE RULE
---------------------------

TEST DATA IS NEVER USED FOR:

    - feature selection
    - model-family selection
    - hyperparameter tuning
    - quality-gate decisions

TEST DATA is used exactly once:

    final candidate -> test evaluation

The final artifact contains:

    sklearn Pipeline(
        preprocessor,
        model
    )

Therefore inference becomes:

    model = joblib.load("model.joblib")
    prediction = model.predict(input_dataframe)

This class ONLY orchestrates existing components.

It does not implement:
    - feature engineering logic
    - preprocessing logic
    - model construction
    - model-family comparison logic
    - hyperparameter search
    - final training logic
    - MLflow internals
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.exception import CustomException
from src.logger import logging

# ======================================================================
# SALARY FEATURE ENGINEERING
# ======================================================================

from src.components.salary_predict.salary_feature_engineering import (
    SalaryFeatureEngineering,
)

from src.configs.salary_predict.salary_feature_engineering_config import (
    SalaryFeatureEngineeringConfig,
)

# ======================================================================
# DATASET SPLITTING
# ======================================================================

from src.components.salary_predict.salary_dataset_splitter import (
    SalaryDatasetSplitter,
)

from src.configs.salary_predict.salary_dataset_splitter_config import (
    SalaryDatasetSplitterConfig,
)

# ======================================================================
# FEATURE EXPERIMENTS
# ======================================================================

from src.components.salary_predict.salary_allFeatures_exp_runner import (
    SalaryFeatureExperimentRunner,
)

# ======================================================================
# PREPROCESSING
# ======================================================================

from src.components.salary_predict.salary_preprocessor_builder import (
    SalaryPreprocessorBuilder,
)

# ======================================================================
# MODEL FAMILY
# ======================================================================

from src.components.salary_predict.salary_model_factory import (
    SalaryModelFactory,
)

from src.components.salary_predict.single_salary_model_runner import (
    SalarySingleModelExperimentRunner,
)

from src.components.salary_predict.salary_model_family_runner import (
    SalaryModelFamilyExperimentRunner,
)

# ======================================================================
# HYPERPARAMETER TUNING
# ======================================================================

from src.components.salary_predict.salary_model_tuning_runner import (
    SalaryModelTuningRunner,
)

# ======================================================================
# FINAL MODEL
# ======================================================================

from src.components.salary_predict.salary_final_model_trainer import (
    SalaryFinalModelTrainer,
)

from src.configs.salary_predict.salary_final_model_trainer_config import (
    SalaryFinalModelConfig,
)

# ======================================================================
# MLFLOW
# ======================================================================

from src.components.salary_predict.salary_mlflow_tracker import (
    SalaryMLflowTracker,
)

# ======================================================================
# PIPELINE RESULT
# ======================================================================


@dataclass
class SalaryModelPipelineResult:
    """
    Complete result of one master salary-model pipeline execution.
    """

    success: bool

    # --------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------

    salary_dataset_path: Optional[str] = None

    train_dataset_path: Optional[str] = None
    validation_dataset_path: Optional[str] = None
    test_dataset_path: Optional[str] = None

    # --------------------------------------------------------------
    # Feature selection
    # --------------------------------------------------------------

    feature_experiment_id: Optional[str] = None
    feature_experiment_name: Optional[str] = None
    feature_config_signature: Optional[str] = None

    # --------------------------------------------------------------
    # Model selection
    # --------------------------------------------------------------

    model_experiment_id: Optional[str] = None
    model_name: Optional[str] = None
    model_config_signature: Optional[str] = None

    # --------------------------------------------------------------
    # Tuning
    # --------------------------------------------------------------

    tuning_config_signature: Optional[str] = None
    preferred_config_signature: Optional[str] = None
    preferred_params: Optional[Dict[str, Any]] = None
    tuning_improved_baseline: Optional[bool] = None

    # --------------------------------------------------------------
    # Final validation
    # --------------------------------------------------------------

    validation_metrics: Optional[Dict[str, float]] = None
    validation_metric: Optional[str] = None
    validation_passed: Optional[bool] = None

    # --------------------------------------------------------------
    # Final test
    # --------------------------------------------------------------

    test_metrics: Optional[Dict[str, float]] = None

    # --------------------------------------------------------------
    # Final artifact
    # --------------------------------------------------------------

    final_model_path: Optional[str] = None
    final_model_artifact_directory: Optional[str] = None

    # --------------------------------------------------------------
    # MLflow
    # --------------------------------------------------------------

    mlflow_run_id: Optional[str] = None

    # --------------------------------------------------------------
    # Timing
    # --------------------------------------------------------------

    stage_times: Optional[Dict[str, float]] = None
    total_execution_seconds: Optional[float] = None

    # --------------------------------------------------------------
    # Error
    # --------------------------------------------------------------

    error: Optional[str] = None


# ======================================================================
# MASTER ORCHESTRATOR
# ======================================================================


class SalaryModelPipelineOrchestrator:
    """
    Master orchestrator for the complete salary prediction lifecycle.

    The orchestrator owns ONLY execution order and stage boundaries.

    Existing components remain responsible for their own work.
    """

    def __init__(
        self,
        feature_engineering_config: Optional[SalaryFeatureEngineeringConfig] = None,
        splitter_config: Optional[SalaryDatasetSplitterConfig] = None,
        final_model_config: Optional[SalaryFinalModelConfig] = None,
        feature_runner: Optional[SalaryFeatureExperimentRunner] = None,
        model_family_runner: Optional[SalaryModelFamilyExperimentRunner] = None,
        tuning_runner: Optional[SalaryModelTuningRunner] = None,
        final_model_trainer: Optional[SalaryFinalModelTrainer] = None,
        mlflow_tracker: Optional[SalaryMLflowTracker] = None,
    ) -> None:

        # ============================================================
        # CONFIGURATION
        # ============================================================

        self.feature_engineering_config = (
            feature_engineering_config or SalaryFeatureEngineeringConfig()
        )

        self.splitter_config = splitter_config or SalaryDatasetSplitterConfig(
            base_artifacts_dir=(self.feature_engineering_config.base_artifacts_dir)
        )

        self.final_model_config = final_model_config or SalaryFinalModelConfig(
            artifact_dir="artifacts/salary_final_model",
            model_filename="model.joblib",
        )

        # ============================================================
        # SHARED MLFLOW TRACKER
        # ============================================================

        self.mlflow_tracker = mlflow_tracker or SalaryMLflowTracker()

        # ============================================================
        # FEATURE ENGINEERING
        # ============================================================

        self.feature_engineer = SalaryFeatureEngineering(
            config=self.feature_engineering_config
        )

        # ============================================================
        # DATASET SPLITTER
        # ============================================================

        self.splitter = SalaryDatasetSplitter(config=self.splitter_config)

        # ============================================================
        # SHARED MODEL FACTORY
        # ============================================================

        self.model_factory = SalaryModelFactory()

        # ============================================================
        # SHARED PREPROCESSOR BUILDER
        # ============================================================

        self.preprocessor_builder = SalaryPreprocessorBuilder()

        # ============================================================
        # FEATURE EXPERIMENT RUNNER
        #
        # Its SalaryTrainingRunner loads the train/validation
        # parquet files created by the splitter.
        # ============================================================

        self.feature_runner = feature_runner or SalaryFeatureExperimentRunner(
            training_runner=None,
            mlflow_tracker=self.mlflow_tracker,
            ranking_metric="annual_mae",
        )

        # ============================================================
        # SINGLE MODEL RUNNER
        # ============================================================

        self.single_model_runner = SalarySingleModelExperimentRunner(
            model_factory=self.model_factory,
            mlflow_tracker=self.mlflow_tracker,
        )

        # ============================================================
        # MODEL FAMILY COMPARISON
        # ============================================================

        self.model_family_runner = (
            model_family_runner
            or SalaryModelFamilyExperimentRunner(
                single_model_runner=(self.single_model_runner),
                mlflow_tracker=self.mlflow_tracker,
            )
        )

        # ============================================================
        # HYPERPARAMETER TUNING
        # ============================================================

        self.tuning_runner = tuning_runner or SalaryModelTuningRunner(
            model_factory=self.model_factory,
            mlflow_tracker=self.mlflow_tracker,
        )

        # ============================================================
        # FINAL MODEL TRAINER
        # ============================================================

        self.final_model_trainer = final_model_trainer or SalaryFinalModelTrainer(
            preprocessor_builder=(self.preprocessor_builder),
            model_factory=self.model_factory,
            mlflow_tracker=self.mlflow_tracker,
            config=self.final_model_config,
        )

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def run(
        self,
        force_rebuild: bool = False,
    ) -> SalaryModelPipelineResult:

        pipeline_start = perf_counter()

        stage_times: Dict[str, float] = {}

        logging.info("")
        logging.info("#" * 90)
        logging.info("SALARY MODEL MASTER PIPELINE STARTED")
        logging.info("#" * 90)

        try:

            # ==========================================================
            # STAGE 1
            # SALARY FEATURE ENGINEERING
            # ==========================================================

            start = perf_counter()

            logging.info("")
            logging.info("=" * 90)
            logging.info("STAGE 1 - SALARY FEATURE ENGINEERING")
            logging.info("=" * 90)

            salary_feature_result = self._run_salary_feature_engineering(
                force_rebuild=force_rebuild
            )

            salary_dataset_path = Path(
                salary_feature_result.salary_modeling_dataset_path
            )

            stage_times["salary_feature_engineering"] = round(
                perf_counter() - start,
                4,
            )

            self._assert_file_exists(
                salary_dataset_path,
                "Salary modeling dataset",
            )

            # ==========================================================
            # STAGE 2
            # DATASET SPLITTING
            # ==========================================================

            start = perf_counter()

            logging.info("")
            logging.info("=" * 90)
            logging.info("STAGE 2 - GROUP-AWARE DATASET SPLITTING")
            logging.info("=" * 90)

            split_result = self.splitter.initiate_dataset_splitting(
                force_rebuild=force_rebuild
            )

            self._assert_split_artifacts(split_result)

            stage_times["dataset_splitting"] = round(
                perf_counter() - start,
                4,
            )

            logging.info(
                "Split status: %s",
                getattr(
                    split_result,
                    "status",
                    "UNKNOWN",
                ),
            )

            # ==========================================================
            # LOAD ALL THREE SPLITS
            # ==========================================================

            train_df = pd.read_parquet(split_result.train_dataset_path)

            validation_df = pd.read_parquet(split_result.validation_dataset_path)

            test_df = pd.read_parquet(split_result.test_dataset_path)

            self._validate_split_frames(
                train_df=train_df,
                validation_df=validation_df,
                test_df=test_df,
            )

            logging.info(
                "TRAIN      : %s",
                train_df.shape,
            )

            logging.info(
                "VALIDATION : %s",
                validation_df.shape,
            )

            logging.info(
                "TEST       : %s",
                test_df.shape,
            )

            # ==========================================================
            # TARGETS
            # ==========================================================

            target_col = self.splitter_config.target_log_col

            self._validate_target(
                train_df,
                target_col,
                "train",
            )

            self._validate_target(
                validation_df,
                target_col,
                "validation",
            )

            self._validate_target(
                test_df,
                target_col,
                "test",
            )

            y_train = train_df[target_col].copy()

            y_validation = validation_df[target_col].copy()

            y_test = test_df[target_col].copy()

            # ==========================================================
            # STAGE 3
            # FEATURE EXPERIMENTS
            # ==========================================================

            start = perf_counter()

            logging.info("")
            logging.info("=" * 90)
            logging.info("STAGE 3 - FEATURE EXPERIMENTS")
            logging.info("=" * 90)

            feature_summary = self.feature_runner.run()

            if feature_summary is None:
                raise RuntimeError("Feature experiment runner returned None.")

            feature_config = feature_summary.winner_config

            feature_experiment_id = feature_summary.best_experiment_id

            feature_config_signature = getattr(
                feature_config,
                "config_signature",
                None,
            )

            logging.info(
                "Winning feature experiment : %s",
                feature_experiment_id,
            )

            logging.info(
                "Winning feature configuration : %s",
                feature_config.experiment_name,
            )

            stage_times["feature_experiments"] = round(
                perf_counter() - start,
                4,
            )

            # ==========================================================
            # STAGE 4
            # WINNING FEATURE DATA PREPARATION
            # ==========================================================

            start = perf_counter()

            logging.info("")
            logging.info("=" * 90)
            logging.info("STAGE 4 - WINNING FEATURE PREPARATION")
            logging.info("=" * 90)

            (
                X_train_raw,
                X_validation_raw,
                X_test_raw,
            ) = self._prepare_raw_winning_features(
                feature_config=feature_config,
                train_df=train_df,
                validation_df=validation_df,
                test_df=test_df,
            )

            # ----------------------------------------------------------
            # Fit preprocessing ONLY on TRAIN
            #
            # This transformed representation is used by:
            #
            #     model family comparison
            #     hyperparameter tuning
            #
            # It is NOT used by final model trainer.
            #
            # Final model trainer builds a fresh preprocessor and
            # fits it inside the final sklearn Pipeline.
            # ----------------------------------------------------------

            (
                X_train_transformed,
                X_validation_transformed,
                X_test_transformed,
            ) = self._prepare_transformed_winning_features(
                feature_config=feature_config,
                X_train_raw=X_train_raw,
                X_validation_raw=X_validation_raw,
                X_test_raw=X_test_raw,
            )

            stage_times["winning_feature_preparation"] = round(
                perf_counter() - start,
                4,
            )

            # ==========================================================
            # STAGE 5
            # MODEL FAMILY COMPARISON
            # ==========================================================

            start = perf_counter()

            logging.info("")
            logging.info("=" * 90)
            logging.info("STAGE 5 - MODEL FAMILY COMPARISON")
            logging.info("=" * 90)

            model_family_summary = self.model_family_runner.run_experiments(
                X_train=X_train_transformed,
                y_train=y_train,
                X_test=X_validation_transformed,
                y_test=y_validation,
                feature_experiment_id=(feature_experiment_id),
                feature_config_signature=(feature_config_signature),
                ranking_metric="RMSE",
            )

            logging.info(
                "Winning model family : %s",
                model_family_summary.winner_model_name,
            )

            logging.info(
                "Winning model ID : %s",
                model_family_summary.winner_experiment_id,
            )

            stage_times["model_family_comparison"] = round(
                perf_counter() - start,
                4,
            )

            # ==========================================================
            # STAGE 6
            # HYPERPARAMETER TUNING
            # ==========================================================

            start = perf_counter()

            logging.info("")
            logging.info("=" * 90)
            logging.info("STAGE 6 - HYPERPARAMETER TUNING")
            logging.info("=" * 90)

            tuning_summary = self.tuning_runner.run_tuning(
                model_family_summary=(model_family_summary),
                X_train=X_train_transformed,
                y_train=y_train,
                X_validation=(X_validation_transformed),
                y_validation=y_validation,
            )

            logging.info(
                "Selected model : %s",
                tuning_summary.model_name,
            )

            logging.info(
                "Preferred parameters : %s",
                tuning_summary.preferred_params,
            )

            logging.info(
                "Tuning improved baseline : %s",
                tuning_summary.tuning_improved_baseline,
            )

            stage_times["hyperparameter_tuning"] = round(
                perf_counter() - start,
                4,
            )

            # ==========================================================
            # IMPORTANT
            #
            # TEST DATA HAS NOT BEEN USED YET.
            #
            # The transformed test matrix above was prepared only
            # because the orchestration contract may require the
            # matrix, but we deliberately do NOT pass it into any
            # model-selection or tuning stage.
            #
            # We will use RAW test data for the final saved pipeline.
            # ==========================================================

            del X_train_transformed
            del X_validation_transformed
            del X_test_transformed

            # ==========================================================
            # STAGE 7
            # FINAL MODEL TRAINING
            # ==========================================================

            start = perf_counter()

            logging.info("")
            logging.info("=" * 90)
            logging.info("STAGE 7 - FINAL MODEL TRAINING")
            logging.info("=" * 90)

            final_result = self.final_model_trainer.run(
                X_train=X_train_raw,
                y_train=y_train,
                X_validation=X_validation_raw,
                y_validation=y_validation,
                feature_config=feature_config,
                model_family_summary=(model_family_summary),
                tuning_summary=tuning_summary,
            )

            stage_times["final_model_training"] = round(
                perf_counter() - start,
                4,
            )

            # ==========================================================
            # STAGE 8
            # VALIDATION QUALITY GATE
            # ==========================================================

            logging.info("")
            logging.info("=" * 90)
            logging.info("STAGE 8 - VALIDATION QUALITY GATE")
            logging.info("=" * 90)

            self._validate_final_training_result(final_result)

            validation_metrics = dict(final_result.validation_metrics)

            validation_passed = bool(final_result.validation_passed)

            logging.info(
                "Validation metrics: %s",
                validation_metrics,
            )

            logging.info(
                "Validation quality gate: %s",
                ("PASSED" if validation_passed else "FAILED"),
            )

            # ----------------------------------------------------------
            # CRITICAL:
            #
            # Do NOT evaluate test data if validation failed.
            # ----------------------------------------------------------

            if not validation_passed:

                logging.error(
                    "Validation quality gate FAILED. "
                    "Test evaluation and model promotion "
                    "will NOT occur."
                )

                return self._build_failed_result(
                    salary_dataset_path=salary_dataset_path,
                    split_result=split_result,
                    feature_summary=feature_summary,
                    model_family_summary=(model_family_summary),
                    tuning_summary=tuning_summary,
                    final_result=final_result,
                    stage_times=stage_times,
                    pipeline_start=pipeline_start,
                    error=("Final model failed the validation " "quality gate."),
                )

            # ==========================================================
            # STAGE 9
            # FINAL HOLDOUT TEST EVALUATION
            # ==========================================================

            start = perf_counter()

            logging.info("")
            logging.info("=" * 90)
            logging.info("STAGE 9 - FINAL HOLDOUT TEST EVALUATION")
            logging.info("=" * 90)

            final_model_path = Path(final_result.model_artifact_path)

            test_metrics = self._evaluate_final_model_on_test(
                model_path=final_model_path,
                X_test=X_test_raw,
                y_test=y_test,
            )

            stage_times["test_evaluation"] = round(
                perf_counter() - start,
                4,
            )

            logging.info("FINAL TEST METRICS")

            for metric, value in test_metrics.items():

                logging.info(
                    "%s : %.6f",
                    metric,
                    value,
                )

            # ==========================================================
            # STAGE 10
            # FINAL ARTIFACT VERIFICATION / PROMOTION
            # ==========================================================

            start = perf_counter()

            logging.info("")
            logging.info("=" * 90)
            logging.info("STAGE 10 - FINAL MODEL VERIFICATION")
            logging.info("=" * 90)

            promoted_model_path = self._promote_final_model(final_model_path)

            stage_times["model_promotion"] = round(
                perf_counter() - start,
                4,
            )

            # ==========================================================
            # PIPELINE SUMMARY
            # ==========================================================

            total_seconds = perf_counter() - pipeline_start

            result = SalaryModelPipelineResult(
                success=True,
                salary_dataset_path=str(salary_dataset_path.resolve()),
                train_dataset_path=str(Path(split_result.train_dataset_path).resolve()),
                validation_dataset_path=str(
                    Path(split_result.validation_dataset_path).resolve()
                ),
                test_dataset_path=str(Path(split_result.test_dataset_path).resolve()),
                feature_experiment_id=(feature_experiment_id),
                feature_experiment_name=(feature_config.experiment_name),
                feature_config_signature=(feature_config_signature),
                model_experiment_id=(model_family_summary.winner_experiment_id),
                model_name=(tuning_summary.model_name),
                model_config_signature=(
                    getattr(
                        model_family_summary,
                        "winner_config_signature",
                        None,
                    )
                ),
                tuning_config_signature=(
                    getattr(
                        tuning_summary,
                        "best_config_signature",
                        None,
                    )
                ),
                preferred_config_signature=(
                    getattr(
                        tuning_summary,
                        "preferred_config_signature",
                        None,
                    )
                ),
                preferred_params=dict(tuning_summary.preferred_params),
                tuning_improved_baseline=(tuning_summary.tuning_improved_baseline),
                validation_metrics=(validation_metrics),
                validation_metric=(final_result.validation_metric),
                validation_passed=(validation_passed),
                test_metrics=test_metrics,
                final_model_path=str(promoted_model_path.resolve()),
                final_model_artifact_directory=(final_result.artifact_directory),
                mlflow_run_id=(
                    getattr(
                        final_result,
                        "mlflow_run_id",
                        None,
                    )
                ),
                stage_times=stage_times,
                total_execution_seconds=round(
                    total_seconds,
                    4,
                ),
            )

            # ==========================================================
            # SAVE MASTER PIPELINE SUMMARY
            # ==========================================================

            self._save_pipeline_summary(result)

            # ==========================================================
            # FINAL LOG
            # ==========================================================

            logging.info("")
            logging.info("#" * 90)
            logging.info("SALARY MODEL MASTER PIPELINE COMPLETED")
            logging.info("#" * 90)

            logging.info(
                "Feature winner : %s",
                result.feature_experiment_id,
            )

            logging.info(
                "Model winner   : %s",
                result.model_name,
            )

            logging.info(
                "Validation     : %s",
                result.validation_metrics,
            )

            logging.info(
                "Test           : %s",
                result.test_metrics,
            )

            logging.info(
                "Final artifact : %s",
                result.final_model_path,
            )

            logging.info(
                "Total time     : %.4f seconds",
                result.total_execution_seconds,
            )

            logging.info("#" * 90)

            return result

        except CustomException:

            raise

        except Exception as e:

            logging.error(
                "Salary model master pipeline failed.",
                exc_info=True,
            )

            raise CustomException(
                e,
                sys,
            ) from e

    # ==================================================================
    # STAGE 1
    # ==================================================================

    def _run_salary_feature_engineering(
        self,
        force_rebuild: bool,
    ) -> Any:

        cfg = self.feature_engineering_config

        feature_store_path = cfg.feature_store_path

        if not feature_store_path.exists():

            raise FileNotFoundError(
                "Common feature store does not exist.\n"
                f"Expected path: {feature_store_path}\n"
                "Run the common data pipeline first."
            )

        salary_dataset_path = cfg.latest_dir / "salary_modeling_dataset.parquet"

        metadata_path = cfg.latest_dir / "salary_pipeline_state.json"

        # --------------------------------------------------------------
        # Reuse logic
        # --------------------------------------------------------------

        if not force_rebuild and salary_dataset_path.exists():

            current_fingerprint = self._calculate_file_fingerprint(feature_store_path)

            previous_fingerprint = self._load_previous_feature_fingerprint(
                metadata_path
            )

            if (
                previous_fingerprint is not None
                and current_fingerprint == previous_fingerprint
            ):

                logging.info("Salary Feature Engineering: REUSED")

                # The feature-engineering component has no "load
                # previous result" API. We therefore return a small
                # compatible object carrying the authoritative path.
                #
                # The actual dataset already exists and Stage 2
                # fingerprints it independently.

                return _SalaryFeatureEngineeringReuseResult(
                    salary_modeling_dataset_path=(salary_dataset_path)
                )

        # --------------------------------------------------------------
        # Execute
        # --------------------------------------------------------------

        logging.info("Salary Feature Engineering: EXECUTING")

        result = self.feature_engineer.initiate_salary_feature_engineering()

        if result is None:

            raise RuntimeError("SalaryFeatureEngineering returned None.")

        self._assert_file_exists(
            Path(result.salary_modeling_dataset_path),
            "Salary modeling dataset",
        )

        return result

    # ==================================================================
    # STAGE 4
    # RAW WINNING FEATURES
    # ==================================================================

    @staticmethod
    def _prepare_raw_winning_features(
        feature_config: Any,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:

        required_features = tuple(feature_config.active_predictor_features)

        # --------------------------------------------------------------
        # E0
        # --------------------------------------------------------------

        if not required_features:

            raise ValueError(
                "Winning feature configuration contains " "no predictor features."
            )

        # --------------------------------------------------------------
        # Validate all three schemas
        # --------------------------------------------------------------

        for column in required_features:

            if column not in train_df.columns:

                raise ValueError(f"Feature '{column}' missing from train dataset.")

            if column not in validation_df.columns:

                raise ValueError(f"Feature '{column}' missing from validation dataset.")

            if column not in test_df.columns:

                raise ValueError(f"Feature '{column}' missing from test dataset.")

        # --------------------------------------------------------------
        # RAW feature frames
        # --------------------------------------------------------------

        X_train = train_df.loc[
            :,
            list(required_features),
        ].copy()

        X_validation = validation_df.loc[
            :,
            list(required_features),
        ].copy()

        X_test = test_df.loc[
            :,
            list(required_features),
        ].copy()

        return (
            X_train,
            X_validation,
            X_test,
        )

    # ==================================================================
    # STAGE 4
    # TRANSFORMED WINNING FEATURES
    # ==================================================================

    def _prepare_transformed_winning_features(
        self,
        feature_config: Any,
        X_train_raw: pd.DataFrame,
        X_validation_raw: pd.DataFrame,
        X_test_raw: pd.DataFrame,
    ) -> Tuple[Any, Any, Any]:

        # --------------------------------------------------------------
        # Build UNFITTED preprocessor
        # --------------------------------------------------------------

        preprocessor = self.preprocessor_builder.build(feature_config)

        if preprocessor is None:

            raise ValueError(
                "Winning feature configuration unexpectedly "
                "produced no preprocessor."
            )

        # --------------------------------------------------------------
        # FIT ONLY ON TRAIN
        # --------------------------------------------------------------

        logging.info("Fitting winning-feature preprocessor on TRAIN only.")

        X_train_transformed = preprocessor.fit_transform(X_train_raw)

        # --------------------------------------------------------------
        # TRANSFORM VALIDATION
        # --------------------------------------------------------------

        X_validation_transformed = preprocessor.transform(X_validation_raw)

        # --------------------------------------------------------------
        # TEST TRANSFORMATION
        #
        # This does NOT mean test is being used for model selection.
        #
        # The fitted train preprocessor is merely capable of transforming
        # test data. We deliberately never pass this representation to
        # feature/model/tuning selection.
        # --------------------------------------------------------------

        X_test_transformed = preprocessor.transform(X_test_raw)

        logging.info("Winning-feature transformation completed.")

        logging.info(
            "Train transformed shape : %s",
            getattr(
                X_train_transformed,
                "shape",
                None,
            ),
        )

        logging.info(
            "Validation transformed shape : %s",
            getattr(
                X_validation_transformed,
                "shape",
                None,
            ),
        )

        logging.info(
            "Test transformed shape : %s",
            getattr(
                X_test_transformed,
                "shape",
                None,
            ),
        )

        return (
            X_train_transformed,
            X_validation_transformed,
            X_test_transformed,
        )

    # ==================================================================
    # TEST EVALUATION
    # ==================================================================

    @staticmethod
    def _evaluate_final_model_on_test(
        model_path: Path,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:

        if not model_path.exists():

            raise FileNotFoundError(f"Final model not found: {model_path}")

        logging.info(
            "Loading final production candidate: %s",
            model_path,
        )

        model = joblib.load(model_path)

        # --------------------------------------------------------------
        # Predict ONLY now
        # --------------------------------------------------------------

        predictions = model.predict(X_test)

        predictions = np.asarray(
            predictions,
            dtype="float64",
        ).reshape(-1)

        if len(predictions) != len(y_test):

            raise ValueError(
                "Test prediction length mismatch: "
                f"{len(predictions)} != {len(y_test)}"
            )

        if not np.isfinite(predictions).all():

            raise ValueError(
                "Final model generated non-finite " "predictions on the test set."
            )

        # --------------------------------------------------------------
        # Metrics
        # --------------------------------------------------------------

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions,
                )
            )
        )

        return {
            "MAE": float(
                mean_absolute_error(
                    y_test,
                    predictions,
                )
            ),
            "RMSE": rmse,
            "R2": float(
                r2_score(
                    y_test,
                    predictions,
                )
            ),
        }

    # ==================================================================
    # FINAL MODEL PROMOTION
    # ==================================================================

    @staticmethod
    def _promote_final_model(
        model_path: Path,
    ) -> Path:

        if not model_path.exists():

            raise FileNotFoundError(f"Cannot promote missing model: {model_path}")

        # --------------------------------------------------------------
        # Verify artifact is loadable
        # --------------------------------------------------------------

        logging.info("Verifying final joblib artifact.")

        pipeline = joblib.load(model_path)

        if not hasattr(
            pipeline,
            "predict",
        ):

            raise TypeError(
                "Final joblib artifact does not contain "
                "a prediction-capable estimator."
            )

        # --------------------------------------------------------------
        # Keep the canonical artifact name
        #
        # The final trainer already saved the complete:
        #
        #     preprocessor + model
        #
        # inside this file.
        # --------------------------------------------------------------

        canonical_path = model_path.parent / "model.joblib"

        if model_path != canonical_path:

            shutil.copy2(
                model_path,
                canonical_path,
            )

        if not canonical_path.exists():

            raise RuntimeError("Canonical model.joblib was not created.")

        # --------------------------------------------------------------
        # Final load verification
        # --------------------------------------------------------------

        verified = joblib.load(canonical_path)

        if not hasattr(
            verified,
            "predict",
        ):

            raise RuntimeError(
                "Canonical model.joblib failed " "post-save verification."
            )

        logging.info(
            "Final model verified successfully: %s",
            canonical_path,
        )

        return canonical_path

    # ==================================================================
    # VALIDATION RESULT
    # ==================================================================

    @staticmethod
    def _validate_final_training_result(
        result: Any,
    ) -> None:

        if result is None:

            raise RuntimeError("Final model trainer returned None.")

        if not result.success:

            raise RuntimeError("Final model trainer reported " "success=False.")

        if result.validation_metrics is None:

            raise RuntimeError("Final model trainer returned no " "validation metrics.")

        if result.validation_passed is None:

            raise RuntimeError(
                "Final model trainer did not return " "validation_passed."
            )

        if not result.model_artifact_path:

            raise RuntimeError(
                "Final model trainer did not return " "a model artifact path."
            )

        model_path = Path(result.model_artifact_path)

        if not model_path.exists():

            raise FileNotFoundError(
                f"Final model artifact does not exist: " f"{model_path}"
            )

    # ==================================================================
    # SPLIT VALIDATION
    # ==================================================================

    @staticmethod
    def _assert_split_artifacts(
        split_result: Any,
    ) -> None:

        paths = (
            split_result.train_dataset_path,
            split_result.validation_dataset_path,
            split_result.test_dataset_path,
        )

        missing = [Path(path) for path in paths if not Path(path).exists()]

        if missing:

            raise FileNotFoundError(
                "Dataset splitting reported completion "
                "but these artifacts are missing: "
                f"{missing}"
            )

    @staticmethod
    def _validate_split_frames(
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:

        for name, df in (
            ("train", train_df),
            ("validation", validation_df),
            ("test", test_df),
        ):

            if df.empty:

                raise ValueError(f"{name} dataset is empty.")

        if not (
            train_df.columns.equals(validation_df.columns)
            and train_df.columns.equals(test_df.columns)
        ):

            raise ValueError("Train, validation and test " "schemas are not identical.")

    # ==================================================================
    # TARGET VALIDATION
    # ==================================================================

    @staticmethod
    def _validate_target(
        df: pd.DataFrame,
        target_col: str,
        dataset_name: str,
    ) -> None:

        if target_col not in df.columns:

            raise ValueError(
                f"{dataset_name} dataset is missing " f"target column '{target_col}'."
            )

        values = df[target_col].to_numpy(dtype="float64")

        if np.isnan(values).any():

            raise ValueError(f"{dataset_name} target contains NaN.")

        if not np.isfinite(values).all():

            raise ValueError(f"{dataset_name} target contains " "non-finite values.")

    # ==================================================================
    # FILE HELPERS
    # ==================================================================

    @staticmethod
    def _assert_file_exists(
        path: Path,
        description: str,
    ) -> None:

        if not path.exists():

            raise FileNotFoundError(f"{description} does not exist: {path}")

    @staticmethod
    def _calculate_file_fingerprint(
        path: Path,
    ) -> str:

        import hashlib

        if not path.exists():

            raise FileNotFoundError(f"Cannot fingerprint missing file: {path}")

        digest = hashlib.sha256()

        with path.open("rb") as file:

            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):

                digest.update(chunk)

        return digest.hexdigest()

    @staticmethod
    def _load_previous_feature_fingerprint(
        metadata_path: Path,
    ) -> Optional[str]:

        if not metadata_path.exists():

            return None

        try:

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            return metadata.get("input_feature_store_fingerprint")

        except (
            json.JSONDecodeError,
            OSError,
        ):

            logging.warning(
                "Could not read salary feature " "pipeline state: %s",
                metadata_path,
            )

            return None

    # ==================================================================
    # FAILED RESULT
    # ==================================================================

    @staticmethod
    def _build_failed_result(
        salary_dataset_path: Path,
        split_result: Any,
        feature_summary: Any,
        model_family_summary: Any,
        tuning_summary: Any,
        final_result: Any,
        stage_times: Dict[str, float],
        pipeline_start: float,
        error: str,
    ) -> SalaryModelPipelineResult:

        return SalaryModelPipelineResult(
            success=False,
            salary_dataset_path=str(salary_dataset_path.resolve()),
            train_dataset_path=str(Path(split_result.train_dataset_path).resolve()),
            validation_dataset_path=str(
                Path(split_result.validation_dataset_path).resolve()
            ),
            test_dataset_path=str(Path(split_result.test_dataset_path).resolve()),
            feature_experiment_id=(feature_summary.best_experiment_id),
            feature_experiment_name=(feature_summary.winner_config.experiment_name),
            feature_config_signature=getattr(
                feature_summary.winner_config,
                "config_signature",
                None,
            ),
            model_experiment_id=(model_family_summary.winner_experiment_id),
            model_name=(tuning_summary.model_name),
            preferred_params=dict(tuning_summary.preferred_params),
            validation_metrics=(
                dict(final_result.validation_metrics)
                if final_result.validation_metrics
                else None
            ),
            validation_metric=(
                getattr(
                    final_result,
                    "validation_metric",
                    None,
                )
            ),
            validation_passed=False,
            final_model_path=(final_result.model_artifact_path),
            final_model_artifact_directory=(final_result.artifact_directory),
            stage_times=stage_times,
            total_execution_seconds=round(
                perf_counter() - pipeline_start,
                4,
            ),
            error=error,
        )

    # ==================================================================
    # PIPELINE SUMMARY
    # ==================================================================

    @staticmethod
    def _save_pipeline_summary(
        result: SalaryModelPipelineResult,
    ) -> None:

        if not result.final_model_artifact_directory:

            logging.warning(
                "No final artifact directory available; "
                "pipeline summary was not persisted."
            )

            return

        directory = Path(result.final_model_artifact_directory)

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_path = directory / "pipeline_summary.json"

        summary_path.write_text(
            json.dumps(
                asdict(result),
                indent=4,
                default=str,
            ),
            encoding="utf-8",
        )

        logging.info(
            "Master pipeline summary saved: %s",
            summary_path,
        )


# ======================================================================
# SMALL REUSE RESULT ADAPTER
# ======================================================================


@dataclass(frozen=True)
class _SalaryFeatureEngineeringReuseResult:
    """
    Minimal compatibility result used when deterministic salary feature
    engineering artifacts are reused.

    The downstream splitter only needs the authoritative modeling-dataset
    path.
    """

    salary_modeling_dataset_path: Path


# ======================================================================
# CONVENIENCE ENTRYPOINT
# ======================================================================


def run_salary_model_pipeline(
    force_rebuild: bool = False,
) -> SalaryModelPipelineResult:

    orchestrator = SalaryModelPipelineOrchestrator()

    return orchestrator.run(force_rebuild=force_rebuild)


# ======================================================================
# CLI
# ======================================================================


if __name__ == "__main__":

    result = run_salary_model_pipeline(force_rebuild=False)

    if result.success:

        logging.info("Salary model pipeline finished successfully.")

        logging.info(
            "Final model: %s",
            result.final_model_path,
        )

    else:

        logging.error(
            "Salary model pipeline finished without " "promotion: %s",
            result.error,
        )

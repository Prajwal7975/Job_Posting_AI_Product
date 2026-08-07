"""
src/components/salary_predict/salary_training_runner.py

Salary Training Runner.

Orchestrates exactly ONE salary experiment:

    SalaryExperimentConfig
            |
            v
    load train.parquet + validation.parquet
            |
            v
    validate targets + feature contract
            |
            v
    SalaryPreprocessorBuilder + SalaryModelFactory
            |
            v
    sklearn training workflow
            |
            v
    fit TRAIN only
            |
            v
    predict VALIDATION only
            |
            v
    log-space + annual-salary metrics
            |
            v
    SalaryTrainingResult

Responsibilities
----------------
- Load TRAIN and VALIDATION split artifacts.
- Resolve predictor columns from SalaryExperimentConfig.
- Build the configured preprocessor and estimator.
- Fit preprocessing + model strictly on TRAIN.
- Predict strictly on VALIDATION.
- Evaluate predictions in:
    1. log salary space
    2. annual salary space
- Return the fitted workflow and structured metrics.

Out of scope
------------
This component does NOT:
- create train/validation/test splits
- read test.parquet
- define TF-IDF/OHE/imputation logic
- construct model families directly
- perform model selection
- perform hyperparameter tuning
- log to MLflow
- save production model artifacts

test.parquet is intentionally never read here.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import CustomException

from src.configs.salary_predict.salary_experiment_config import (
    SalaryExperimentConfig,
)
from src.configs.salary_predict.salary_dataset_splitter_config import (
    SalaryDatasetSplitterConfig,
)

from src.components.salary_predict.salary_preprocessor_builder import (
    SalaryPreprocessorBuilder,
)
from src.components.salary_predict.salary_model_factory import (
    SalaryModelFactory,
)

# ======================================================================
# RESULT ENTITY
# ======================================================================


@dataclass
class SalaryTrainingResult:
    """
    Structured result produced after training and validating one
    salary experiment.
    """

    experiment_id: str
    experiment_name: str
    model_name: str
    config_signature: str

    train_row_count: int
    validation_row_count: int

    raw_feature_columns: Tuple[str, ...]
    raw_feature_count: int
    transformed_feature_count: Optional[int]

    training_seconds: float
    validation_prediction_seconds: float

    log_metrics: Dict[str, float]
    annual_metrics: Dict[str, float]

    # E0:
    #     fitted DummyRegressor
    #
    # E1+:
    #     fitted sklearn Pipeline(
    #         preprocessor,
    #         model,
    #     )
    fitted_workflow: Any


# ======================================================================
# TRAINING RUNNER
# ======================================================================


class SalaryTrainingRunner:
    """
    Train and validate exactly ONE SalaryExperimentConfig.

    The runner is deliberately orchestration-only.

    It does not:
    - decide which experiment wins
    - tune hyperparameters
    - use the test split
    - track experiments in MLflow
    - persist the final production model
    """

    def __init__(
        self,
        preprocessor_builder: Optional[SalaryPreprocessorBuilder] = None,
        model_factory: Optional[SalaryModelFactory] = None,
        train_path: Optional[Path] = None,
        validation_path: Optional[Path] = None,
        splitter_config: Optional[SalaryDatasetSplitterConfig] = None,
    ) -> None:

        self.preprocessor_builder = preprocessor_builder or SalaryPreprocessorBuilder()

        self.model_factory = model_factory or SalaryModelFactory()

        resolved_splitter_config = splitter_config or SalaryDatasetSplitterConfig()

        self.train_path = (
            Path(train_path)
            if train_path is not None
            else resolved_splitter_config.train_output_path
        )

        self.validation_path = (
            Path(validation_path)
            if validation_path is not None
            else resolved_splitter_config.validation_output_path
        )

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def run(
        self,
        config: SalaryExperimentConfig,
    ) -> SalaryTrainingResult:

        try:

            # ----------------------------------------------------------
            # Validate public boundary BEFORE reading config attributes.
            # ----------------------------------------------------------

            self._validate_config_type(config)

            logging.info("=" * 70)
            logging.info("SALARY EXPERIMENT TRAINING STARTED")
            logging.info("=" * 70)

            logging.info(f"Experiment ID   : {config.experiment_id}")
            logging.info(f"Experiment Name : {config.experiment_name}")
            logging.info(f"Model           : {config.model_name}")
            logging.info(f"Config Signature: {config.config_signature}")

            # ----------------------------------------------------------
            # Load TRAIN + VALIDATION only
            # ----------------------------------------------------------

            train_df, validation_df = self._load_training_data()

            # ----------------------------------------------------------
            # Validate target contracts
            # ----------------------------------------------------------

            self._validate_target_column(
                train_df,
                dataset_name="train",
                target_col=config.training_target_col,
                require_positive=False,
            )

            self._validate_target_column(
                validation_df,
                dataset_name="validation",
                target_col=config.training_target_col,
                require_positive=False,
            )

            # Annual target is required on validation because business
            # metrics are calculated in annual salary space.
            self._validate_target_column(
                validation_df,
                dataset_name="validation",
                target_col=config.annual_target_col,
                require_positive=True,
            )

            logging.info(f"Train rows      : {len(train_df):,}")
            logging.info(f"Validation rows : {len(validation_df):,}")

            # ----------------------------------------------------------
            # Construct UNFITTED components
            # ----------------------------------------------------------

            preprocessor = self.preprocessor_builder.build(config)

            model = self.model_factory.build(config)

            # ----------------------------------------------------------
            # Resolve X
            # ----------------------------------------------------------

            (
                X_train,
                X_validation,
                raw_features,
            ) = self._prepare_xy(
                config=config,
                train_df=train_df,
                validation_df=validation_df,
                preprocessor=preprocessor,
            )

            if raw_features:

                logging.info(
                    "Required raw features: %s",
                    list(raw_features),
                )

            else:

                logging.info("Required raw features: " "(none - dummy baseline)")

            # ----------------------------------------------------------
            # Resolve y
            # ----------------------------------------------------------

            y_train = train_df[config.training_target_col].to_numpy(dtype="float64")

            y_validation_log = validation_df[config.training_target_col].to_numpy(
                dtype="float64"
            )

            annual_actual = validation_df[config.annual_target_col].to_numpy(
                dtype="float64"
            )

            # ----------------------------------------------------------
            # Build training workflow
            # ----------------------------------------------------------

            workflow = self._build_training_workflow(
                preprocessor=preprocessor,
                model=model,
            )

            if preprocessor is None:

                logging.info("No preprocessor required " "(dummy baseline).")

            else:

                logging.info("Preprocessor constructed.")

            logging.info("Estimator constructed.")

            logging.info("Training workflow assembled.")

            # ----------------------------------------------------------
            # TRAIN
            # ----------------------------------------------------------

            logging.info("Training started...")

            (
                fitted_workflow,
                training_seconds,
            ) = self._fit_workflow(workflow=workflow, X_train=X_train, y_train=y_train)

            logging.info("Training completed in %.4f sec", training_seconds)

            # ----------------------------------------------------------
            # VALIDATION PREDICTION
            # ----------------------------------------------------------

            logging.info("Validation prediction started...")

            (
                predictions_log,
                validation_prediction_seconds,
            ) = self._predict_validation(
                fitted_workflow=fitted_workflow,
                X_validation=X_validation,
            )

            logging.info(
                "Validation prediction completed " "in %.4f sec",
                validation_prediction_seconds,
            )

            # ----------------------------------------------------------
            # Validate log-space predictions
            # ----------------------------------------------------------

            self._validate_predictions(
                predictions=predictions_log,
                expected_len=len(validation_df),
                prediction_space="log-space",
            )

            # ----------------------------------------------------------
            # LOG-SPACE METRICS
            # ----------------------------------------------------------

            log_metrics = self._compute_metrics(
                y_true=y_validation_log,
                y_pred=predictions_log,
                include_median_ape=False,
            )

            # ----------------------------------------------------------
            # Convert predictions back to annual salary
            #
            # Feature engineering creates:
            #
            # target_log_salary =
            #     np.log1p(target_annual_salary)
            #
            # therefore:
            #
            # target_annual_salary =
            #     np.expm1(target_log_salary)
            # ----------------------------------------------------------

            annual_predicted = self._inverse_log_target(predictions_log)

            # ----------------------------------------------------------
            # Validate annual predictions after expm1
            # ----------------------------------------------------------

            self._validate_predictions(
                predictions=annual_predicted,
                expected_len=len(validation_df),
                prediction_space="annual-space",
            )

            # ----------------------------------------------------------
            # ANNUAL-SALARY METRICS
            # ----------------------------------------------------------

            annual_metrics = self._compute_metrics(
                y_true=annual_actual,
                y_pred=annual_predicted,
                include_median_ape=True,
            )

            # ----------------------------------------------------------
            # Feature count
            # ----------------------------------------------------------

            transformed_feature_count = self._get_transformed_feature_count(
                fitted_workflow=fitted_workflow,
                preprocessor=preprocessor,
            )

            # ----------------------------------------------------------
            # Metric logging
            # ----------------------------------------------------------

            logging.info("-" * 70)
            logging.info("VALIDATION METRICS")
            logging.info("-" * 70)

            logging.info(
                "Log MAE       : %.6f",
                log_metrics["mae"],
            )

            logging.info(
                "Log RMSE      : %.6f",
                log_metrics["rmse"],
            )

            logging.info(
                "Log R2        : %.6f",
                log_metrics["r2"],
            )

            logging.info(
                "Annual MAE    : %.2f",
                annual_metrics["mae"],
            )

            logging.info(
                "Annual RMSE   : %.2f",
                annual_metrics["rmse"],
            )

            logging.info(
                "Annual R2     : %.6f",
                annual_metrics["r2"],
            )

            median_ape = annual_metrics.get("median_ape")

            if median_ape is not None and np.isfinite(median_ape):

                logging.info(
                    "Median APE    : %.2f%%",
                    median_ape,
                )

            if transformed_feature_count is not None:

                logging.info(
                    "Transformed Features: %d",
                    transformed_feature_count,
                )

            # ----------------------------------------------------------
            # Result
            # ----------------------------------------------------------

            result = SalaryTrainingResult(
                experiment_id=(config.experiment_id),
                experiment_name=(config.experiment_name),
                model_name=(config.model_name),
                config_signature=(config.config_signature),
                train_row_count=(len(train_df)),
                validation_row_count=(len(validation_df)),
                raw_feature_columns=(raw_features),
                raw_feature_count=(len(raw_features)),
                transformed_feature_count=(transformed_feature_count),
                training_seconds=round(
                    training_seconds,
                    4,
                ),
                validation_prediction_seconds=round(
                    validation_prediction_seconds,
                    4,
                ),
                log_metrics=(log_metrics),
                annual_metrics=(annual_metrics),
                fitted_workflow=(fitted_workflow),
            )

            logging.info("=" * 70)
            logging.info("SALARY EXPERIMENT TRAINING COMPLETED")
            logging.info("=" * 70)

            return result

        except CustomException:
            # Avoid wrapping an already-normalized project exception.
            raise

        except Exception as e:

            logging.exception("Salary experiment training failed.")

            raise CustomException(
                e,
                sys,
            ) from e

    # ==================================================================
    # CONFIG VALIDATION
    # ==================================================================

    @staticmethod
    def _validate_config_type(
        config: Any,
    ) -> None:

        if not isinstance(
            config,
            SalaryExperimentConfig,
        ):

            raise TypeError(
                "SalaryTrainingRunner.run() expects "
                "a SalaryExperimentConfig, "
                f"got {type(config).__name__}."
            )

    # ==================================================================
    # DATA LOADING
    # ==================================================================

    def _load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if not self.train_path.exists():

            raise FileNotFoundError("Train dataset not found at: " f"{self.train_path}")

        if not self.validation_path.exists():

            raise FileNotFoundError(
                "Validation dataset not found at: " f"{self.validation_path}"
            )

        logging.info(f"Train path      : {self.train_path}")

        logging.info(f"Validation path : {self.validation_path}")

        train_df = pd.read_parquet(self.train_path)

        validation_df = pd.read_parquet(self.validation_path)

        if train_df.empty:

            raise ValueError("Train dataset contains 0 rows.")

        if validation_df.empty:

            raise ValueError("Validation dataset contains 0 rows.")

        logging.info(f"Train shape      : {train_df.shape}")

        logging.info(f"Validation shape : {validation_df.shape}")

        return (
            train_df,
            validation_df,
        )

    # ==================================================================
    # TARGET VALIDATION
    # ==================================================================

    @staticmethod
    def _validate_target_column(
        df: pd.DataFrame,
        dataset_name: str,
        target_col: str,
        require_positive: bool,
    ) -> None:

        if target_col not in df.columns:

            raise ValueError(
                f"{dataset_name} dataset is missing "
                f"required target column "
                f"'{target_col}'."
            )

        if not pd.api.types.is_numeric_dtype(df[target_col]):

            raise ValueError(
                f"{dataset_name} target column " f"'{target_col}' must be numeric."
            )

        values = df[target_col].to_numpy(dtype="float64")

        if np.isnan(values).any():

            raise ValueError(
                f"{dataset_name} target column " f"'{target_col}' contains NaN values."
            )

        if not np.isfinite(values).all():

            raise ValueError(
                f"{dataset_name} target column "
                f"'{target_col}' contains "
                "non-finite values."
            )

        if require_positive and np.any(values <= 0):

            raise ValueError(
                f"{dataset_name} target column "
                f"'{target_col}' must contain "
                "strictly positive values."
            )

    # ==================================================================
    # FEATURE RESOLUTION
    # ==================================================================

    @staticmethod
    def _resolve_feature_columns(
        config: SalaryExperimentConfig,
    ) -> Tuple[str, ...]:

        return tuple(config.active_predictor_features)

    @staticmethod
    def _validate_feature_contract(
        config: SalaryExperimentConfig,
        raw_features: Tuple[str, ...],
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
    ) -> None:

        # --------------------------------------------------------------
        # Defense-in-depth target leakage protection
        # --------------------------------------------------------------

        protected_targets = {
            config.training_target_col,
            config.annual_target_col,
        }

        leaked = protected_targets & set(raw_features)

        if leaked:

            raise ValueError(
                "Target column(s) must never appear "
                "in predictor features. Leakage found: "
                f"{sorted(leaked)}"
            )

        # --------------------------------------------------------------
        # Required feature presence
        # --------------------------------------------------------------

        missing_train = [
            column for column in raw_features if column not in train_df.columns
        ]

        missing_validation = [
            column for column in raw_features if column not in validation_df.columns
        ]

        if missing_train:

            raise ValueError(
                "Train dataset is missing required " f"feature columns: {missing_train}"
            )

        if missing_validation:

            raise ValueError(
                "Validation dataset is missing required "
                f"feature columns: {missing_validation}"
            )

    # ==================================================================
    # X PREPARATION
    # ==================================================================

    def _prepare_xy(
        self,
        config: SalaryExperimentConfig,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        preprocessor: Optional[Any],
    ) -> Tuple[
        Any,
        Any,
        Tuple[str, ...],
    ]:

        # --------------------------------------------------------------
        # E0 dummy baseline
        # --------------------------------------------------------------

        if preprocessor is None:

            # DummyRegressor ignores predictor values, but sklearn
            # still expects a valid 2D X object.
            X_train = np.zeros(
                (len(train_df), 1),
                dtype="float64",
            )

            X_validation = np.zeros(
                (len(validation_df), 1),
                dtype="float64",
            )

            return (
                X_train,
                X_validation,
                (),
            )

        # --------------------------------------------------------------
        # Feature-based experiments
        # --------------------------------------------------------------

        raw_features = self._resolve_feature_columns(config)

        if not raw_features:

            raise ValueError(
                f"Experiment '{config.experiment_id}' "
                "has a preprocessor but no active "
                "predictor features."
            )

        self._validate_feature_contract(
            config=config,
            raw_features=raw_features,
            train_df=train_df,
            validation_df=validation_df,
        )

        X_train = train_df.loc[
            :,
            list(raw_features),
        ].copy()

        X_validation = validation_df.loc[
            :,
            list(raw_features),
        ].copy()

        return (
            X_train,
            X_validation,
            raw_features,
        )

    # ==================================================================
    # WORKFLOW
    # ==================================================================

    @staticmethod
    def _build_training_workflow(
        preprocessor: Optional[Any],
        model: Any,
    ) -> Any:

        if preprocessor is None:
            return model

        return Pipeline(
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

    # ==================================================================
    # FIT
    # ==================================================================

    @staticmethod
    def _fit_workflow(
        workflow: Any,
        X_train: Any,
        y_train: np.ndarray,
    ) -> Tuple[Any, float]:

        start = time.perf_counter()

        fitted_workflow = workflow.fit(
            X_train,
            y_train,
        )

        elapsed = time.perf_counter() - start

        return (
            fitted_workflow,
            elapsed,
        )

    # ==================================================================
    # VALIDATION PREDICTION
    # ==================================================================

    @staticmethod
    def _predict_validation(
        fitted_workflow: Any,
        X_validation: Any,
    ) -> Tuple[np.ndarray, float]:

        start = time.perf_counter()

        predictions = fitted_workflow.predict(X_validation)

        elapsed = time.perf_counter() - start

        predictions_array = np.asarray(
            predictions,
            dtype="float64",
        ).reshape(-1)

        return (
            predictions_array,
            elapsed,
        )

    # ==================================================================
    # PREDICTION VALIDATION
    # ==================================================================

    @staticmethod
    def _validate_predictions(
        predictions: np.ndarray,
        expected_len: int,
        prediction_space: str,
    ) -> None:

        if predictions.ndim != 1:

            raise ValueError(
                f"{prediction_space} predictions " "must be one-dimensional."
            )

        if len(predictions) != expected_len:

            raise ValueError(
                f"{prediction_space} prediction count "
                f"({len(predictions)}) does not match "
                f"expected validation row count "
                f"({expected_len})."
            )

        if not np.isfinite(predictions).all():

            raise ValueError(
                f"{prediction_space} predictions " "contain NaN or infinite values."
            )

    # ==================================================================
    # TARGET INVERSE TRANSFORM
    # ==================================================================

    @staticmethod
    def _inverse_log_target(
        log_values: np.ndarray,
    ) -> np.ndarray:

        # salary_feature_engineering.py defines:
        #
        # target_log_salary =
        #     np.log1p(target_annual_salary)
        #
        # therefore the mathematically correct inverse is expm1.

        with np.errstate(
            over="ignore",
            invalid="ignore",
        ):

            annual_values = np.expm1(log_values)

        return np.asarray(
            annual_values,
            dtype="float64",
        )

    # ==================================================================
    # METRICS
    # ==================================================================

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        include_median_ape: bool,
    ) -> Dict[str, float]:

        mae = float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        )

        r2 = float(
            r2_score(
                y_true,
                y_pred,
            )
        )

        metrics: Dict[str, float] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

        if include_median_ape:

            metrics["median_ape"] = self._median_absolute_percentage_error(
                y_true,
                y_pred,
            )

        return metrics

    # ==================================================================
    # MEDIAN ABSOLUTE PERCENTAGE ERROR
    # ==================================================================

    @staticmethod
    def _median_absolute_percentage_error(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:

        y_true = np.asarray(
            y_true,
            dtype="float64",
        )

        y_pred = np.asarray(
            y_pred,
            dtype="float64",
        )

        safe_mask = y_true != 0

        if not safe_mask.any():

            return float("nan")

        ape = (
            np.abs((y_true[safe_mask] - y_pred[safe_mask]) / y_true[safe_mask]) * 100.0
        )

        return float(np.median(ape))

    # ==================================================================
    # TRANSFORMED FEATURE COUNT
    # ==================================================================

    @staticmethod
    def _get_transformed_feature_count(
        fitted_workflow: Any,
        preprocessor: Optional[Any],
    ) -> Optional[int]:

        # E0 has no preprocessing feature space.
        if preprocessor is None:
            return None

        try:

            fitted_preprocessor = fitted_workflow.named_steps["preprocessor"]

            feature_names = fitted_preprocessor.get_feature_names_out()

            return int(len(feature_names))

        except Exception as e:

            # Feature count is diagnostic metadata only.
            # Failure to retrieve names must never invalidate
            # an otherwise successfully trained experiment.

            logging.warning("Could not determine transformed " f"feature count: {e}")

            return None

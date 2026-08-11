"""
src/orchestration/master_pipeline_orchestrator.py

MASTER PIPELINE ORCHESTRATOR
============================

Single entry point for the complete ML product training workflow.

Architecture
------------

                    MASTER ORCHESTRATOR
                            |
             +--------------+--------------+
             |                             |
             v                             v
       DATA PIPELINE                  ML PIPELINE
             |                             |
             |                       Salary ML
             |                       Pipeline
             |
             v
      Common Feature Store
             |
             +---------------------------> ML Pipeline
                                           |
                                           +-- Dataset Split
                                           +-- Feature Experiments
                                           +-- Model Family
                                           +-- Hyperparameter Tuning
                                           +-- Final Model Training
                                           +-- Validation Gate
                                           +-- Test Evaluation
                                           +-- model.joblib

Responsibilities
----------------

This class ONLY coordinates high-level pipelines.

It does NOT implement:

    - ingestion
    - validation
    - cleaning
    - feature engineering
    - fingerprinting
    - preprocessing
    - model selection
    - hyperparameter tuning
    - model training
    - MLflow internals
    - API serving

Those responsibilities remain inside their existing components.

The data pipeline owns its own state/fingerprint logic.

The salary ML pipeline owns its own model-development lifecycle.

The master orchestrator simply connects them.

Important
---------

Prediction requests MUST NOT call this orchestrator.

This orchestrator is for:

    training
    retraining
    rebuilding
    CI/CD
    scheduled model updates

The deployed API loads model.joblib directly.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional

from src.exception import CustomException
from src.logger import logging

# ======================================================================
# DATA PIPELINE
# ======================================================================

# IMPORTANT:
#
# Replace this import ONLY if the actual module containing your current
# state-aware data pipeline has a different path.
#
# The state-aware data pipeline is the implementation that uses:
#
#     PipelineStateManager
#     hash_file()
#     hash_directory()
#
# and returns PipelineResult with:
#
#     stage_status
#     pipeline_reused
#
from src.pipelines.data_pipeline_orchestrator import (
    run_pipeline as run_data_pipeline,
)

# ======================================================================
# SALARY ML PIPELINE
# ======================================================================

from src.pipelines.salary_model_pipeline_orchestrator import (
    SalaryModelPipelineOrchestrator,
    SalaryModelPipelineResult,
)

# ======================================================================
# RESULT ENTITY
# ======================================================================


@dataclass
class MasterPipelineResult:
    """
    Complete result of one master training execution.
    """

    success: bool

    # --------------------------------------------------------------
    # Data pipeline
    # --------------------------------------------------------------

    data_pipeline_executed: bool = False
    data_pipeline_reused: bool = False

    data_stage_status: Optional[Dict[str, str]] = None

    # --------------------------------------------------------------
    # ML pipeline
    # --------------------------------------------------------------

    ml_pipeline_executed: bool = False

    feature_experiment_id: Optional[str] = None
    feature_experiment_name: Optional[str] = None

    model_experiment_id: Optional[str] = None
    model_name: Optional[str] = None

    tuning_config_signature: Optional[str] = None
    preferred_params: Optional[Dict[str, Any]] = None

    validation_metrics: Optional[Dict[str, float]] = None
    validation_passed: Optional[bool] = None

    test_metrics: Optional[Dict[str, float]] = None

    # --------------------------------------------------------------
    # Final artifact
    # --------------------------------------------------------------

    final_model_path: Optional[str] = None
    final_model_artifact_directory: Optional[str] = None

    # --------------------------------------------------------------
    # Execution
    # --------------------------------------------------------------

    stage_times: Optional[Dict[str, float]] = None
    total_execution_seconds: float = 0.0

    # --------------------------------------------------------------
    # Failure
    # --------------------------------------------------------------

    error: Optional[str] = None


# ======================================================================
# MASTER ORCHESTRATOR
# ======================================================================


class MasterPipelineOrchestrator:
    """
    High-level orchestrator connecting the common data pipeline
    and salary ML pipeline.

    The data pipeline is authoritative for data freshness.

    The salary ML pipeline is authoritative for model development.

    This class does not duplicate either pipeline's internal logic.
    """

    def __init__(
        self,
        salary_ml_pipeline: Optional[SalaryModelPipelineOrchestrator] = None,
    ) -> None:

        self.salary_ml_pipeline = (
            salary_ml_pipeline or SalaryModelPipelineOrchestrator()
        )

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def run(
        self,
        force_data_pipeline: bool = False,
        force_ml_pipeline: bool = False,
    ) -> MasterPipelineResult:

        total_start = perf_counter()

        stage_times: Dict[str, float] = {}

        logging.info("")
        logging.info("#" * 90)
        logging.info("MASTER AI/ML PIPELINE STARTED")
        logging.info("#" * 90)

        try:

            # ==========================================================
            # 1. DATA PIPELINE
            # ==========================================================

            logging.info("")
            logging.info("=" * 90)
            logging.info("MASTER STAGE 1 - DATA PIPELINE")
            logging.info("=" * 90)

            data_start = perf_counter()

            data_result = self._run_data_pipeline(force=force_data_pipeline)

            data_time = perf_counter() - data_start

            stage_times["data_pipeline"] = round(
                data_time,
                4,
            )

            data_reused = bool(
                getattr(
                    data_result,
                    "pipeline_reused",
                    False,
                )
            )

            data_stage_status = dict(
                getattr(
                    data_result,
                    "stage_status",
                    {},
                )
            )

            logging.info(
                "Data pipeline reused: %s",
                data_reused,
            )

            logging.info(
                "Data stage status: %s",
                data_stage_status,
            )

            # ==========================================================
            # 2. DETERMINE WHETHER ML PIPELINE MUST RUN
            # ==========================================================

            #
            # Current rule:
            #
            #   force_ml_pipeline
            #       -> RUN
            #
            #   force_data_pipeline
            #       -> RUN
            #
            #   data pipeline actually rebuilt
            #       -> RUN
            #
            #   data pipeline completely reused
            #       -> SKIP
            #
            # This is deliberately conservative.
            #
            # Later, when the ML pipeline gets its own state manager,
            # this decision will additionally consider:
            #
            #   ML source-code fingerprint
            #   ML config fingerprint
            #   feature-config fingerprint
            #   model-config fingerprint
            #   tuning-config fingerprint
            #
            # so that:
            #
            #   DATA unchanged + MODEL config changed
            #
            # still triggers ML retraining.
            #

            ml_required = force_ml_pipeline or force_data_pipeline or not data_reused

            if not ml_required:

                logging.info("")
                logging.info("=" * 90)
                logging.info("MASTER DECISION - ML PIPELINE NOT REQUIRED")
                logging.info("=" * 90)

                logging.info("Data pipeline reused successfully.")

                logging.info("No ML pipeline execution requested.")

                total_time = perf_counter() - total_start

                result = MasterPipelineResult(
                    success=True,
                    data_pipeline_executed=True,
                    data_pipeline_reused=True,
                    data_stage_status=data_stage_status,
                    ml_pipeline_executed=False,
                    stage_times=stage_times,
                    total_execution_seconds=round(
                        total_time,
                        4,
                    ),
                )

                self._save_master_summary(result)

                logging.info("")
                logging.info("MASTER PIPELINE COMPLETED " "WITHOUT ML RETRAINING.")

                return result

            # ==========================================================
            # 3. SALARY ML PIPELINE
            # ==========================================================

            logging.info("")
            logging.info("=" * 90)
            logging.info("MASTER STAGE 2 - SALARY ML PIPELINE")
            logging.info("=" * 90)

            ml_start = perf_counter()

            ml_result = self._run_salary_ml_pipeline()

            ml_time = perf_counter() - ml_start

            stage_times["salary_ml_pipeline"] = round(
                ml_time,
                4,
            )

            # ==========================================================
            # 4. VALIDATE ML RESULT
            # ==========================================================

            self._validate_ml_result(ml_result)

            # ==========================================================
            # 5. BUILD MASTER RESULT
            # ==========================================================

            total_time = perf_counter() - total_start

            result = MasterPipelineResult(
                success=True,
                data_pipeline_executed=True,
                data_pipeline_reused=data_reused,
                data_stage_status=data_stage_status,
                ml_pipeline_executed=True,
                feature_experiment_id=(ml_result.feature_experiment_id),
                feature_experiment_name=(ml_result.feature_experiment_name),
                model_experiment_id=(ml_result.model_experiment_id),
                model_name=(ml_result.model_name),
                tuning_config_signature=(ml_result.tuning_config_signature),
                preferred_params=(
                    dict(ml_result.preferred_params)
                    if ml_result.preferred_params
                    else None
                ),
                validation_metrics=(
                    dict(ml_result.validation_metrics)
                    if ml_result.validation_metrics
                    else None
                ),
                validation_passed=(ml_result.validation_passed),
                test_metrics=(
                    dict(ml_result.test_metrics) if ml_result.test_metrics else None
                ),
                final_model_path=(ml_result.final_model_path),
                final_model_artifact_directory=(
                    ml_result.final_model_artifact_directory
                ),
                stage_times=stage_times,
                total_execution_seconds=round(
                    total_time,
                    4,
                ),
            )

            # ==========================================================
            # 6. SAVE MASTER SUMMARY
            # ==========================================================

            self._save_master_summary(result)

            # ==========================================================
            # FINAL LOGGING
            # ==========================================================

            logging.info("")
            logging.info("#" * 90)
            logging.info("MASTER AI/ML PIPELINE COMPLETED")
            logging.info("#" * 90)

            logging.info(
                "Data pipeline reused : %s",
                result.data_pipeline_reused,
            )

            logging.info(
                "ML pipeline executed : %s",
                result.ml_pipeline_executed,
            )

            logging.info(
                "Feature winner       : %s",
                result.feature_experiment_id,
            )

            logging.info(
                "Model winner         : %s",
                result.model_name,
            )

            logging.info(
                "Validation passed    : %s",
                result.validation_passed,
            )

            logging.info(
                "Validation metrics   : %s",
                result.validation_metrics,
            )

            logging.info(
                "Test metrics         : %s",
                result.test_metrics,
            )

            logging.info(
                "Final model          : %s",
                result.final_model_path,
            )

            logging.info(
                "Total execution time: %.4f sec",
                result.total_execution_seconds,
            )

            logging.info("#" * 90)

            return result

        except CustomException:

            raise

        except Exception as e:

            logging.error(
                "MASTER PIPELINE FAILED",
                exc_info=True,
            )

            failed_result = MasterPipelineResult(
                success=False,
                stage_times=stage_times,
                total_execution_seconds=round(
                    perf_counter() - total_start,
                    4,
                ),
                error=str(e),
            )

            self._save_master_summary(failed_result)

            raise CustomException(
                e,
                sys,
            ) from e

    # ==================================================================
    # DATA PIPELINE
    # ==================================================================

    @staticmethod
    def _run_data_pipeline(
        force: bool,
    ) -> Any:

        #
        # Your current state-aware data pipeline owns the fingerprint
        # and reuse logic.
        #
        # Do NOT calculate another data fingerprint here.
        #

        if force:

            logging.info("FORCE DATA PIPELINE EXECUTION REQUESTED.")

        return run_data_pipeline()

    # ==================================================================
    # SALARY ML PIPELINE
    # ==================================================================

    def _run_salary_ml_pipeline(self) -> SalaryModelPipelineResult:

        logging.info("Starting Salary ML Pipeline...")

        result = self.salary_ml_pipeline.run()

        return result

    # ==================================================================
    # ML RESULT VALIDATION
    # ==================================================================

    @staticmethod
    def _validate_ml_result(
        result: SalaryModelPipelineResult,
    ) -> None:

        if result is None:

            raise RuntimeError("Salary ML Pipeline returned None.")

        if not result.success:

            raise RuntimeError("Salary ML Pipeline completed with " "success=False.")

        # --------------------------------------------------------------
        # Validation quality gate
        # --------------------------------------------------------------

        if result.validation_passed is not True:

            raise RuntimeError(
                "Salary ML Pipeline did not pass " "the final validation quality gate."
            )

        # --------------------------------------------------------------
        # Test evaluation
        # --------------------------------------------------------------

        if not result.test_metrics:

            raise RuntimeError("Salary ML Pipeline returned no " "final test metrics.")

        # --------------------------------------------------------------
        # Final artifact
        # --------------------------------------------------------------

        if not result.final_model_path:

            raise RuntimeError(
                "Salary ML Pipeline did not return " "a final model artifact."
            )

        model_path = Path(result.final_model_path)

        if not model_path.exists():

            raise FileNotFoundError(
                "Salary ML Pipeline reported a model "
                "artifact that does not exist: "
                f"{model_path}"
            )

    # ==================================================================
    # SUMMARY
    # ==================================================================

    @staticmethod
    def _save_master_summary(
        result: MasterPipelineResult,
    ) -> None:

        summary_dir = Path("artifacts/master_pipeline")

        summary_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_path = summary_dir / "latest_run.json"

        with summary_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                asdict(result),
                file,
                indent=4,
                default=str,
            )

        logging.info(
            "Master pipeline summary saved: %s",
            summary_path,
        )


# ======================================================================
# CONVENIENCE FUNCTION
# ======================================================================


def run_master_pipeline(
    force_data_pipeline: bool = False,
    force_ml_pipeline: bool = False,
) -> MasterPipelineResult:

    orchestrator = MasterPipelineOrchestrator()

    return orchestrator.run(
        force_data_pipeline=force_data_pipeline,
        force_ml_pipeline=force_ml_pipeline,
    )


# ======================================================================
# CLI
# ======================================================================


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Run the complete AI/ML master pipeline."
    )

    parser.add_argument(
        "--force-data",
        action="store_true",
        help="Force execution of the data pipeline.",
    )

    parser.add_argument(
        "--force-ml",
        action="store_true",
        help="Force execution of the ML pipeline.",
    )

    args = parser.parse_args()

    result = run_master_pipeline(
        force_data_pipeline=args.force_data,
        force_ml_pipeline=args.force_ml,
    )

    if result.success:

        logging.info("")
        logging.info("MASTER PIPELINE SUCCESS.")

        if result.final_model_path:

            logging.info(
                "Production candidate: %s",
                result.final_model_path,
            )

    else:

        logging.error(
            "MASTER PIPELINE FAILED: %s",
            result.error,
        )
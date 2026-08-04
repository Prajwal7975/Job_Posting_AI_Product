"""
src/pipelines/salary_training_pipeline.py

Training pipeline for the Salary Prediction model.

Current stages
--------------
1. Salary Feature Engineering
2. Salary Dataset Splitting

Future stages
-------------
3. Train-only Preprocessing
4. Model Training
5. Model Evaluation
6. Model Registration

The pipeline supports reuse of an existing salary feature store when the
input common feature store has not changed. The dataset-splitting stage
manages its own reuse decision internally (see SalaryDatasetSplitter),
based on a fingerprint of its own input plus its split configuration.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional

from src.logger import logging
from src.exception import CustomException

from src.components.salary_predict.salary_feature_engineering import (
    SalaryFeatureEngineering,
)

from src.configs.salary_predict.salary_feature_engineering_config import (
    SalaryFeatureEngineeringConfig,
)

from src.entity.salary_predict.salary_feature_engineering_entity import (
    SalaryFeatureEngineeringResult,
)

from src.components.salary_predict.salary_dataset_splitter import (
    SalaryDatasetSplitter,
)

from src.configs.salary_predict.salary_dataset_splitter_config import (
    SalaryDatasetSplitterConfig,
)

from src.entity.salary_predict.salary_dataset_splitter_entity import (
    SalaryDatasetSplitResult,
)


# =====================================================================
# PIPELINE RESULT
# =====================================================================


@dataclass
class SalaryTrainingPipelineResult:

    salary_feature_engineering_result: Optional[
        SalaryFeatureEngineeringResult
    ]

    salary_feature_store_path: Path

    feature_engineering_executed: bool

    input_fingerprint: str

    dataset_split_result: Optional[
        SalaryDatasetSplitResult
    ]

    dataset_splitting_status: str

    stage_times: dict[str, float]


# =====================================================================
# FILE FINGERPRINT
# =====================================================================


def _calculate_file_fingerprint(
    file_path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Calculate SHA-256 fingerprint of a file.

    Used to determine whether the common feature store changed since the
    previous salary feature-engineering run.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Cannot calculate fingerprint. File not found: {file_path}"
        )

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:

        while True:

            chunk = file.read(chunk_size)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# =====================================================================
# CACHE METADATA
# =====================================================================


def _load_previous_input_fingerprint(
    metadata_path: Path,
) -> Optional[str]:

    if not metadata_path.exists():
        return None

    try:

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(file)

        return metadata.get(
            "input_feature_store_fingerprint"
        )

    except (
        json.JSONDecodeError,
        OSError,
    ) as exc:

        logging.warning(
            "Could not read previous salary pipeline metadata "
            "from %s: %s",
            metadata_path,
            exc,
        )

        return None


def _save_pipeline_metadata(
    metadata_path: Path,
    input_fingerprint: str,
    salary_dataset_path: Path,
) -> None:

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {

        "input_feature_store_fingerprint":
            input_fingerprint,

        "salary_modeling_dataset_path":
            str(salary_dataset_path),
    }

    temp_path = metadata_path.with_suffix(
        ".tmp"
    )

    temp_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    temp_path.replace(
        metadata_path
    )


# =====================================================================
# CACHE DECISION — FEATURE ENGINEERING
# =====================================================================


def _should_run_salary_feature_engineering(
    input_fingerprint: str,
    previous_fingerprint: Optional[str],
    salary_dataset_path: Path,
    force_rebuild: bool,
) -> bool:

    if force_rebuild:

        logging.info(
            "Force rebuild enabled. "
            "Salary Feature Engineering will run."
        )

        return True

    if not salary_dataset_path.exists():

        logging.info(
            "Salary modeling dataset does not exist. "
            "Feature Engineering will run."
        )

        return True

    if previous_fingerprint is None:

        logging.info(
            "No previous input fingerprint found. "
            "Feature Engineering will run."
        )

        return True

    if input_fingerprint != previous_fingerprint:

        logging.info(
            "Common feature store changed. "
            "Salary Feature Engineering will run."
        )

        return True

    logging.info(
        "Common feature store unchanged and salary "
        "modeling dataset already exists."
    )

    return False


# =====================================================================
# BOUNDARY INTEGRITY CHECK — DATASET SPLITTING
# =====================================================================


def _verify_split_artifacts_exist(
    dataset_split_result: SalaryDatasetSplitResult,
) -> None:
    """
    Stage 2 reporting EXECUTED or REUSED is a claim, not a guarantee at the
    pipeline boundary. This gives Stage 3 (and anyone reading this pipeline)
    a hard contract: if this function returns without raising, all three
    split files are physically present on disk right now.
    """

    split_paths = [
        dataset_split_result.train_dataset_path,
        dataset_split_result.validation_dataset_path,
        dataset_split_result.test_dataset_path,
    ]

    missing_paths = [
        path
        for path in split_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Dataset splitting completed/reused, but expected "
            f"split artifacts are missing: {missing_paths}"
        )


# =====================================================================
# PIPELINE
# =====================================================================


def run_salary_training_pipeline(
    force_rebuild: bool = False,
) -> SalaryTrainingPipelineResult:

    pipeline_start = perf_counter()

    try:

        logging.info("#" * 70)
        logging.info("SALARY TRAINING PIPELINE STARTED")
        logging.info("#" * 70)

        # -------------------------------------------------------------
        # Configuration
        # -------------------------------------------------------------

        feature_config = (
            SalaryFeatureEngineeringConfig()
        )

        feature_store_path = (
            feature_config.feature_store_path
        )

        salary_dataset_path = (
            feature_config.latest_dir
            / "salary_modeling_dataset.parquet"
        )

        pipeline_metadata_path = (
            feature_config.latest_dir
            / "salary_pipeline_state.json"
        )

        splitter_config = (
            SalaryDatasetSplitterConfig(
                base_artifacts_dir=
                    feature_config.base_artifacts_dir
            )
        )

        # -------------------------------------------------------------
        # Validate common feature store
        # -------------------------------------------------------------

        if not feature_store_path.exists():

            raise FileNotFoundError(
                "Common feature store does not exist. "
                "Run the common data pipeline first.\n"
                f"Expected path: {feature_store_path}"
            )

        # -------------------------------------------------------------
        # Fingerprint current feature store
        # -------------------------------------------------------------

        logging.info(
            "Calculating common feature-store fingerprint..."
        )

        input_fingerprint = (
            _calculate_file_fingerprint(
                feature_store_path
            )
        )

        logging.info(
            "Current feature-store fingerprint: %s...",
            input_fingerprint[:16],
        )

        # -------------------------------------------------------------
        # Load previous pipeline state
        # -------------------------------------------------------------

        previous_fingerprint = (
            _load_previous_input_fingerprint(
                pipeline_metadata_path
            )
        )

        # -------------------------------------------------------------
        # Decide whether Salary Feature Engineering must run
        # -------------------------------------------------------------

        should_run_feature_engineering = (
            _should_run_salary_feature_engineering(

                input_fingerprint=
                    input_fingerprint,

                previous_fingerprint=
                    previous_fingerprint,

                salary_dataset_path=
                    salary_dataset_path,

                force_rebuild=
                    force_rebuild,
            )
        )

        feature_engineering_result = None
        feature_engineering_time = 0.0

        # =============================================================
        # STAGE 1 — SALARY FEATURE ENGINEERING
        # =============================================================

        if should_run_feature_engineering:

            logging.info(
                "Starting Salary Feature Engineering..."
            )

            stage_start = perf_counter()

            feature_engineer = (
                SalaryFeatureEngineering(
                    config=feature_config
                )
            )

            feature_engineering_result = (
                feature_engineer
                .initiate_salary_feature_engineering()
            )

            feature_engineering_time = (
                perf_counter()
                - stage_start
            )

            salary_dataset_path = (
                feature_engineering_result
                .salary_modeling_dataset_path
            )

            # ---------------------------------------------------------
            # Save fingerprint only AFTER successful FE completion
            # ---------------------------------------------------------

            _save_pipeline_metadata(

                metadata_path=
                    pipeline_metadata_path,

                input_fingerprint=
                    input_fingerprint,

                salary_dataset_path=
                    salary_dataset_path,
            )

            logging.info(
                "Salary Feature Engineering completed "
                "in %.2f sec.",
                feature_engineering_time,
            )

        else:

            logging.info(
                "Skipping Salary Feature Engineering."
            )

            logging.info(
                "Reusing existing salary dataset: %s",
                salary_dataset_path,
            )

        # =============================================================
        # STAGE 2 — SALARY DATASET SPLITTING
        # =============================================================
        #
        # The splitter is always invoked — it independently fingerprints
        # its own input (the salary modeling dataset produced above,
        # whether freshly built or reused) together with its split
        # configuration, and decides EXECUTED vs REUSED on its own. This
        # means Stage 2 correctly reruns whenever Stage 1 produced a new
        # dataset, even if Stage 1 itself was skipped as unchanged from
        # a *previous* run but a rebuild was forced further downstream.

        logging.info(
            "Starting Salary Dataset Splitting..."
        )

        stage_start = perf_counter()

        splitter = (
            SalaryDatasetSplitter(
                config=splitter_config
            )
        )

        dataset_split_result = (
            splitter
            .initiate_dataset_splitting(
                force_rebuild=
                    force_rebuild,
            )
        )

        dataset_splitting_time = (
            perf_counter()
            - stage_start
        )

        logging.info(
            "Salary Dataset Splitting %s in %.2f sec.",
            dataset_split_result.status,
            dataset_splitting_time,
        )

        _verify_split_artifacts_exist(
            dataset_split_result
        )

        # =============================================================
        # FUTURE STAGES
        # =============================================================

        # -------------------------------------------------------------
        # Stage 3 — Train-only preprocessing
        # -------------------------------------------------------------
        #
        # preprocessor = SalaryPreprocessor(...)
        #
        # TF-IDF is FIT ONLY on training data here.
        # Categorical encoders are FIT ONLY on training data here.
        # Imputers are FIT ONLY on training data here.
        #

        # -------------------------------------------------------------
        # Stage 4 — Model Training
        # -------------------------------------------------------------
        #
        # Ridge
        # Random Forest
        # Gradient Boosting / XGBoost etc.
        #

        # -------------------------------------------------------------
        # Stage 5 — Evaluation
        # -------------------------------------------------------------

        total_time = (
            perf_counter()
            - pipeline_start
        )

        logging.info("#" * 70)
        logging.info(
            "SALARY TRAINING PIPELINE COMPLETED"
        )
        logging.info("#" * 70)

        return SalaryTrainingPipelineResult(

            salary_feature_engineering_result=
                feature_engineering_result,

            salary_feature_store_path=
                salary_dataset_path,

            feature_engineering_executed=
                should_run_feature_engineering,

            input_fingerprint=
                input_fingerprint,

            dataset_split_result=
                dataset_split_result,

            dataset_splitting_status=
                dataset_split_result.status,

            stage_times={

                "salary_feature_engineering_sec":
                    round(
                        feature_engineering_time,
                        2,
                    ),

                "salary_dataset_splitting_sec":
                    round(
                        dataset_splitting_time,
                        2,
                    ),

                "total_sec":
                    round(
                        total_time,
                        2,
                    ),
            },
        )

    except CustomException:
        raise

    except Exception as exc:

        logging.exception(
            "Salary Training Pipeline failed."
        )

        raise CustomException(
            exc,
            sys,
        ) from exc


# =====================================================================
# CLI ENTRYPOINT
# =====================================================================


if __name__ == "__main__":

    result = (
        run_salary_training_pipeline()
    )

    fe_status = (
        "EXECUTED"
        if result.feature_engineering_executed
        else "REUSED"
    )

    banner = f"""
======================================================================
SALARY TRAINING PIPELINE SUMMARY
======================================================================
Salary Feature Engineering : {fe_status}
Salary Dataset Splitting   : {result.dataset_splitting_status}

Salary Dataset:
{result.salary_feature_store_path}

Split Artifacts:
train      -> {result.dataset_split_result.train_dataset_path}
validation -> {result.dataset_split_result.validation_dataset_path}
test       -> {result.dataset_split_result.test_dataset_path}

Input Feature Store Fingerprint:
{result.input_fingerprint[:16]}...

Feature Engineering Time : {result.stage_times['salary_feature_engineering_sec']} sec
Dataset Splitting Time    : {result.stage_times['salary_dataset_splitting_sec']} sec
Total Pipeline Time       : {result.stage_times['total_sec']} sec
======================================================================
"""

    print(banner)

    logging.info(banner)
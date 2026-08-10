"""
State-aware Data Pipeline Orchestrator.

Execution strategy
------------------
The orchestrator uses PipelineStateManager to determine whether each
stage can be reused from a previous successful execution.

The individual components remain responsible ONLY for processing.
They do not know about fingerprints, caching, or pipeline state.

A stage is reusable when:

    1. Its previous execution succeeded.
    2. Its input fingerprint has not changed.
    3. Its configuration fingerprint has not changed.
    4. Its persisted cached result still exists.

"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

import joblib
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.schema_alignment import SchemaAlignment
from src.components.data_validation import (
    DataValidation,
    DataValidationResult,
)
from src.components.data_cleaning import (
    DataCleaning,
    DataCleaningResult,
)
from src.components.data_consolidation import (
    MasterDatasetBuilder,
    DatasetBuildResult,
)
from src.components.common_feature_engineering import (
    CommonFeatureEngineering,
)
from src.configs.common_feature_engineering_config import (
    CommonFeatureEngineeringConfig,
)
from src.entity.common_feature_engineering_entity import (
    FeatureEngineeringResult,
)
from src.configs.pipeline_config import PipelineConfig
from src.exception import CustomException
from src.logger import logging

from src.utils.pipeline_state_manager import (
    PipelineStateManager,
)
from src.utils.fingerprint_utils import (
    hash_directory,
    hash_file,
)


# ============================================================================
# Configuration
# ============================================================================

pipeline_config = PipelineConfig()

CACHE_DIR = Path("artifacts/pipeline_cache")

STAGE_NAMES = (
    "data_ingestion",
    "schema_alignment",
    "data_validation",
    "data_cleaning",
    "master_dataset",
    "common_feature_engineering",
)


# ============================================================================
# Pipeline Result
# ============================================================================


@dataclass
class PipelineResult:
    """
    Complete result of the common data pipeline.

    The result contains the actual stage outputs plus execution metadata.
    """

    cleaning_result: DataCleaningResult
    validation_result: DataValidationResult
    dataset_build_result: DatasetBuildResult
    feature_engineering_result: FeatureEngineeringResult

    stage_times: Dict[str, float]

    stage_status: Dict[str, str]

    pipeline_reused: bool = False


# ============================================================================
# Utility Functions
# ============================================================================


def _get_cleaned_memory_usage_mb(
    data: Dict[str, Dict[str, pd.DataFrame]],
) -> float:
    """Calculate memory usage of cleaned relational DataFrames."""

    total_bytes = sum(
        df.memory_usage(deep=True).sum()
        for version in data.values()
        for df in version.values()
    )

    return round(
        total_bytes / (1024 * 1024),
        2,
    )


def _get_consolidated_memory_usage_mb(
    consolidated_data: Dict[str, pd.DataFrame],
) -> float:
    """Calculate memory usage of consolidated DataFrames."""

    total_bytes = sum(
        df.memory_usage(deep=True).sum()
        for df in consolidated_data.values()
    )

    return round(
        total_bytes / (1024 * 1024),
        2,
    )


def _get_feature_store_memory_mb(
    df: pd.DataFrame,
) -> float:
    """Calculate memory usage of the feature store."""

    return round(
        df.memory_usage(deep=True).sum()
        / (1024 * 1024),
        2,
    )


def _save_json_artifact(
    file_path: Path,
    data: Any,
) -> None:
    """Safely write dictionaries or dataclass objects to JSON."""

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if is_dataclass(data) and not isinstance(data, type):
        data = asdict(data)

    with open(
        file_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
            default=str,
        )

    logging.info(
        "Saved artifact: %s",
        file_path,
    )


# ============================================================================
# Cache Helpers
# ============================================================================


def _cache_path(stage_name: str) -> Path:
    """
    Return persistent cache location for a pipeline stage.
    """

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return CACHE_DIR / f"{stage_name}.joblib"


def _save_stage_cache(
    stage_name: str,
    result: Any,
) -> Path:
    """
    Persist a stage result so it can be reused by a future process.
    """

    path = _cache_path(stage_name)

    temporary_path = path.with_suffix(".tmp")

    joblib.dump(
        result,
        temporary_path,
    )

    temporary_path.replace(path)

    logging.info(
        "Saved stage cache: %s",
        path,
    )

    return path


def _load_stage_cache(
    stage_name: str,
) -> Any:
    """
    Load a previously persisted stage result.
    """

    path = _cache_path(stage_name)

    if not path.exists():
        raise FileNotFoundError(
            f"Stage cache does not exist: {path}"
        )

    logging.info(
        "Loading cached stage result: %s",
        path,
    )

    return joblib.load(path)


# ============================================================================
# Configuration Fingerprinting
# ============================================================================


def _configuration_fingerprint(
    state_manager: PipelineStateManager,
    config: Any,
) -> str:
    """
    Create a deterministic fingerprint for a component configuration.

    Supports:
        dataclasses
        objects with to_dict()
        normal Python objects with __dict__
        None
    """

    if config is None:
        payload: Any = None

    elif hasattr(config, "config_signature"):
        return str(config.config_signature)

    elif hasattr(config, "to_dict"):
        payload = config.to_dict()

    elif is_dataclass(config):
        payload = asdict(config)
        
    elif hasattr(config, "__dict__"):
        payload = {key: value for key, value in vars(config).items() if not key.startswith("_") }

    else:
        payload = str(config)

    return state_manager.fingerprint(payload)


def _component_config(
    component: Any,
) -> Any:
    """
    Extract a component's configuration without forcing every component
    to expose a particular interface.

    Components that have no public config simply produce None.
    """

    return getattr(
        component,
        "config",
        None,
    )


# ============================================================================
# Reuse Decision
# ============================================================================


def _can_reuse_stage(
    state_manager: PipelineStateManager,
    stage_name: str,
    input_fingerprint: str,
    config_fingerprint: str,
) -> bool:
    """
    Determine whether a stage can safely be reused.

    PipelineStateManager is responsible for:
        - previous success
        - input fingerprint
        - configuration fingerprint
        - artifact existence
    """

    return state_manager.is_stage_reusable(
        stage_name=stage_name,
        input_fingerprint=input_fingerprint,
        config_fingerprint=config_fingerprint,
        verify_artifacts=True,
    )

# ============================================================================
# Main Pipeline
# ============================================================================


def run_pipeline() -> PipelineResult:

    total_pipeline_start = perf_counter()

    state_manager = PipelineStateManager(pipeline_name="data_pipeline")

    stage_times: Dict[str, float] = {}

    stage_status: Dict[str, str] = {}

    current_stage: str | None = None

    try:

        logging.info("=" * 75)
        logging.info("STATE-AWARE DATA PIPELINE STARTED")
        logging.info("=" * 75)

        # ==================================================================
        # 1. DATA INGESTION
        # ==================================================================

        ingestion = DataIngestion()

        raw_data_dir = getattr(
            ingestion.config,
            "raw_data_dir",
            None,
        )

        if raw_data_dir is None:
            raise AttributeError(
                "DataIngestion config must expose 'raw_data_dir' "
                "for pipeline fingerprinting."
            )

        raw_input_fingerprint = hash_directory(
            raw_data_dir
        )

        ingestion_config_fingerprint = _configuration_fingerprint(
            state_manager,
            _component_config(ingestion),
        )

        if _can_reuse_stage(
            state_manager,
            "data_ingestion",
            raw_input_fingerprint,
            ingestion_config_fingerprint,
        ):

            logging.info(
                "[REUSE] Data Ingestion"
            )

            start_time = perf_counter()

            raw_datasets = _load_stage_cache(
                "data_ingestion"
            )

            stage_times["ingestion_sec"] = round(
                perf_counter() - start_time,
                2,
            )

            stage_status["data_ingestion"] = "reused"

        else:

            logging.info(
                "[RUN] Data Ingestion"
            )

            start_time = perf_counter()
            current_stage = "data_ingestion"

            raw_datasets = ingestion.initiate_data_ingestion()

            elapsed = perf_counter() - start_time

            stage_times["ingestion_sec"] = round(
                elapsed,
                2,
            )

            cache_path = _save_stage_cache(
                "data_ingestion",
                raw_datasets,
            )

            output_fingerprint = hash_file(
                cache_path
            )

            state_manager.record_stage_success(
                stage_name="data_ingestion",
                input_fingerprint=raw_input_fingerprint,
                config_fingerprint=ingestion_config_fingerprint,
                output_fingerprint=output_fingerprint,
                output_artifacts={
                    "cache": str(cache_path),
                },
            )

            stage_status["data_ingestion"] = "executed"

        # ==================================================================
        # 2. SCHEMA ALIGNMENT
        # ==================================================================

        aligner = SchemaAlignment()

        ingestion_output_fingerprint = (
            state_manager.get_output_fingerprint(
                "data_ingestion"
            )
        )

        if ingestion_output_fingerprint is None:
            raise RuntimeError(
                "Data ingestion output fingerprint is unavailable."
            )

        alignment_config_fingerprint = _configuration_fingerprint(
            state_manager,
            _component_config(aligner),
        )

        if _can_reuse_stage(
            state_manager,
            "schema_alignment",
            ingestion_output_fingerprint,
            alignment_config_fingerprint,
        ):

            logging.info(
                "[REUSE] Schema Alignment"
            )

            start_time = perf_counter()

            alignment_result = _load_stage_cache(
                "schema_alignment"
            )

            stage_times["alignment_sec"] = round(
                perf_counter() - start_time,
                2,
            )

            stage_status["schema_alignment"] = "reused"

        else:

            logging.info(
                "[RUN] Schema Alignment"
            )

            current_stage = "schema_alignment"
            start_time = perf_counter()

            alignment_result = (
                aligner.initiate_schema_alignment(
                    raw_datasets
                )
            )

            elapsed = perf_counter() - start_time

            stage_times["alignment_sec"] = round(
                elapsed,
                2,
            )

            cache_path = _save_stage_cache(
                "schema_alignment",
                alignment_result,
            )

            output_fingerprint = hash_file(
                cache_path
            )

            state_manager.record_stage_success(
                stage_name="schema_alignment",
                input_fingerprint=ingestion_output_fingerprint,
                config_fingerprint=alignment_config_fingerprint,
                output_fingerprint=output_fingerprint,
                output_artifacts={
                    "cache": str(cache_path),
                },
            )

            stage_status["schema_alignment"] = "executed"

        # ==================================================================
        # 3. DATA VALIDATION
        # ==================================================================

        validator = DataValidation()

        alignment_output_fingerprint = (
            state_manager.get_output_fingerprint(
                "schema_alignment"
            )
        )

        if alignment_output_fingerprint is None:
            raise RuntimeError(
                "Schema Alignment output fingerprint is unavailable."
            )

        validation_config_fingerprint = _configuration_fingerprint(
            state_manager,
            _component_config(validator),
        )

        if _can_reuse_stage(
            state_manager,
            "data_validation",
            alignment_output_fingerprint,
            validation_config_fingerprint,
        ):

            logging.info(
                "[REUSE] Data Validation"
            )

            start_time = perf_counter()

            validation_result = _load_stage_cache(
                "data_validation"
            )

            stage_times["validation_sec"] = round(
                perf_counter() - start_time,
                2,
            )

            stage_status["data_validation"] = "reused"

        else:

            logging.info(
                "[RUN] Data Validation"
            )

            current_stage = "data_validation"
            start_time = perf_counter()

            validation_result = (
                validator.initiate_data_validation(
                    alignment_result.aligned_data
                )
            )

            elapsed = perf_counter() - start_time

            stage_times["validation_sec"] = round(
                elapsed,
                2,
            )

            cache_path = _save_stage_cache(
                "data_validation",
                validation_result,
            )

            output_fingerprint = hash_file(
                cache_path
            )

            state_manager.record_stage_success(
                stage_name="data_validation",
                input_fingerprint=alignment_output_fingerprint,
                config_fingerprint=validation_config_fingerprint,
                output_fingerprint=output_fingerprint,
                output_artifacts={
                    "cache": str(cache_path),
                },
            )

            stage_status["data_validation"] = "executed"

        # ==================================================================
        # 4. DATA CLEANING
        # ==================================================================

        cleaner = DataCleaning()

        validation_output_fingerprint = (
            state_manager.get_output_fingerprint(
                "data_validation"
            )
        )

        if validation_output_fingerprint is None:
            raise RuntimeError(
                "Data Validation output fingerprint is unavailable."
            )

        cleaning_config_fingerprint = _configuration_fingerprint(
            state_manager,
            _component_config(cleaner),
        )

        if _can_reuse_stage(
            state_manager,
            "data_cleaning",
            validation_output_fingerprint,
            cleaning_config_fingerprint,
        ):

            logging.info(
                "[REUSE] Data Cleaning"
            )

            start_time = perf_counter()

            cleaning_result = _load_stage_cache(
                "data_cleaning"
            )

            stage_times["cleaning_sec"] = round(
                perf_counter() - start_time,
                2,
            )

            stage_status["data_cleaning"] = "reused"

        else:

            logging.info(
                "[RUN] Data Cleaning"
            )

            current_stage = "data_cleaning"
            start_time = perf_counter()

            cleaning_result = (
                cleaner.initiate_data_cleaning(
                    validation_result.validated_data
                )
            )

            elapsed = perf_counter() - start_time

            stage_times["cleaning_sec"] = round(
                elapsed,
                2,
            )

            cache_path = _save_stage_cache(
                "data_cleaning",
                cleaning_result,
            )

            output_fingerprint = hash_file(
                cache_path
            )

            state_manager.record_stage_success(
                stage_name="data_cleaning",
                input_fingerprint=validation_output_fingerprint,
                config_fingerprint=cleaning_config_fingerprint,
                output_fingerprint=output_fingerprint,
                output_artifacts={
                    "cache": str(cache_path),
                },
            )

            stage_status["data_cleaning"] = "executed"

        # ==================================================================
        # 5. MASTER DATASET
        # ==================================================================

        builder = MasterDatasetBuilder()

        cleaning_output_fingerprint = (
            state_manager.get_output_fingerprint(
                "data_cleaning"
            )
        )

        if cleaning_output_fingerprint is None:
            raise RuntimeError(
                "Data Cleaning output fingerprint is unavailable."
            )

        builder_config_fingerprint = _configuration_fingerprint(
            state_manager,
            _component_config(builder),
        )

        if _can_reuse_stage(
            state_manager,
            "master_dataset",
            cleaning_output_fingerprint,
            builder_config_fingerprint,
        ):

            logging.info(
                "[REUSE] Master Dataset Builder"
            )

            start_time = perf_counter()

            dataset_build_result = _load_stage_cache(
                "master_dataset"
            )

            stage_times["consolidation_sec"] = round(
                perf_counter() - start_time,
                2,
            )

            stage_status["master_dataset"] = "reused"

        else:

            logging.info(
                "[RUN] Master Dataset Builder"
            )

            current_stage = "master_dataset"
            start_time = perf_counter()

            dataset_build_result = (
                builder.build_master_dataset(
                    cleaning_result.cleaned_data
                )
            )

            builder.save_result(
                result=dataset_build_result,
                output_dir=pipeline_config.master_dataset_dir,
            )

            elapsed = perf_counter() - start_time

            stage_times["consolidation_sec"] = round(
                elapsed,
                2,
            )

            cache_path = _save_stage_cache(
                "master_dataset",
                dataset_build_result,
            )

            output_fingerprint = hash_file(
                cache_path
            )

            state_manager.record_stage_success(
                stage_name="master_dataset",
                input_fingerprint=cleaning_output_fingerprint,
                config_fingerprint=builder_config_fingerprint,
                output_fingerprint=output_fingerprint,
                output_artifacts={
                    "cache": str(cache_path),
                    "master_dataset_dir": str(
                        pipeline_config.master_dataset_dir
                    ),
                },
            )

            stage_status["master_dataset"] = "executed"

        # ==================================================================
        # 6. COMMON FEATURE ENGINEERING
        # ==================================================================

        feature_engineering_config = (
            CommonFeatureEngineeringConfig()
        )

        feature_engineer = CommonFeatureEngineering(
            config=feature_engineering_config
        )

        master_dataset_output_fingerprint = (
            state_manager.get_output_fingerprint(
                "master_dataset"
            )
        )

        if master_dataset_output_fingerprint is None:
            raise RuntimeError(
                "Master Dataset output fingerprint is unavailable."
            )

        feature_config_fingerprint = _configuration_fingerprint(
            state_manager,
            feature_engineering_config,
        )

        if _can_reuse_stage(
            state_manager,
            "common_feature_engineering",
            master_dataset_output_fingerprint,
            feature_config_fingerprint,
        ):

            logging.info(
                "[REUSE] Common Feature Engineering"
            )

            start_time = perf_counter()

            feature_engineering_result = _load_stage_cache(
                "common_feature_engineering"
            )

            stage_times["feature_engineering_sec"] = round(
                perf_counter() - start_time,
                2,
            )

            stage_status[
                "common_feature_engineering"
            ] = "reused"

        else:

            logging.info(
                "[RUN] Common Feature Engineering"
            )

            current_stage = "common_feature_engineering"
            start_time = perf_counter()

            feature_engineering_result = (
                feature_engineer.run()
            )

            elapsed = perf_counter() - start_time

            stage_times[
                "feature_engineering_sec"
            ] = round(
                elapsed,
                2,
            )

            cache_path = _save_stage_cache(
                "common_feature_engineering",
                feature_engineering_result,
            )

            output_fingerprint = hash_file(
                cache_path
            )

            state_manager.record_stage_success(
                stage_name="common_feature_engineering",
                input_fingerprint=master_dataset_output_fingerprint,
                config_fingerprint=feature_config_fingerprint,
                output_fingerprint=output_fingerprint,
                output_artifacts={
                    "cache": str(cache_path),
                },
            )

            stage_status[
                "common_feature_engineering"
            ] = "executed"

        # ==================================================================
        # COMPLETE
        # ==================================================================

        total_execution_time = (
            perf_counter()
            - total_pipeline_start
        )

        stage_times["total_sec"] = round(
            total_execution_time,
            2,
        )

        pipeline_reused = all(
            status == "reused"
            for status in stage_status.values()
        )

        logging.info("=" * 75)
        logging.info(
            "STATE-AWARE DATA PIPELINE COMPLETED"
        )
        logging.info("=" * 75)

        for stage, status in stage_status.items():

            logging.info(
                "%-30s : %s",
                stage,
                status.upper(),
            )

        logging.info(
            "Total execution time: %.2f sec",
            total_execution_time,
        )

        return PipelineResult(
            cleaning_result=cleaning_result,
            validation_result=validation_result,
            dataset_build_result=dataset_build_result,
            feature_engineering_result=feature_engineering_result,
            stage_times=stage_times,
            stage_status=stage_status,
            pipeline_reused=pipeline_reused,
        )

    except CustomException:
        raise
    
    except Exception as e:
        
        if current_stage is not None:
            
            state_manager.record_stage_failure(stage_name=current_stage, error=str(e))

        logging.exception("State-aware data pipeline failed at stage '%s'", current_stage )

        raise CustomException( e, sys) from e

# ============================================================================
# CLI
# ============================================================================


def _write_pipeline_reports(
    pipeline_outputs: PipelineResult,
) -> None:
    """
    Write human-readable pipeline summary artifacts.

    Reporting is intentionally outside the execution logic so that
    reused stages still produce a fresh high-level execution report.
    """

    result = pipeline_outputs.cleaning_result
    validation_result = pipeline_outputs.validation_result
    dataset_build_result = (
        pipeline_outputs.dataset_build_result
    )
    feature_engineering_result = (
        pipeline_outputs.feature_engineering_result
    )

    stage_times = pipeline_outputs.stage_times

    relational_memory_mb = (
        _get_cleaned_memory_usage_mb(
            result.cleaned_data
        )
    )

    consolidated_memory_mb = (
        _get_consolidated_memory_usage_mb(
            dataset_build_result.consolidated_data
        )
    )

    feature_store_memory_mb = (
        _get_feature_store_memory_mb(
            feature_engineering_result.feature_store
        )
    )

    reports_dir = pipeline_config.summary_reports_dir

    pipeline_summary_data = {

        "pipeline_reused":
            pipeline_outputs.pipeline_reused,

        "stage_status":
            pipeline_outputs.stage_status,

        "versions_processed":
            result.summary.versions_processed,

        "tables_processed":
            result.summary.tables_processed,

        "rows_before":
            result.summary.total_rows_before,

        "rows_after":
            result.summary.total_rows_after,

        "rows_removed":
            result.summary.total_rows_removed,

        "percentage_rows_removed":
            result.summary.percentage_rows_removed,

        "nulls_filled":
            result.summary.total_nulls_filled,

        "duplicates_removed":
            result.summary.total_duplicates_removed,

        "orphans_removed":
            result.summary.total_orphan_rows_removed,

        "invalid_values_handled":
            result.summary.total_invalid_values_handled,

        "dtype_conversions":
            result.summary.total_dtype_conversions,

        "feature_store_rows":
            feature_engineering_result.summary.rows_after,

        "feature_store_columns":
            feature_engineering_result.summary.columns_after,

        "feature_store_schema_hash":
            feature_engineering_result.summary.schema_hash,

        "relational_memory_mb":
            relational_memory_mb,

        "consolidated_memory_mb":
            consolidated_memory_mb,

        "feature_store_memory_mb":
            feature_store_memory_mb,

        "stage_execution_times":
            stage_times,
    }

    _save_json_artifact(
        reports_dir / "pipeline_summary.json",
        pipeline_summary_data,
    )

    _save_json_artifact(
        reports_dir / "cleaning_summary.json",
        result,
    )

    _save_json_artifact(
        reports_dir / "validation_summary.json",
        validation_result,
    )

    _save_json_artifact(
        reports_dir / "consolidation_summary.json",
        dataset_build_result,
    )

    _save_json_artifact(
        reports_dir / "feature_engineering_summary.json",
        feature_engineering_result,
    )


# ============================================================================
# Entry Point
# ============================================================================


if __name__ == "__main__":

    try:

        outputs = run_pipeline()

        _write_pipeline_reports(outputs)

        print()
        print("=" * 75)
        print("STATE-AWARE DATA PIPELINE SUMMARY")
        print("=" * 75)

        for stage, status in outputs.stage_status.items():

            print(
                f"{stage:<35} : {status.upper()}"
            )

        print("-" * 75)

        print(
            f"Total execution time : "
            f"{outputs.stage_times['total_sec']} sec"
        )

        print("=" * 75)

    except CustomException:

        sys.exit(1)

    except Exception as e:

        logging.exception(
            "Pipeline execution failed: %s",
            e,
        )

        sys.exit(1)
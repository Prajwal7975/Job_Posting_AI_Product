"""
Pipeline entrypoint.

Chains the pipeline stages together, in order:

    Data Ingestion -> Schema Alignment -> Data Validation -> Data Cleaning -> Master Dataset Builder -> Common Feature Engineering -> Feature Store

Run with:
    python -m src.pipeline

Execution logic belongs in exactly one place: here.
"""

import json
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

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
from src.components.common_feature_engineering import CommonFeatureEngineering
from src.configs.common_feature_engineering_config import (
    CommonFeatureEngineeringConfig,
)
from src.entity.common_feature_engineering_entity import (
    FeatureEngineeringResult,
)
from src.configs.pipeline_config import PipelineConfig
from src.exception import CustomException
from src.logger import logging

pipeline_config = PipelineConfig()


@dataclass
class PipelineResult:
    cleaning_result: DataCleaningResult
    validation_result: DataValidationResult
    dataset_build_result: DatasetBuildResult
    feature_engineering_result: FeatureEngineeringResult
    stage_times: dict[str, float]


def _get_cleaned_memory_usage_mb(data: Dict[str, Dict[str, pd.DataFrame]]) -> float:
    """Calculates memory usage of all DataFrames in memory in MB."""
    total_bytes = sum(
        df.memory_usage(deep=True).sum()
        for version in data.values()
        for df in version.values()
    )
    return round(total_bytes / (1024 * 1024), 2)


def _get_consolidated_memory_usage_mb(consolidated_data: Dict[str, pd.DataFrame]) -> float:
    """Calculates memory usage of consolidated analytical DataFrames in MB."""
    total_bytes = sum(df.memory_usage(deep=True).sum() for df in consolidated_data.values())
    return round(total_bytes / (1024 * 1024), 2)


def _get_feature_store_memory_mb(df: pd.DataFrame) -> float:
    """Calculates memory usage of the feature store DataFrame in MB."""
    return round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)


def _save_json_artifact(file_path: Path, data: Any) -> None:
    """Safely writes dictionaries or dataclass objects to a JSON artifact."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if is_dataclass(data) and not isinstance(data, type):
        data = asdict(data)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)
        
    logging.info("Saved artifact: %s", file_path)


def run_pipeline() -> PipelineResult:
    """
    Runs every pipeline stage implemented so far:

        1. Data Ingestion           - discover + load raw CSVs per dataset version
        2. Schema Alignment         - align every table to its canonical schema
        3. Data Validation          - detect data-quality issues
        4. Data Cleaning            - clean strings, fill nulls, convert dtypes, drop orphans
        5. Master Dataset Builder   - aggregate & denormalize into 1:1 job_id Analytical Base Table
        6. Common Feature Engineering - build model-agnostic feature store

    Returns execution artifacts, validation, cleaning, consolidation, and feature engineering results.
    """
    total_pipeline_start = perf_counter()

    try:
        logging.info("#" * 70)
        logging.info("PIPELINE RUN STARTED")
        logging.info("#" * 70)

        # 1. Data Ingestion
        start_time = perf_counter()
        logging.info("Starting Data Ingestion...")
        ingestion = DataIngestion()
        raw_datasets: Dict[str, Dict[str, pd.DataFrame]] = (
            ingestion.initiate_data_ingestion()
        )
        ingestion_time = perf_counter() - start_time
        logging.info("Data Ingestion completed in %.2f sec", ingestion_time)

        # 2. Schema Alignment
        start_time = perf_counter()
        logging.info("Starting Schema Alignment...")
        aligner = SchemaAlignment()
        alignment_result = aligner.initiate_schema_alignment(raw_datasets)
        alignment_time = perf_counter() - start_time
        logging.info("Schema Alignment completed in %.2f sec", alignment_time)

        # 3. Data Validation
        start_time = perf_counter()
        logging.info("Starting Data Validation...")
        validator = DataValidation()
        validation_result = validator.initiate_data_validation(
            alignment_result.aligned_data
        )
        validation_time = perf_counter() - start_time
        logging.info("Data Validation completed in %.2f sec", validation_time)

        # 4. Data Cleaning
        start_time = perf_counter()
        logging.info("Starting Data Cleaning...")
        cleaner = DataCleaning()
        cleaning_result: DataCleaningResult = cleaner.initiate_data_cleaning(
            validation_result.validated_data
        )
        cleaning_time = perf_counter() - start_time
        logging.info("Data Cleaning completed in %.2f sec", cleaning_time)

        # 5. Master Dataset Builder
        start_time = perf_counter()
        logging.info("Starting Master Dataset Builder...")

        builder = MasterDatasetBuilder()

        dataset_build_result: DatasetBuildResult = builder.build_master_dataset(cleaning_result.cleaned_data)

        # Persist master dataset and metadata
        builder.save_result(result=dataset_build_result, output_dir=pipeline_config.master_dataset_dir)

        consolidation_time = perf_counter() - start_time
        logging.info("Master Dataset Builder completed in %.2f sec", consolidation_time)

        # 6. Common Feature Engineering
        start_time = perf_counter()
        logging.info("Starting Common Feature Engineering...")

        feature_engineering_config = CommonFeatureEngineeringConfig()
        feature_engineer = CommonFeatureEngineering( config=feature_engineering_config )
        feature_engineering_result: FeatureEngineeringResult = feature_engineer.run()

        feature_engineering_time = perf_counter() - start_time
        logging.info(
            "Common Feature Engineering completed in %.2f sec",
            feature_engineering_time,
        )

        total_execution_time = perf_counter() - total_pipeline_start

        logging.info("#" * 70)
        logging.info("PIPELINE RUN COMPLETED SUCCESSFULLY")
        logging.info("#" * 70)

        return PipelineResult(
            cleaning_result=cleaning_result,
            validation_result=validation_result,
            dataset_build_result=dataset_build_result,
            feature_engineering_result=feature_engineering_result,
            stage_times={
                "ingestion_sec": round(ingestion_time, 2),
                "alignment_sec": round(alignment_time, 2),
                "validation_sec": round(validation_time, 2),
                "cleaning_sec": round(cleaning_time, 2),
                "consolidation_sec": round(consolidation_time, 2),
                "feature_engineering_sec": round(feature_engineering_time, 2),
                "total_sec": round(total_execution_time, 2),
            },
        )

    except CustomException:
        raise
    except Exception as e:
        logging.exception("Pipeline run failed")
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    pipeline_outputs = run_pipeline()
    
    result: DataCleaningResult = pipeline_outputs.cleaning_result
    validation_result = pipeline_outputs.validation_result
    dataset_build_result = pipeline_outputs.dataset_build_result
    feature_engineering_result = pipeline_outputs.feature_engineering_result
    stage_times = pipeline_outputs.stage_times

    # Calculate memory footprint
    relational_memory_mb = _get_cleaned_memory_usage_mb(result.cleaned_data)
    consolidated_memory_mb = _get_consolidated_memory_usage_mb(
        dataset_build_result.consolidated_data
    )
    feature_store_memory_mb = _get_feature_store_memory_mb(
        feature_engineering_result.feature_store
    )

    # -------------------------------------------------------------------------
    # Save Metrics Artifacts to `summary_reports_dir`
    # -------------------------------------------------------------------------
    reports_dir = pipeline_config.summary_reports_dir

    # 1. Pipeline High-Level Summary
    pipeline_summary_data = {
        "versions_processed": result.summary.versions_processed,
        "tables_processed": result.summary.tables_processed,
        "rows_before": result.summary.total_rows_before,
        "rows_after": result.summary.total_rows_after,
        "rows_removed": result.summary.total_rows_removed,
        "percentage_rows_removed": result.summary.percentage_rows_removed,
        "nulls_filled": result.summary.total_nulls_filled,
        "duplicates_removed": result.summary.total_duplicates_removed,
        "orphans_removed": result.summary.total_orphan_rows_removed,
        "invalid_values_handled": result.summary.total_invalid_values_handled,
        "dtype_conversions": result.summary.total_dtype_conversions,
        "feature_store_rows": feature_engineering_result.summary.rows_after,
        "feature_store_columns": feature_engineering_result.summary.columns_after,
        "feature_store_schema_hash": feature_engineering_result.summary.schema_hash,
        "relational_memory_mb": relational_memory_mb,
        "consolidated_memory_mb": consolidated_memory_mb,
        "feature_store_memory_mb": feature_store_memory_mb,
        "stage_execution_times": stage_times,
    }
    _save_json_artifact(reports_dir / "pipeline_summary.json", pipeline_summary_data)

    # 2. Cleaning Detailed Summary
    _save_json_artifact(reports_dir / "cleaning_summary.json", result)

    # 3. Validation Detailed Summary
    _save_json_artifact(reports_dir / "validation_summary.json", validation_result)

    # 4. Consolidation Detailed Summary
    _save_json_artifact(reports_dir / "consolidation_summary.json", dataset_build_result)

    # 5. Feature Engineering Detailed Summary
    _save_json_artifact(reports_dir / "feature_engineering_summary.json", feature_engineering_result)

    # -------------------------------------------------------------------------
    # Console Output Banner
    # -------------------------------------------------------------------------
    banner = f"""
==========================================================
PIPELINE SUMMARY
==========================================================
Versions Processed    : {result.summary.versions_processed}
Tables Processed      : {result.summary.tables_processed}
Rows Before (Raw)     : {result.summary.total_rows_before:,}
Rows After (Cleaned)  : {result.summary.total_rows_after:,}
Rows Removed          : {result.summary.total_rows_removed:,}
Nulls Filled          : {result.summary.total_nulls_filled:,}
Duplicates Removed    : {result.summary.total_duplicates_removed:,}
Orphans Removed       : {result.summary.total_orphan_rows_removed:,}
----------------------------------------------------------
CONSOLIDATED ANALYTICAL BASE TABLES
----------------------------------------------------------
Analytical Rows       : {dataset_build_result.summary.rows_processed:,}
Total Output Columns  : {dataset_build_result.summary.new_columns:,}
----------------------------------------------------------
FEATURE STORE
----------------------------------------------------------
Rows                  : {feature_engineering_result.summary.rows_after:,}
Columns               : {feature_engineering_result.summary.columns_after:,}
Schema Hash           : {feature_engineering_result.summary.schema_hash[:16]}...
----------------------------------------------------------
Total Execution Time  : {stage_times['total_sec']} sec
==========================================================
Relational Memory     : {relational_memory_mb} MB
Consolidated Memory   : {consolidated_memory_mb} MB
Feature Store Memory  : {feature_store_memory_mb} MB
==========================================================
"""
    print(banner)
    logging.info(banner)
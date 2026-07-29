"""
Pipeline entrypoint.

Chains the pipeline stages together, in order:

    Data Ingestion -> Schema Alignment -> Data Validation -> Data Cleaning -> (Consolidation -> ...)

Run with:
    python -m src.pipeline

Why this file exists (and `if __name__ == "__main__"` doesn't live in
each component instead):

Each component (`DataIngestion`, `SchemaAlignment`, `DataValidation`, and
`DataCleaning`) is meant to be a pure, importable library with zero execution
side-effects -- importing any of them should never have a chance of kicking off a
pipeline run. Execution logic belongs in exactly one place: here. This also means
there is a single, obvious spot to look when you want to know "what does an
end-to-end run actually do", instead of that answer being scattered across N
components' `__main__` blocks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Dict, Any

import pandas as pd
from dataclasses import dataclass
from src.components.data_ingestion import DataIngestion
from src.components.schema_alignment import SchemaAlignment
from src.components.data_validation import DataValidation, DataValidationResult
from src.components.data_cleaning import DataCleaning, DataCleaningResult
from src.exception import CustomException
from src.logger import logging
from src.configs.Pipeline_Config import PipelineConfig

pipeline_config = PipelineConfig()

@dataclass
class PipelineResult:
    cleaning_result: DataCleaningResult
    validation_result: DataValidationResult
    stage_times: dict[str, float]


def _get_cleaned_memory_usage_mb(data: Dict[str, Dict[str, pd.DataFrame]]) -> float:
    """Calculates memory usage of all DataFrames in memory in MB."""
    total_bytes = sum(
        df.memory_usage(deep=True).sum()
        for version in data.values()
        for df in version.values()
    )
    return round(total_bytes / (1024 * 1024), 2)


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

        1. Data Ingestion   - discover + load raw CSVs per dataset version
        2. Schema Alignment - align every table to its canonical schema
        3. Data Validation  - detect (never fix) data-quality issues
        4. Data Cleaning    - clean strings, fill nulls, convert dtypes, handle 
                              invalids, deduplicate, and drop orphans

    Returns execution artifacts, validation results, and DataCleaningResult.
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

        total_execution_time = perf_counter() - total_pipeline_start

        logging.info("#" * 70)
        logging.info("PIPELINE RUN COMPLETED SUCCESSFULLY")
        logging.info("#" * 70)

        return PipelineResult(
            cleaning_result=cleaning_result,
            validation_result=validation_result,
            stage_times={
                "ingestion_sec": round(ingestion_time, 2),
                "alignment_sec": round(alignment_time, 2),
                "validation_sec": round(validation_time, 2),
                "cleaning_sec": round(cleaning_time, 2),
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
    stage_times = pipeline_outputs.stage_times

    # Calculate memory footprint
    memory_usage_mb = _get_cleaned_memory_usage_mb(result.cleaned_data)

    # -------------------------------------------------------------------------
    # Save Metrics Artifacts to `artifacts/`
    # -------------------------------------------------------------------------

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
        "memory_usage_mb": memory_usage_mb,
        "stage_execution_times": stage_times,
    }
    _save_json_artifact(pipeline_config.artifacts_dir / "pipeline_summary.json", pipeline_summary_data)

    # 2. Cleaning Detailed Summary & Per-Table Reports
    cleaning_summary_data = {
        "summary": result.summary,
        "reports": result.reports,
    }
    _save_json_artifact(pipeline_config.artifacts_dir / "cleaning_summary.json", cleaning_summary_data)

    # 3. Validation Detailed Summary & Reports
    if hasattr(validation_result, "summary") or is_dataclass(validation_result):
        _save_json_artifact(pipeline_config.artifacts_dir / "validation_summary.json", validation_result)
    else:
        _save_json_artifact(pipeline_config.artifacts_dir / "validation_summary.json", {"validation": str(validation_result)})

    # -------------------------------------------------------------------------
    # Console Output Banner
    # -------------------------------------------------------------------------
    banner = f"""
==========================================================
PIPELINE SUMMARY
==========================================================
Versions Processed : {result.summary.versions_processed}
Tables Processed   : {result.summary.tables_processed}
Rows Before        : {result.summary.total_rows_before:,}
Rows After         : {result.summary.total_rows_after:,}
Rows Removed       : {result.summary.total_rows_removed:,}
Nulls Filled       : {result.summary.total_nulls_filled:,}
Duplicates Removed : {result.summary.total_duplicates_removed:,}
Orphans Removed    : {result.summary.total_orphan_rows_removed:,}
Execution Time     : {stage_times['total_sec']} sec
==========================================================
Memory Usage       : {memory_usage_mb} MB
==========================================================
"""
    print(banner)
    logging.info(banner)
from __future__ import annotations

import sys
from typing import Dict

import pandas as pd

from src.components.data_cleaning import DataCleaning, DataCleaningResult
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.schema_alignment import SchemaAlignment
from src.exception import CustomException
from src.logger import logging


def run_pipeline() -> DataCleaningResult:
    """
    Runs every pipeline stage implemented so far:

        1. Data Ingestion   - discover + load raw CSVs per dataset version
        2. Schema Alignment - align every table to its canonical schema
        3. Data Validation  - detect (never fix) data-quality issues
        4. Data Cleaning    - clean strings, fill nulls, convert dtypes, handle
                              invalids, deduplicate, and drop orphans

    Returns the `DataCleaningResult` (cleaned data, per-table reports,
    and a run summary) so a caller -- a notebook, a future orchestrator,
    a test -- can inspect the outcome directly instead of re-parsing logs.

    As later stages (Dataset Consolidation, Feature Engineering, ...) are
    implemented, they get added here, in order, each consuming the
    previous stage's output.
    """

    try:
        logging.info("#" * 70)
        logging.info("PIPELINE RUN STARTED")
        logging.info("#" * 70)

        # 1. Data Ingestion
        ingestion = DataIngestion()
        raw_datasets: Dict[str, Dict[str, pd.DataFrame]] = (
            ingestion.initiate_data_ingestion()
        )

        # 2. Schema Alignment
        aligner = SchemaAlignment()
        alignment_result = aligner.initiate_schema_alignment(raw_datasets)

        # 3. Data Validation
        validator = DataValidation()
        validation_result = validator.initiate_data_validation(
            alignment_result.aligned_data
        )

        # 4. Data Cleaning
        cleaner = DataCleaning()
        cleaning_result: DataCleaningResult = cleaner.initiate_data_cleaning(
            validation_result.validated_data
        )

        logging.info("#" * 70)
        logging.info("PIPELINE RUN COMPLETED SUCCESSFULLY")
        logging.info("#" * 70)

        return cleaning_result

    except CustomException:
        raise
    except Exception as e:
        logging.exception("Pipeline run failed")
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    result = run_pipeline()

    # Logging High-Level Execution Results
    logging.info("Cleaned versions: %s", list(result.cleaned_data.keys()))
    logging.info(
        "Tables processed: %d (Passed: %d, Failed: %d)",
        result.summary.tables_processed,
        result.summary.tables_passed,
        result.summary.tables_failed,
    )
    logging.info(
        "Rows summary: %d -> %d (%d removed / %.2f%% reduction)",
        result.summary.total_rows_before,
        result.summary.total_rows_after,
        result.summary.total_rows_removed,
        result.summary.percentage_rows_removed,
    )
    logging.info(
        "Cleaning operations: [Nulls Filled: %d] [Invalids Handled: %d] "
        "[Duplicates Removed: %d] [Orphans Purged: %d] [Dtype Conversions: %d]",
        result.summary.total_nulls_filled,
        result.summary.total_invalid_values_handled,
        result.summary.total_duplicates_removed,
        result.summary.total_orphan_rows_removed,
        result.summary.total_dtype_conversions,
    )
    logging.info("Total execution time: %.4f seconds", result.summary.execution_time_seconds)
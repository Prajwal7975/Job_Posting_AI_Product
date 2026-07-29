"""
Pipeline entrypoint.

Chains the pipeline stages together, in order:

    Data Ingestion -> Schema Alignment -> Data Validation -> (Data Cleaning -> ... )

Run with:
    python -m src.pipeline

Why this file exists (and `if __name__ == "__main__"` doesn't live in
each component instead):

Each component (`DataIngestion`, `SchemaAlignment`, `DataValidation`, and
future stages like Data Cleaning) is meant to be a pure, importable
library with zero execution side-effects -- importing any of them should
never have a chance of kicking off a pipeline run. Execution logic
belongs in exactly one place: here. This also means there is a single,
obvious spot to look when you want to know "what does an end-to-end run
actually do", instead of that answer being scattered across N
components' `__main__` blocks.
"""

from __future__ import annotations

import sys
from typing import Dict

import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation, DataValidationResult
from src.components.schema_alignment import SchemaAlignment
from src.exception import CustomException
from src.logger import logging


def run_pipeline() -> DataValidationResult:
    """
    Runs every pipeline stage implemented so far:

        1. Data Ingestion   - discover + load raw CSVs per dataset version
        2. Schema Alignment - align every table to its canonical schema
        3. Data Validation  - detect (never fix) data-quality issues

    Returns the `DataValidationResult` (validated data, per-table
    reports, and a run summary) so a caller -- a notebook, a future
    orchestrator, a test -- can inspect the outcome directly instead of
    re-parsing logs.

    As later stages (Data Cleaning, Dataset Consolidation, ...) are
    implemented, they get added here, in order, each consuming the
    previous stage's output.
    """

    try:
        logging.info("#" * 70)
        logging.info("PIPELINE RUN STARTED")
        logging.info("#" * 70)

        ingestion = DataIngestion()
        raw_datasets: Dict[str, Dict[str, pd.DataFrame]] = (
            ingestion.initiate_data_ingestion()
        )

        aligner = SchemaAlignment()
        alignment_result = aligner.initiate_schema_alignment(raw_datasets)

        validator = DataValidation()
        validation_result = validator.initiate_data_validation(
            alignment_result.aligned_data
        )

        logging.info("#" * 70)
        logging.info("PIPELINE RUN COMPLETED")
        logging.info("#" * 70)

        return validation_result

    except CustomException:
        raise
    except Exception as e:
        logging.exception("Pipeline run failed")
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    result = run_pipeline()

    logging.info("Validated versions: %s", list(result.validated_data.keys()))
    logging.info(
        "Tables passed: %d, Tables failed: %d, Tables skipped (no rules): %d",
        result.summary.tables_passed,
        result.summary.tables_failed,
        result.summary.tables_skipped_no_rules,
    )
    if not result.summary.passed:
        logging.warning(
            "Tables that failed validation: %s",
            [(r.version, r.table) for r in result.reports if r.rules_defined and not r.passed],
        )
    logging.info("Validation summary: %s", result.summary.as_log_dict())
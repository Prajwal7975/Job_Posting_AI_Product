from __future__ import annotations
import sys
from typing import Dict
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.schema_alignment import SchemaAlignment, SchemaAlignmentResult
from src.exception import CustomException
from src.logger import logging


def run_pipeline() -> SchemaAlignmentResult:
    
    try:
        logging.info("#" * 70)
        logging.info("PIPELINE RUN STARTED")
        logging.info("#" * 70)

        ingestion = DataIngestion()
        raw_datasets: Dict[str, Dict[str, pd.DataFrame]] = (
            ingestion.initiate_data_ingestion()
        )
        
        logging.info("Loaded %d dataset version(s): %s",len(raw_datasets),list(raw_datasets.keys()),
)

        aligner = SchemaAlignment()
        alignment_result = aligner.initiate_schema_alignment(raw_datasets)
        
        logging.info("Pipeline Summary: %s",alignment_result.summary.as_log_dict()
)

        logging.info("#" * 70)
        logging.info("PIPELINE RUN COMPLETED")
        logging.info("#" * 70)

        return alignment_result

    except CustomException:
        raise
    except Exception as e:
        logging.exception("Pipeline run failed")
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    result = run_pipeline()

    logging.info("Aligned versions: %s", list(result.aligned_data.keys()))
    logging.info(
        "Tables with structural issues: %s",
        [
            (r.version, r.table)
            for r in result.reports
            if r.has_structural_issues()
        ],
    )
    logging.info("Alignment summary: %s", result.summary.as_log_dict())
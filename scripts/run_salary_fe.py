from __future__ import annotations

import sys

from src.logger import logging
from src.exception import CustomException

from src.components.salary_predict.salary_feature_engineering import (
    SalaryFeatureEngineering,
)

from src.configs.salary_predict.salary_feature_engineering_config import (
    SalaryFeatureEngineeringConfig,
)


def main() -> None:

    logging.info("=" * 80)
    logging.info("SALARY FEATURE ENGINEERING - STANDALONE RUN")
    logging.info("=" * 80)

    try:

        # ----------------------------------------------------------
        # Load configuration
        # ----------------------------------------------------------

        config = SalaryFeatureEngineeringConfig()

        logging.info(
            "Input feature store: %s",
            config.feature_store_path,
        )

        logging.info(
            "Output directory: %s",
            config.latest_dir,
        )

        # ----------------------------------------------------------
        # Run ONLY Salary Feature Engineering
        # ----------------------------------------------------------

        feature_engineering = SalaryFeatureEngineering(
            config=config
        )

        result = (
            feature_engineering
            .initiate_salary_feature_engineering()
        )

        # ----------------------------------------------------------
        # Result
        # ----------------------------------------------------------

        print("\n" + "=" * 80)
        print("SALARY FEATURE ENGINEERING COMPLETED")
        print("=" * 80)

        print(
            f"Input rows              : "
            f"{result.summary.input_row_count:,}"
        )

        print(
            f"Salary candidates       : "
            f"{result.summary.salary_candidate_count:,}"
        )

        print(
            f"Valid USD salaries      : "
            f"{result.summary.valid_usd_salary_count:,}"
        )

        print(
            f"Rows removed - salary   : "
            f"{result.summary.rows_removed_missing_salary:,}"
        )

        print(
            f"Rows removed - currency : "
            f"{result.summary.rows_removed_currency:,}"
        )

        print(
            f"Rows removed - period   : "
            f"{result.summary.rows_removed_unsupported_pay_period:,}"
        )

        print(
            f"Rows removed - bounds   : "
            f"{result.summary.rows_removed_out_of_bounds:,}"
        )

        print(
            f"Final rows              : "
            f"{result.summary.final_row_count:,}"
        )

        print(
            f"Target coverage         : "
            f"{result.summary.target_coverage_pct:.2f}%"
        )

        print(
            f"Target sources          : "
            f"{result.summary.target_source_counts}"
        )

        print(
            f"\nOutput dataset:"
            f"\n{result.salary_modeling_dataset_path}"
        )

        print(
            f"\nMetadata:"
            f"\n{result.metadata_path}"
        )

        print(
            f"\nReport:"
            f"\n{result.report_path}"
        )

        print(
            f"\nSchema fingerprint:"
            f"\n{result.schema_fingerprint_path}"
        )

        print("\n" + "=" * 80)
        print("ONLY SALARY FEATURE ENGINEERING WAS EXECUTED")
        print("=" * 80)

    except Exception as e:

        logging.exception(
            "Standalone Salary Feature Engineering failed."
        )

        raise CustomException(
            e,
            sys,
        ) from e


if __name__ == "__main__":
    main()
from __future__ import annotations

import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from src.components.salary_predict.salary_model_factory import (
    SalaryModelFactory,
)

from src.components.salary_predict.salary_mlflow_tracker import (
    SalaryMLflowTracker,
)

from src.components.salary_predict.single_salary_model_runner import (
    SalarySingleModelExperimentRunner,
)

from src.components.salary_predict.salary_model_family_runner import (
    SalaryModelFamilyExperimentRunner,
)


def build_test_dataset():
    """
    Small synthetic regression dataset used only to verify
    the model-family experiment flow.

    This is NOT a model-quality evaluation.
    """

    X_train = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature_2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        }
    )

    y_train = pd.Series(
        [100, 120, 150, 170, 200, 230, 250, 280, 300, 330],
        name="salary",
    )

    X_test = pd.DataFrame(
        {
            "feature_1": [11, 12, 13],
            "feature_2": [0, -1, -2],
        }
    )

    y_test = pd.Series(
        [350, 380, 400],
        name="salary",
    )

    return X_train, y_train, X_test, y_test


def main() -> None:

    logging.info("=" * 80)
    logging.info("SALARY MODEL FAMILY SMOKE TEST")
    logging.info("=" * 80)

    try:

        # --------------------------------------------------------------
        # 1. Create small synthetic dataset
        # --------------------------------------------------------------

        X_train, y_train, X_test, y_test = build_test_dataset()

        logging.info(
            "Test dataset created: train=%s, test=%s",
            X_train.shape,
            X_test.shape,
        )

        # --------------------------------------------------------------
        # 2. Create model factory
        # --------------------------------------------------------------

        factory = SalaryModelFactory()

        logging.info(
            "Registered model families: %s",
            factory.list_supported_models(),
        )

        # --------------------------------------------------------------
        # 3. Create MLflow tracker
        # --------------------------------------------------------------
        #
        # For this smoke test we are only validating the model-family
        # execution flow. If your tracker supports a disabled/test mode,
        # use it here.
        #
        # Otherwise, use your normal tracker but point it to a local
        # development experiment.
        # --------------------------------------------------------------

        mlflow_tracker = SalaryMLflowTracker()

        # --------------------------------------------------------------
        # 4. Create single-model runner
        # --------------------------------------------------------------

        single_runner = SalarySingleModelExperimentRunner(
            model_factory=factory,
            mlflow_tracker=mlflow_tracker,
        )

        # --------------------------------------------------------------
        # 5. Create model-family orchestrator
        # --------------------------------------------------------------

        family_runner = SalaryModelFamilyExperimentRunner(
            single_model_runner=single_runner,
            mlflow_tracker=mlflow_tracker,
        )

        # --------------------------------------------------------------
        # 6. Run all configured model-family experiments
        # --------------------------------------------------------------

        summary = family_runner.run_experiments(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_experiment_id="E3B_TEST",
            ranking_metric="RMSE",
        )

        # --------------------------------------------------------------
        # 7. Display results
        # --------------------------------------------------------------

        print("\n")
        print("=" * 80)
        print("MODEL FAMILY EXPERIMENT RESULTS")
        print("=" * 80)

        print(f"Feature configuration : {summary.feature_experiment_id}")
        print(f"Experiments           : {summary.experiment_count}")
        print(f"Successful            : {summary.successful_experiment_count}")
        print(f"Failed                : {summary.failed_experiment_count}")
        print(f"Ranking metric        : {summary.ranking_metric}")
        print(f"Direction             : {summary.ranking_direction}")

        print("\nRanking")
        print("-" * 80)

        for rank, result in enumerate(
            summary.ranked_results,
            start=1,
        ):

            score = result.metrics.get(summary.ranking_metric)

            print(
                f"{rank}. "
                f"{result.model_name:<15} "
                f"{summary.ranking_metric}={score:.4f} "
                f"success={result.success}"
            )

        # --------------------------------------------------------------
        # 8. Winner
        # --------------------------------------------------------------

        print("\n")
        print("=" * 80)
        print("WINNER")
        print("=" * 80)

        print(f"Experiment : " f"{summary.winner_experiment_id}")

        print(f"Model      : " f"{summary.winner_model_name}")

        print(f"Class      : " f"{summary.winner_model_class}")

        print(f"Score      : " f"{summary.winner_score:.4f}")

        print("=" * 80)

        logging.info("Model family smoke test completed successfully.")

    except CustomException as e:

        logging.error(
            "Model family smoke test failed: %s",
            e,
            exc_info=True,
        )

        sys.exit(1)

    except Exception as e:

        logging.error(
            "Unexpected smoke test failure: %s",
            e,
            exc_info=True,
        )

        sys.exit(1)


if __name__ == "__main__":
    main()

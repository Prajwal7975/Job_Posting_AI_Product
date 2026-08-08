import pandas as pd
import scipy.sparse as sp
from src.logger import logging

from src.configs.salary_predict.salary_experiment_config import (
    get_experiment_config,
)
from src.components.salary_predict.salary_preprocessor_builder import (
    SalaryPreprocessorBuilder,
)

TRAIN_PATH = "artifacts/salary_dataset_splits/latest/train.parquet"
VALIDATION_PATH = "artifacts/salary_dataset_splits/latest/validation.parquet"


def main():
    # ----------------------------------------------------------
    # Load REAL split artifacts
    # ----------------------------------------------------------

    train_df = pd.read_parquet(TRAIN_PATH)
    validation_df = pd.read_parquet(VALIDATION_PATH)

    # Small samples are enough for a smoke test.
    # We do NOT need to TF-IDF all 30k+ rows just to verify construction.
    train_sample = train_df.head(5000).copy()
    validation_sample = validation_df.head(1000).copy()

    builder = SalaryPreprocessorBuilder()

    # ----------------------------------------------------------
    # Experiments currently available before E3 winner selection
    # ----------------------------------------------------------

    experiment_ids = [
        "E0",
        "E1",
        "E2",
        "E3A",
        "E3B",
    ]

    logging.info("=" * 70)
    logging.info("SALARY PREPROCESSOR SMOKE TEST")
    logging.info("=" * 70)

    logging.info(f"Train shape      : {train_df.shape}")
    logging.info(f"Validation shape : {validation_df.shape}")

    for exp_id in experiment_ids:

        logging.info("=" * 70)
        logging.info(f"TESTING {exp_id}")
        logging.info("=" * 70)

        config = get_experiment_config(exp_id)

        missing_train = builder.validate_input_columns(
            train_sample.columns,
            config,
        )

        missing_validation = builder.validate_input_columns(
            validation_sample.columns,
            config,
        )

        assert (
            not missing_train
        ), f"{exp_id}: Train data missing required columns: {missing_train}"

        assert not missing_validation, (
            f"{exp_id}: Validation data missing required columns: "
            f"{missing_validation}"
        )

        preprocessor = builder.build(config)

        if exp_id == "E0":

            assert preprocessor is None

            logging.info(f"{exp_id} PASS - Dummy baseline correctly returned None.")

            continue

        assert preprocessor is not None

        X_train = preprocessor.fit_transform(train_sample)

        X_validation = preprocessor.transform(validation_sample)

        assert X_train.shape[0] == len(train_sample)
        assert X_validation.shape[0] == len(validation_sample)

        assert X_train.shape[1] == X_validation.shape[1]

        feature_names = preprocessor.get_feature_names_out()

        assert len(feature_names) == X_train.shape[1]

        train_sparse = sp.issparse(X_train)
        validation_sparse = sp.issparse(X_validation)

        logging.info(f"{exp_id} PASS")
        logging.info(f"Train matrix      : {X_train.shape}")
        logging.info(f"Validation matrix : {X_validation.shape}")
        logging.info(f"Train type        : {type(X_train).__name__}")
        logging.info(f"Validation type   : {type(X_validation).__name__}")
        logging.info(f"Train sparse      : {train_sparse}")
        logging.info(f"Validation sparse : {validation_sparse}")
        logging.info(f"Generated features: {len(feature_names):,}")
        logging.info(f"First features    : {feature_names[:10].tolist()}")

    logging.info("=" * 70)
    logging.info("ALL SALARY PREPROCESSOR SMOKE TESTS PASSED")
    logging.info("=" * 70)
    print("ALL SALARY PREPROCESSOR SMOKE TESTS PASSED")

if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        logging.exception(f"SALARY PREPROCESSOR SMOKE TEST FAILED: {e}")
        raise

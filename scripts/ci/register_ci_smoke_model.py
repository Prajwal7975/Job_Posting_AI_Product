"""
scripts/ci/register_ci_smoke_model.py
The pipeline uses the REAL SalaryPreprocessorBuilder (E3B architecture:
title + skill_list multihot + core categoricals + skill_count), so it
is schema-compatible with the real SalaryInferenceFeatureBuilder output
and genuinely exercises production preprocessing code during the smoke
test -- it does not stub or bypass inference logic. The training data
is small, synthetic, and non-personal; the model's actual predictions
are not production salary figures and must never be treated as such.

Usage (only ever against a disposable CI MLflow instance):

    MLFLOW_TRACKING_URI=http://localhost:5000 \
        python scripts/ci/register_ci_smoke_model.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Running `python scripts/ci/register_ci_smoke_model.py` puts THIS
# script's directory (scripts/ci/) at sys.path[0], not the repository
# root, so `src` would not be importable. Add the repo root explicitly,
# derived from this file's location (scripts/ci/<this file> -> parents[2]),
# so the script works from any working directory on Windows and Linux
# without requiring PYTHONPATH to be set.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src.components.salary_predict.salary_preprocessor_builder import (  # noqa: E402
    SalaryPreprocessorBuilder,
)
from src.configs.salary_predict.salary_experiment_config import (  # noqa: E402
    build_e3b_config,
)

REGISTERED_MODEL_NAME = os.getenv(
    "SALARY_REGISTERED_MODEL_NAME", "salary_prediction_model"
)
MODEL_ALIAS = os.getenv("SALARY_MODEL_ALIAS", "production")


def _build_ci_smoke_model() -> Pipeline:
    """A tiny, deterministic pipeline on fixed synthetic data -- never
    derived from or compared against production data/artifacts."""
    df = pd.DataFrame(
        {
            "title": [
                "Data Scientist",
                "Data Engineer",
                "Data Analyst",
                "Senior Data Scientist",
                "Lead Data Engineer",
                "Staff Data Analyst",
                "Machine Learning Engineer",
                "Software Engineer",
            ],
            "skill_list": [
                "python|sql",
                "python|aws",
                "sql|docker",
                "aws|docker",
                "python|docker",
                "python|sql|aws",
                "sql|docker|aws",
                "python|docker|sql",
            ],
            "formatted_experience_level": [
                "Entry level",
                "Mid-Senior level",
                "Associate",
                "Entry level",
                "Mid-Senior level",
                "Associate",
                "Entry level",
                "Mid-Senior level",
            ],
            "company_state": ["CA", "NY", None, "TX", "CA", "NY", "TX", "CA"],
            "company_country": ["US"] * 8,
            "top_industry": [
                "Tech",
                "Tech",
                "Tech",
                "Retail",
                "Tech",
                "Tech",
                "Retail",
                "Tech",
            ],
            "skill_count": [2, 2, 2, 2, 2, 3, 3, 3],
        }
    )
    y = np.log1p(
        [95000.0, 130000.0, 80000.0, 60000.0, 110000.0, 140000.0, 105000.0, 150000.0]
    )

    preprocessor = SalaryPreprocessorBuilder().build(build_e3b_config())
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=1.0, random_state=42)),
        ]
    )
    pipeline.fit(df, y)
    return pipeline


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print(
            "MLFLOW_TRACKING_URI is not set. Refusing to run -- this "
            "script must only ever target a disposable CI MLflow "
            "instance, and must never fall back to an implicit or "
            "default tracking URI.",
            file=sys.stderr,
        )
        sys.exit(1)

    if "localhost" not in tracking_uri and "127.0.0.1" not in tracking_uri:
        print(
            f"Refusing to run against a non-local MLFLOW_TRACKING_URI "
            f"({tracking_uri!r}). This script must only ever target the "
            "disposable CI MLflow instance started by the Docker smoke "
            "test workflow.",
            file=sys.stderr,
        )
        sys.exit(1)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("ci_smoke_test")

    pipeline = _build_ci_smoke_model()

    # In MLflow 3.x, log_model stores artifacts under the LoggedModel entity's
    # artifact_location (mlflow-artifacts:/EXPERIMENT_ID/models/MODEL_ID/artifacts),
    # NOT under the classic run artifact path
    # (mlflow-artifacts:/EXPERIMENT_ID/RUN_ID/artifacts/model).
    #
    # model_info.model_uri is 'models:/m-MODEL_ID' -- a LoggedModel ID URI.
    # Using this as the model version source causes get_model_version_download_uri
    # to return 'models:/m-MODEL_ID', which ModelsArtifactRepository resolves via
    # client.get_logged_model(model_id).artifact_location -- the path where the
    # files actually reside.
    #
    # Using the old 'runs:/RUN_ID/model' source causes ModelsArtifactRepository to
    # call RunsArtifactRepository.download_artifacts('') which calls
    # MlflowArtifactsRepository.list_artifacts('') on the empty run artifact path
    # and returns an empty directory (no exception raised), so the model artifact
    # fallback in RunsArtifactRepository is bypassed.  The result is an empty local
    # directory with no MLmodel file -> 'No such artifact: MLmodel'.
    #
    # skops_trusted_types must list every non-sklearn type present in the
    # serialised pipeline.  The exact set is determined by
    # skops.io.get_untrusted_types(file=<saved .skops path>); the four below come
    # directly from the UntrustedTypesFoundException raised on the first smoke-test
    # run.  numpy.dtype is required because the custom transformers store fitted
    # numpy arrays as instance attributes whose dtype metadata skops serialises
    # separately.  Do NOT use trusted=True (removed in skops >= 0.10 due to
    # CVE-2024-37065) and do NOT switch to cloudpickle/pickle to bypass this check.
    _SKOPS_TRUSTED_TYPES = [
        "numpy.dtype",
        "src.components.salary_predict.salary_preprocessor_builder.SafeCategoricalTransformer",
        "src.components.salary_predict.salary_preprocessor_builder.SafeTextTransformer",
        "src.components.salary_predict.salary_preprocessor_builder.SkillMultiHotTransformer",
    ]

    with mlflow.start_run(run_name="ci_smoke_model") as run:
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            skops_trusted_types=_SKOPS_TRUSTED_TYPES,
        )
        mlflow.set_tag("model_name", "ridge")
        mlflow.set_tag("model_class_name", "Ridge")
        mlflow.set_tag("ci_smoke_test", "true")
        run_id = run.info.run_id
        # model_info.model_uri = 'models:/m-MODEL_ID' (LoggedModel ID URI).
        # This resolves to the logged model artifact_location where files
        # are actually stored, making models:/NAME@ALIAS load correctly.
        source_uri = model_info.model_uri

    client = MlflowClient()

    try:
        client.create_registered_model(REGISTERED_MODEL_NAME)
        print(f"Created registered model '{REGISTERED_MODEL_NAME}'.")
    except MlflowException as e:
        error_code = getattr(e, "error_code", "")
        if error_code == "RESOURCE_ALREADY_EXISTS" or "RESOURCE_ALREADY_EXISTS" in str(
            e
        ):
            print(
                f"Registered model '{REGISTERED_MODEL_NAME}' already exists "
                "in this CI MLflow instance; reusing it."
            )
        else:
            raise

    model_version = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=source_uri,
        run_id=run_id,
        tags={
            "model_name": "ridge",
            "model_class_name": "Ridge",
            "ci_smoke_test": "true",
        },
    )

    timeout_seconds = 120
    start_time = time.monotonic()

    print(f"Waiting for model version {model_version.version} to reach READY status...")
    while True:
        current_version = client.get_model_version(
            REGISTERED_MODEL_NAME, model_version.version
        )
        if current_version.status == "READY":
            print(f"Model version {model_version.version} is READY.")
            break
        elif current_version.status in ("FAILED_REGISTRATION", "FAILED"):
            raise RuntimeError(
                f"Model version registration failed with status: {current_version.status}"
            )

        if time.monotonic() - start_time > timeout_seconds:
            raise TimeoutError(
                f"Model version {model_version.version} did not become READY within {timeout_seconds} seconds."
            )

        time.sleep(2)

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, MODEL_ALIAS, model_version.version
    )

    print(
        f"Registered CI smoke-test model: {REGISTERED_MODEL_NAME} "
        f"v{model_version.version}, alias {MODEL_ALIAS!r} -> "
        f"v{model_version.version}."
    )


if __name__ == "__main__":
    main()
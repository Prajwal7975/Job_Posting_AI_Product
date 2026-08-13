from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SalaryServingConfig:

    # ---------------------------------------------------------
    # MLflow
    # ---------------------------------------------------------

    tracking_uri: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        "sqlite:///mlflow.db",
    )

    registered_model_name: str = os.getenv(
        "SALARY_REGISTERED_MODEL_NAME",
        "salary_prediction_model",
    )

    model_alias: str = os.getenv(
        "SALARY_MODEL_ALIAS",
        "production",
    )

    # ---------------------------------------------------------
    # API
    # ---------------------------------------------------------

    api_host: str = os.getenv(
        "API_HOST",
        "0.0.0.0",
    )

    api_port: int = int(
        os.getenv(
            "API_PORT",
            "8000",
        )
    )

    # ---------------------------------------------------------
    # Runtime
    # ---------------------------------------------------------

    reload: bool = (
        os.getenv(
            "API_RELOAD",
            "false",
        ).lower()
        == "true"
    )

    @property
    def model_uri(self) -> str:
        """
        MLflow alias-based model URI.

        Example:

        models:/salary_prediction_model@production
        """

        return f"models:/{self.registered_model_name}" f"@{self.model_alias}"

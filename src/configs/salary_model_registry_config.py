from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

CONFIG_VERSION = "1.0"


@dataclass(frozen=True)
class SalaryModelRegistryConfig:
    """
    Configuration contract for registering a validated salary model.
    """

    config_version: str = CONFIG_VERSION

    # --------------------------------------------------------------
    # Registry
    # --------------------------------------------------------------

    registered_model_name: str = "SalaryPredictionModel"

    # --------------------------------------------------------------
    # MLflow
    # --------------------------------------------------------------

    tracking_uri: Optional[str] = None

    # --------------------------------------------------------------
    # Promotion
    # --------------------------------------------------------------

    production_alias: str = "production"

    # --------------------------------------------------------------
    # Behavior
    # --------------------------------------------------------------

    allow_existing_model: bool = True

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    def __post_init__(self) -> None:

        if (
            not isinstance(
                self.registered_model_name,
                str,
            )
            or not self.registered_model_name.strip()
        ):
            raise ValueError("registered_model_name must be a " "non-empty string.")

        if self.tracking_uri is not None:

            if (
                not isinstance(
                    self.tracking_uri,
                    str,
                )
                or not self.tracking_uri.strip()
            ):
                raise ValueError("tracking_uri must be None or " "a non-empty string.")

        if (
            not isinstance(
                self.production_alias,
                str,
            )
            or not self.production_alias.strip()
        ):
            raise ValueError("production_alias must be " "a non-empty string.")

        if not isinstance(
            self.allow_existing_model,
            bool,
        ):
            raise ValueError("allow_existing_model must be a bool.")

"""
src/components/salary_predict/salary_model_registry.py

Salary Model Registry.

Responsibilities
----------------
- Register an already validated MLflow model.
- Create a model version.
- Attach useful lineage metadata.
- Optionally assign the production alias.
- Return a structured registration result.

This component does NOT:
    - train models
    - preprocess data
    - evaluate models
    - perform tuning
    - select model families
    - create predictions
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.configs.salary_model_registry_config import (
    SalaryModelRegistryConfig,
)

from src.entity.salary_model_registry_entity import (
    SalaryModelRegistryResult,
)

from src.exception import CustomException
from src.logger import logging


class SalaryModelRegistry:
    """
    Registers validated salary models in MLflow Model Registry.
    """

    def __init__(
        self,
        config: Optional[SalaryModelRegistryConfig] = None,
    ) -> None:

        self.config = config or SalaryModelRegistryConfig()

        if self.config.tracking_uri:
            mlflow.set_tracking_uri(self.config.tracking_uri)

        self.client = MlflowClient()

    # ==============================================================
    # PUBLIC API
    # ==============================================================

    def register(
        self,
        *,
        model_uri: str,
        source_run_id: Optional[str] = None,
        model_artifact_path: Optional[str] = None,
        validation_passed: bool = False,
        validation_metrics: Optional[Dict[str, float]] = None,
        test_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        promote_to_production: bool = True,
    ) -> SalaryModelRegistryResult:

        validation_metrics = validation_metrics or {}

        test_metrics = test_metrics or {}

        # ----------------------------------------------------------
        # Safety gate
        # ----------------------------------------------------------

        if not validation_passed:

            raise ValueError(
                "Model registration blocked because " "validation_passed=False."
            )

        if not model_uri:
            raise ValueError("model_uri must be provided.")

        try:

            logging.info("=" * 70)
            logging.info("SALARY MODEL REGISTRATION STARTED")
            logging.info("=" * 70)

            logging.info(
                "Registered model : %s",
                self.config.registered_model_name,
            )

            logging.info(
                "Model URI        : %s",
                model_uri,
            )

            # ------------------------------------------------------
            # 1. Create registered model if necessary
            # ------------------------------------------------------

            self._ensure_registered_model()

            # ------------------------------------------------------
            # 2. Create model version
            # ------------------------------------------------------

            version = self.client.create_model_version(
                name=(self.config.registered_model_name),
                source=model_uri,
                run_id=source_run_id,
            )

            model_version = str(version.version)

            logging.info(
                "Registered model version: %s",
                model_version,
            )

            # ------------------------------------------------------
            # 3. Attach metadata
            # ------------------------------------------------------

            self._set_tags(
                model_version=model_version,
                source_run_id=source_run_id,
                validation_metrics=(validation_metrics),
                test_metrics=test_metrics,
                metadata=metadata,
            )

            # ------------------------------------------------------
            # 4. Promote using alias
            # ------------------------------------------------------

            alias_updated = False

            if promote_to_production:

                self.client.set_registered_model_alias(
                    name=(self.config.registered_model_name),
                    alias=(self.config.production_alias),
                    version=model_version,
                )

                alias_updated = True

                logging.info(
                    "Production alias '%s' " "→ version %s",
                    self.config.production_alias,
                    model_version,
                )

            logging.info("SALARY MODEL REGISTRATION COMPLETED")

            return SalaryModelRegistryResult(
                success=True,
                registered_model_name=(self.config.registered_model_name),
                model_version=model_version,
                model_uri=model_uri,
                source_run_id=source_run_id,
                production_alias=(
                    self.config.production_alias if promote_to_production else None
                ),
                alias_updated=alias_updated,
                model_artifact_path=(model_artifact_path),
                validation_passed=True,
                validation_metrics=(validation_metrics),
                test_metrics=test_metrics,
            )

        except Exception as e:

            logging.exception("Salary model registration failed.")

            return SalaryModelRegistryResult(
                success=False,
                registered_model_name=(self.config.registered_model_name),
                model_uri=model_uri,
                source_run_id=source_run_id,
                model_artifact_path=(model_artifact_path),
                validation_passed=(validation_passed),
                validation_metrics=(validation_metrics),
                test_metrics=test_metrics,
                error=str(e),
            )

    # ==============================================================
    # REGISTERED MODEL
    # ==============================================================

    def _ensure_registered_model(self) -> None:

        name = self.config.registered_model_name

        try:

            self.client.get_registered_model(name)

            logging.info(
                "Registered model already exists: %s",
                name,
            )

        except MlflowException:

            if not self.config.allow_existing_model:

                raise

            self.client.create_registered_model(name=name)

            logging.info(
                "Created registered model: %s",
                name,
            )

    # ==============================================================
    # TAGS
    # ==============================================================

    def _set_tags(
        self,
        *,
        model_version: str,
        source_run_id: Optional[str],
        validation_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]],
    ) -> None:

        name = self.config.registered_model_name

        if source_run_id:

            self.client.set_model_version_tag(
                name=name,
                version=model_version,
                key="source_run_id",
                value=str(source_run_id),
            )

        self.client.set_model_version_tag(
            name=name,
            version=model_version,
            key="validation_passed",
            value="true",
        )

        for key, value in validation_metrics.items():

            self.client.set_model_version_tag(
                name=name,
                version=model_version,
                key=f"validation_{key}",
                value=str(value),
            )

        for key, value in test_metrics.items():

            self.client.set_model_version_tag(
                name=name,
                version=model_version,
                key=f"test_{key}",
                value=str(value),
            )

        if metadata:

            for key, value in metadata.items():

                self.client.set_model_version_tag(
                    name=name,
                    version=model_version,
                    key=str(key),
                    value=str(value),
                )

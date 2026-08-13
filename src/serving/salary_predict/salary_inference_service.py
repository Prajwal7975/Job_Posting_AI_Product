"""
src/serving/salary_predict/salary_inference/salary_inference_service.py

Application-level inference service for salary prediction.

Responsibilities:
    - receive validated API request
    - delegate raw feature construction
    - call the production model loader
    - inverse-transform log salary
    - construct response

Does NOT:
    - perform TF-IDF
    - perform encoding
    - perform scaling
    - load MLflow directly
"""

from __future__ import annotations

import math

import numpy as np

from src.logger import logging

from .salary_inference_feature_builder import (
    SalaryInferenceFeatureBuilder,
)

from .salary_inference_schema import (
    SalaryPredictionRequest,
    SalaryPredictionResponse,
)

from .salary_model_loader import (
    SalaryModelLoader,
)


class SalaryInferenceService:

    def __init__(
        self,
        model_loader: SalaryModelLoader,
        feature_builder: SalaryInferenceFeatureBuilder | None = None,
    ) -> None:

        self.model_loader = model_loader

        self.feature_builder = (
            feature_builder
            or SalaryInferenceFeatureBuilder()
        )

    # ==========================================================
    # PREDICT
    # ==========================================================

    def predict(
        self,
        request: SalaryPredictionRequest,
    ) -> SalaryPredictionResponse:

        logging.info(
            "Running salary prediction."
        )

        # ------------------------------------------------------
        # Build raw model features
        # ------------------------------------------------------

        features = self.feature_builder.build(
            title=request.title,
            skill_list=request.skill_list,
            formatted_experience_level=(
                request.formatted_experience_level
            ),
            company_state=request.company_state,
            company_country=request.company_country,
            top_industry=request.top_industry,
        )

        logging.info(
            "Inference feature DataFrame created successfully."
        )

        # ------------------------------------------------------
        # Model prediction
        # ------------------------------------------------------

        prediction = self.model_loader.predict(
            features
        )

        predicted_log_salary = float(
            np.asarray(prediction).reshape(-1)[0]
        )

        if not math.isfinite(
            predicted_log_salary
        ):
            raise ValueError(
                "Model returned a non-finite prediction."
            )

        # ------------------------------------------------------
        # Reverse log1p transformation
        #
        # Training:
        #
        #     log_salary = log1p(annual_salary)
        #
        # Inference:
        #
        #     annual_salary = expm1(log_salary)
        # ------------------------------------------------------

        predicted_annual_salary = float(
            np.expm1(
                predicted_log_salary
            )
        )

        if not math.isfinite(
            predicted_annual_salary
        ):
            raise ValueError(
                "Inverse-transformed salary is non-finite."
            )

        logging.info(
            "Salary prediction completed successfully. "
            "Predicted annual salary: %.2f",
            predicted_annual_salary,
        )

        # ------------------------------------------------------
        # Response
        # ------------------------------------------------------

        return SalaryPredictionResponse(
            predicted_annual_salary=round(
                predicted_annual_salary,
                2,
            ),
            predicted_log_salary=round(
                predicted_log_salary,
                6,
            ),
            model_name="ridge",
            model_alias=(
                self.model_loader.config.model_alias
            ),
            registered_model_name=(
                self.model_loader.config.registered_model_name
            ),
        )
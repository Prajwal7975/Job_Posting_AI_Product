"""
src/serving/salary_predict/salary_inference/salary_inference_feature_builder.py

Inference Feature Builder
-------------------------

Responsible for converting user/API input into the raw feature DataFrame
expected by the trained salary prediction pipeline.

This component does NOT:
    - load the MLflow model
    - perform TF-IDF
    - perform OneHotEncoding
    - scale numeric features
    - perform prediction
    - contain model-specific preprocessing logic

The fitted preprocessing pipeline stored inside the trained sklearn model
is responsible for transforming these raw features.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from src.exception import CustomException
from src.logger import logging


class SalaryInferenceFeatureBuilder:
    """
    Builds the raw feature DataFrame required by the production
    salary prediction model.

    The builder acts as the boundary between:

        User/API representation
                    ↓
        Model feature representation

    Optional user fields are allowed to be None. Missing-value handling
    is delegated to the fitted preprocessing pipeline.
    """

    # ------------------------------------------------------------------
    # Model input schema
    # ------------------------------------------------------------------

    REQUIRED_FEATURES = (
        "title",
        "skill_list",
        "formatted_experience_level",
    )

    OPTIONAL_FEATURES = (
        "company_state",
        "company_country",
        "top_industry",
    )

    DERIVED_FEATURES = (
        "skill_count",
    )

    MODEL_FEATURE_COLUMNS = (
        "title",
        "skill_list",
        "formatted_experience_level",
        "company_state",
        "company_country",
        "top_industry",
        "skill_count",
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        *,
        title: str,
        skill_list: str,
        formatted_experience_level: str,
        company_state: Optional[str] = None,
        company_country: Optional[str] = None,
        top_industry: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Build a one-row DataFrame containing the raw features expected
        by the trained salary model.

        Args:
            title:
                Job title.

            skill_list:
                Pipe-separated skills, for example:
                "Python|SQL|Docker|AWS"

            formatted_experience_level:
                Experience level used during model training.

            company_state:
                Optional company state/location.

            company_country:
                Optional company country.

            top_industry:
                Optional industry.

        Returns:
            A one-row pandas DataFrame with the exact model feature schema.

        Raises:
            ValueError:
                If required fields are missing or invalid.
        """

        try:
            logging.info(
                "Building salary inference feature DataFrame."
            )

            # ----------------------------------------------------------
            # Validate required fields
            # ----------------------------------------------------------

            title = self._validate_required_text(
                title,
                "title",
            )

            skill_list = self._validate_required_text(
                skill_list,
                "skill_list",
            )

            formatted_experience_level = self._validate_required_text(
                formatted_experience_level,
                "formatted_experience_level",
            )

            # ----------------------------------------------------------
            # Normalize optional fields
            # ----------------------------------------------------------

            company_state = self._normalize_optional_text(
                company_state
            )

            company_country = self._normalize_optional_text(
                company_country
            )

            top_industry = self._normalize_optional_text(
                top_industry
            )

            # ----------------------------------------------------------
            # Derived feature
            # ----------------------------------------------------------

            skill_count = self._calculate_skill_count(
                skill_list
            )

            # ----------------------------------------------------------
            # Construct raw model input
            # ----------------------------------------------------------

            row: Dict[str, Any] = {
                "title": title,
                "skill_list": skill_list,
                "formatted_experience_level": (
                    formatted_experience_level
                ),
                "company_state": company_state,
                "company_country": company_country,
                "top_industry": top_industry,
                "skill_count": skill_count,
            }

            features = pd.DataFrame(
                [row],
                columns=list(self.MODEL_FEATURE_COLUMNS),
            )

            # ----------------------------------------------------------
            # Final schema validation
            # ----------------------------------------------------------

            self._validate_model_features(features)

            logging.info(
                "Salary inference features built successfully. "
                "Columns=%s",
                list(features.columns),
            )

            return features

        except ValueError:
            raise

        except Exception as exc:
            logging.error(
                "Failed to build salary inference features: %s",
                exc,
            )
            raise CustomException(
                exc,
                __import__("sys"),
            ) from exc

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_required_text(
        value: Any,
        field_name: str,
    ) -> str:
        """
        Validate a required textual input.
        """

        if value is None:
            raise ValueError(
                f"{field_name} is required."
            )

        if not isinstance(value, str):
            raise ValueError(
                f"{field_name} must be a string."
            )

        value = value.strip()

        if not value:
            raise ValueError(
                f"{field_name} must not be empty."
            )

        return value

    # ------------------------------------------------------------------
    # Optional field handling
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_optional_text(
        value: Optional[str],
    ) -> Optional[str]:
        """
        Normalize optional text fields.

        Empty strings are converted to None.

        Missing-value handling itself is deliberately NOT performed here.
        The fitted sklearn preprocessing pipeline handles missing values.
        """

        if value is None:
            return None

        if not isinstance(value, str):
            raise ValueError(
                "Optional categorical fields must be strings or None."
            )

        value = value.strip()

        return value if value else None

    # ------------------------------------------------------------------
    # Derived features
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_skill_count(
        skill_list: str,
    ) -> int:
        """
        Calculate the number of unique skills in a pipe-separated
        skill string.

        Example:

            "Python|SQL|Docker|Python"

        becomes:

            3
        """

        skills = {
            skill.strip().lower()
            for skill in skill_list.split("|")
            if skill.strip()
        }

        return len(skills)

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------

    @classmethod
    def _validate_model_features(
        cls,
        features: pd.DataFrame,
    ) -> None:
        """
        Ensure the resulting DataFrame exactly contains the features
        expected by the production salary model.
        """

        expected = set(
            cls.MODEL_FEATURE_COLUMNS
        )

        actual = set(
            features.columns
        )

        missing = expected - actual
        unexpected = actual - expected

        if missing:
            raise ValueError(
                "Inference feature DataFrame is missing "
                f"required columns: {sorted(missing)}"
            )

        if unexpected:
            raise ValueError(
                "Inference feature DataFrame contains "
                f"unexpected columns: {sorted(unexpected)}"
            )

        if list(features.columns) != list(
            cls.MODEL_FEATURE_COLUMNS
        ):
            raise ValueError(
                "Inference feature columns are not in the "
                "expected order."
            )

        if len(features) != 1:
            raise ValueError(
                "Salary inference currently expects exactly "
                "one prediction row."
            )
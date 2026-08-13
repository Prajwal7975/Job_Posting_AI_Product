
from __future__ import annotations
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class SalaryPredictionRequest(BaseModel):
    """
    User-facing request schema.

    Only the information that a normal user is expected to know
    is required.

    Location and industry are optional because the trained
    preprocessing pipeline already supports missing categorical values.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    # ---------------------------------------------------------
    # Required user inputs
    # ---------------------------------------------------------

    title: str = Field(
        ...,
        min_length=1,
        description="Job title.",
        examples=["Machine Learning Engineer"],
    )

    skill_list: str = Field(
        ...,
        min_length=1,
        description=(
            "Pipe-separated skills."
        ),
        examples=[
            "Python|Machine Learning|SQL|Docker|AWS"
        ],
    )

    formatted_experience_level: str = Field(
        ...,
        min_length=1,
        description="Experience level.",
        examples=["Mid-Senior level"],
    )

    # ---------------------------------------------------------
    # Optional user inputs
    # ---------------------------------------------------------

    company_state: Optional[str] = Field(
        default=None,
        description=(
            "Optional company state or region."
        ),
        examples=["CA"],
    )

    company_country: Optional[str] = Field(
        default=None,
        description=(
            "Optional company country."
        ),
        examples=["US"],
    )

    top_industry: Optional[str] = Field(
        default=None,
        description=(
            "Optional industry."
        ),
        examples=["Technology"],
    )

    # ---------------------------------------------------------
    # Derived feature
    # ---------------------------------------------------------

    skill_count: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Optional skill count. If omitted, it is calculated "
            "automatically from skill_list."
        ),
    )


class SalaryPredictionResponse(BaseModel):
    """
    Prediction response returned to the frontend/client.
    """

    predicted_annual_salary: float
    predicted_log_salary: float

    model_name: str
    model_alias: str
    registered_model_name: str
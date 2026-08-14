from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class SalaryPredictionRequest(BaseModel):
    """
    User-facing request schema for salary prediction.

    The client supplies only raw business-level information.
    Derived model features are generated server-side.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    title: str = Field(
        ...,
        min_length=1,
        description="Job title.",
        examples=["Machine Learning Engineer"],
    )

    skill_list: str = Field(
        ...,
        min_length=1,
        description="Pipe-separated skills.",
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

    company_state: Optional[str] = Field(
        default=None,
        description="Optional company state or region.",
        examples=["CA"],
    )

    company_country: Optional[str] = Field(
        default=None,
        description="Optional company country.",
        examples=["US"],
    )

    top_industry: Optional[str] = Field(
        default=None,
        description="Optional industry.",
        examples=["Technology"],
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
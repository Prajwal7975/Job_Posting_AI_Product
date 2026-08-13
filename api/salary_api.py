from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.logger import logging

from src.serving.salary_predict.salary_inference_schema import (
    SalaryPredictionRequest,
    SalaryPredictionResponse,
)

from src.serving.salary_predict.salary_inference_service import (
    SalaryInferenceService,
)

from src.serving.salary_predict.salary_model_loader import (
    SalaryModelLoader,
)

from src.serving.salary_predict.salary_model_config import (
    SalaryServingConfig,
)


# ==========================================================
# GLOBAL SERVICES
# ==========================================================

serving_config = SalaryServingConfig()

model_loader = SalaryModelLoader(
    config=serving_config
)

inference_service = SalaryInferenceService(
    model_loader=model_loader
)


# ==========================================================
# APPLICATION LIFECYCLE
# ==========================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    logging.info(
        "Starting Salary Prediction API."
    )

    try:
        model_loader.load()

        logging.info(
            "Production salary model loaded during API startup."
        )

    except Exception as exc:

        logging.exception(
            "Failed to load production salary model: %s",
            exc,
        )

    yield

    logging.info(
        "Shutting down Salary Prediction API."
    )


# ==========================================================
# FASTAPI APPLICATION
# ==========================================================

app = FastAPI(
    title="LinkedIn Job Intelligence - Salary Prediction API",
    description=(
        "Production salary prediction API backed by "
        "MLflow Model Registry."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# ROOT
# ==========================================================

@app.get(
    "/",
    tags=["System"],
)
def root() -> dict[str, str]:

    return {
        "service": "salary-prediction-api",
        "status": "running",
        "model": (
            serving_config.registered_model_name
        ),
        "alias": (
            serving_config.model_alias
        ),
    }


# ==========================================================
# HEALTH
# ==========================================================

@app.get(
    "/health",
    tags=["System"],
)
def health() -> dict[str, object]:

    return {
        "status": (
            "healthy"
            if model_loader.is_loaded
            else "degraded"
        ),
        "model_loaded": model_loader.is_loaded,
        "registered_model_name": (
            serving_config.registered_model_name
        ),
        "model_alias": (
            serving_config.model_alias
        ),
    }


# ==========================================================
# PREDICT
# ==========================================================

@app.post(
    "/api/v1/predict",
    response_model=SalaryPredictionResponse,
    tags=["Salary Prediction"],
)
def predict(
    request: SalaryPredictionRequest,
) -> SalaryPredictionResponse:

    logging.info(
        "Received salary prediction request."
    )

    try:

        return inference_service.predict(
            request
        )

    except ValueError as exc:

        logging.exception(
            "Salary prediction validation failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logging.exception(
            "Salary prediction failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc
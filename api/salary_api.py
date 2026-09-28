from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, status
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

model_loader = SalaryModelLoader(config=serving_config)

inference_service = SalaryInferenceService(model_loader=model_loader)


# ==========================================================
# APPLICATION LIFECYCLE
# ==========================================================


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:

    logging.info(
        "Starting Salary Prediction API (app_env=%s).",
        serving_config.app_env,
    )

    try:

        # --------------------------------------------------
        # Load production model during startup
        # --------------------------------------------------

        model_loader.load()

        logging.info("Production salary model loaded successfully.")

        logging.info(
            "Registered model: %s",
            serving_config.registered_model_name,
        )

        logging.info(
            "Model alias: %s",
            serving_config.model_alias,
        )

    except Exception:

        logging.exception("Failed to load production salary model.")

        # --------------------------------------------------
        # Fail fast
        #
        # The API should not start if the production model
        # cannot be loaded.
        # --------------------------------------------------

        raise

    yield

    logging.info("Shutting down Salary Prediction API.")


# ==========================================================
# FASTAPI APPLICATION
# ==========================================================

app = FastAPI(
    title="LinkedIn Job Intelligence - Salary Prediction API",
    description=(
        "Production salary prediction API backed by " "MLflow Model Registry."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ==========================================================
# CORS
#
# Origins come from CORS_ALLOWED_ORIGINS (comma-separated), with a
# local-development-friendly default when unset -- see
# salary_model_config.py. Wildcard origins are rejected at config
# validation time, before the app even starts.
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(serving_config.cors_allowed_origins),
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
    }


# ==========================================================
# HEALTH  (liveness: "the process is up")
#
# Deliberately independent of model state or any MLflow call -- a
# health check should never require a network round-trip. Use
# /readiness to ask whether the API can actually serve predictions.
# ==========================================================


@app.get(
    "/health",
    tags=["System"],
)
def health() -> dict[str, str]:

    return {
        "status": "alive",
        "service": "salary-prediction-api",
    }


# ==========================================================
# READINESS  (readiness: "loaded and able to serve predictions")
#
# Cheap -- reads the already-loaded, in-memory model's cached state
# and metadata; makes no MLflow network call of its own. Because the
# lifespan above fails fast on a load error, a running process should
# always report ready=True; this still checks is_loaded defensively
# rather than assuming it, and surfaces model_version when available.
# ==========================================================


@app.get(
    "/readiness",
    tags=["System"],
)
def readiness() -> dict[str, object]:

    metadata = model_loader.metadata

    if not model_loader.is_loaded:

        return {
            "status": "not_ready",
            "model_loaded": False,
            "registered_model_name": (serving_config.registered_model_name),
            "model_alias": (serving_config.model_alias),
        }

    return {
        "status": "ready",
        "model_loaded": True,
        "registered_model_name": (serving_config.registered_model_name),
        "model_alias": (serving_config.model_alias),
        "model_version": (metadata.model_version if metadata is not None else None),
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

    logging.info("Received salary prediction request.")

    try:

        prediction = inference_service.predict(request)

        logging.info("Salary prediction request completed successfully.")

        return prediction

    except ValueError as exc:

        logging.warning(
            "Salary prediction validation failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception:

        logging.exception("Salary prediction failed unexpectedly.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed.",
        ) from None

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

# ==============================================================
# Environment variable names (single source of truth for names)
# ==============================================================

ENV_APP_ENV = "APP_ENV"
ENV_MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"
ENV_REGISTERED_MODEL_NAME = "SALARY_REGISTERED_MODEL_NAME"
ENV_MODEL_ALIAS = "SALARY_MODEL_ALIAS"
ENV_API_HOST = "API_HOST"
ENV_API_PORT = "API_PORT"
ENV_API_RELOAD = "API_RELOAD"
ENV_CORS_ALLOWED_ORIGINS = "CORS_ALLOWED_ORIGINS"

# Local-development-only fallback. NEVER used when APP_ENV=production --
# see SalaryServingConfig.__post_init__ and .resolved_tracking_uri.
_LOCAL_DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"

# Local-development-only default CORS origins (typical dev server ports
# for the existing frontend). Production is expected to set
# CORS_ALLOWED_ORIGINS explicitly.
_LOCAL_DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000,http://127.0.0.1:3000,"
    "http://localhost:5500,http://127.0.0.1:5500"
)

_PRODUCTION_ENV_VALUES = {"production", "prod"}


# ==============================================================
# Env-reading helpers (each does its own type coercion so the
# dataclass fields below are already correctly typed)
# ==============================================================


def _env_app_env() -> str:
    return os.getenv(ENV_APP_ENV, "local").strip().lower()


def _env_tracking_uri() -> Optional[str]:
    """
    None means "not explicitly configured". Whether that's an error
    depends on app_env, which is why this returns Optional instead of
    substituting the local default itself -- see __post_init__ and
    resolved_tracking_uri.
    """
    raw = os.getenv(ENV_MLFLOW_TRACKING_URI)
    return raw.strip() if raw and raw.strip() else None


def _env_registered_model_name() -> str:
    return os.getenv(ENV_REGISTERED_MODEL_NAME, "salary_prediction_model").strip()


def _env_model_alias() -> str:
    return os.getenv(ENV_MODEL_ALIAS, "production").strip()


def _env_api_host() -> str:
    return os.getenv(ENV_API_HOST, "0.0.0.0").strip()


def _env_api_port() -> int:
    raw = os.getenv(ENV_API_PORT, "8000")
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{ENV_API_PORT} must be an integer, got {raw!r}."
        ) from exc


def _env_api_reload() -> bool:
    return os.getenv(ENV_API_RELOAD, "false").strip().lower() == "true"


def _env_cors_allowed_origins() -> Tuple[str, ...]:
    raw = os.getenv(ENV_CORS_ALLOWED_ORIGINS, _LOCAL_DEFAULT_CORS_ORIGINS)
    return tuple(origin.strip() for origin in raw.split(",") if origin.strip())


# ==============================================================
# Config
# ==============================================================


@dataclass(frozen=True)
class SalaryServingConfig:
    """
    Serving-layer configuration, read from environment variables at
    construction time (via default_factory, so a fresh instance always
    reflects the current environment -- important for tests that
    monkeypatch env vars).

    Production safety: when APP_ENV=production, MLFLOW_TRACKING_URI is
    REQUIRED. This config will never silently fall back to a local
    sqlite store in production -- see resolved_tracking_uri.
    """

    app_env: str = field(default_factory=_env_app_env)

    # Optional/unresolved on purpose -- see resolved_tracking_uri.
    tracking_uri: Optional[str] = field(default_factory=_env_tracking_uri)

    registered_model_name: str = field(default_factory=_env_registered_model_name)
    model_alias: str = field(default_factory=_env_model_alias)

    api_host: str = field(default_factory=_env_api_host)
    api_port: int = field(default_factory=_env_api_port)
    reload: bool = field(default_factory=_env_api_reload)

    cors_allowed_origins: Tuple[str, ...] = field(
        default_factory=_env_cors_allowed_origins
    )

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------

    def __post_init__(self) -> None:
        if not self.registered_model_name:
            raise ValueError(
                f"{ENV_REGISTERED_MODEL_NAME} must not be empty."
            )

        if not self.model_alias:
            raise ValueError(f"{ENV_MODEL_ALIAS} must not be empty.")

        if self.is_production and not self.tracking_uri:
            raise ValueError(
                f"{ENV_MLFLOW_TRACKING_URI} is required when "
                f"{ENV_APP_ENV}={self.app_env!r}. Refusing to silently "
                "fall back to a local sqlite MLflow store in production."
            )

        if not (1 <= self.api_port <= 65535):
            raise ValueError(
                f"{ENV_API_PORT} must be between 1 and 65535, "
                f"got {self.api_port}."
            )

        if not self.cors_allowed_origins:
            raise ValueError(
                f"{ENV_CORS_ALLOWED_ORIGINS} resolved to no origins; at "
                "least one allowed origin must be configured."
            )

        if "*" in self.cors_allowed_origins:
            raise ValueError(
                f"{ENV_CORS_ALLOWED_ORIGINS} must not include '*' -- "
                "wildcard CORS is not permitted."
            )

    # ----------------------------------------------------------
    # Derived / resolved values
    # ----------------------------------------------------------

    @property
    def is_production(self) -> bool:
        return self.app_env in _PRODUCTION_ENV_VALUES

    @property
    def resolved_tracking_uri(self) -> str:
        """
        The tracking URI actually used at runtime. Only ever falls back
        to the local sqlite default outside production -- __post_init__
        guarantees `tracking_uri` is non-empty whenever `is_production`
        is True, so this never silently masks a missing production
        configuration.
        """
        return self.tracking_uri or _LOCAL_DEFAULT_TRACKING_URI

    @property
    def model_uri(self) -> str:
        """
        Resolve the production model using an MLflow alias.

        Example:

            models:/salary_prediction_model@production
        """
        return f"models:/{self.registered_model_name}@{self.model_alias}"
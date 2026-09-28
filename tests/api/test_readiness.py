"""
tests/api/test_readiness.py

Tests for GET /readiness on the real FastAPI app (api/salary_api.py).

CHANGED: a dedicated /readiness endpoint now exists, separate from
/health (see test_health.py). /readiness answers "is the API able to
serve predictions right now" by reading the already-loaded, in-memory
model's cached state -- it makes no MLflow network call of its own.

Because the app's `lifespan` handler calls `model_loader.load()`
synchronously at startup and re-raises on failure (fail-fast: the app
never finishes starting without a model), a *live* TestClient can only
ever observe `model_loaded=True` through normal startup. The "not ready"
response shape is still real, reachable code (the `if not
model_loader.is_loaded:` branch in `readiness()`), so it's exercised here
by resetting model state on an already-started client rather than by
calling the route function directly -- unlike the previous version of
this file, a real GET request now suffices since /readiness is a real
endpoint.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.api


class TestReadinessWhenModelLoaded:
    def test_readiness_returns_200(self, client):
        response = client.get("/readiness")
        assert response.status_code == 200

    def test_ready_response_structure_and_values(self, client, api_module):
        body = client.get("/readiness").json()
        assert set(body.keys()) == {
            "status",
            "model_loaded",
            "registered_model_name",
            "model_alias",
            "model_version",
        }
        assert body["status"] == "ready"
        assert body["model_loaded"] is True
        assert (
            body["registered_model_name"]
            == api_module.serving_config.registered_model_name
        )
        assert body["model_alias"] == api_module.serving_config.model_alias
        # From the fixture's SalaryModelMetadata -- see conftest.py.
        assert body["model_version"] == "3"


class TestReadinessWhenModelNotLoaded:
    def test_not_ready_response_shape(self, client, api_module):
        # Reset state on an already-started client -- readiness() reads
        # current in-memory state on every call, no restart needed.
        api_module.model_loader._model = None
        api_module.model_loader._metadata = None

        response = client.get("/readiness")
        body = response.json()

        assert response.status_code == 200  # readiness reports state, doesn't error
        assert body["status"] == "not_ready"
        assert body["model_loaded"] is False
        assert "model_version" not in body
        assert (
            body["registered_model_name"]
            == api_module.serving_config.registered_model_name
        )
        assert body["model_alias"] == api_module.serving_config.model_alias


class TestStartupFailureBehavior:
    def test_app_refuses_to_start_when_model_cannot_be_loaded(
        self, monkeypatch, api_module
    ):
        api_module.model_loader._model = None
        api_module.model_loader._metadata = None

        def _raise_load_error():
            raise RuntimeError(
                f"Unable to load the registered salary model from URI: "
                f"{api_module.serving_config.model_uri}"
            )

        monkeypatch.setattr(api_module.model_loader, "load", _raise_load_error)

        with pytest.raises(RuntimeError):
            with TestClient(api_module.app):
                pass  # lifespan startup must raise before this body ever runs

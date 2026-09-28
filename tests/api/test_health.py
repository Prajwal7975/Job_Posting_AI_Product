"""
tests/api/test_health.py

Tests for GET /health and GET / on the real FastAPI app
(api/salary_api.py), via TestClient.

CHANGED: /health is now a pure liveness check ("the process is up") and
no longer touches model state at all -- that's what /readiness is for
(see test_readiness.py). This is an intentional behavior change per the
serving-layer production-readiness work: a liveness probe should never
depend on model state or any MLflow call. The old model-dependent
assertions moved to test_readiness.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.api


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_response_body(self, client):
        assert client.get("/").json() == {
            "service": "salary-prediction-api",
            "status": "running",
        }


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        body = client.get("/health").json()
        assert set(body.keys()) == {"status", "service"}

    def test_health_reports_alive(self, client):
        body = client.get("/health").json()
        assert body["status"] == "alive"
        assert body["service"] == "salary-prediction-api"

    def test_health_does_not_depend_on_model_state(self, client, api_module):
        # Even with the model "unloaded", liveness must still report
        # alive -- health is about the process, not the model.
        api_module.model_loader._model = None
        api_module.model_loader._metadata = None

        body = client.get("/health").json()
        assert body["status"] == "alive"

    def test_health_field_types(self, client):
        body = client.get("/health").json()
        assert isinstance(body["status"], str)
        assert isinstance(body["service"], str)

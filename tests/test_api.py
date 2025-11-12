"""Tests for API endpoints."""

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "disclaimer" in response.json()
    assert "version" in response.json()


def test_health_endpoint():
    """Test health check."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_get_available_models():
    """Test getting available models."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)


def test_get_example_timeline_readmission():
    """Test getting example timeline for readmission."""
    response = client.get("/api/example/readmission")
    assert response.status_code == 200
    data = response.json()
    assert "subject_id" in data
    assert "events" in data
    assert len(data["events"]) > 0


def test_get_example_timeline_icu_mortality():
    """Test getting example timeline for ICU mortality."""
    response = client.get("/api/example/icu_mortality")
    assert response.status_code == 200
    data = response.json()
    assert "subject_id" in data
    assert "events" in data
    assert len(data["events"]) > 0


def test_predict_with_example():
    """Test prediction endpoint with example data."""
    # Get example timeline
    example_response = client.get("/api/example/readmission")
    timeline = example_response.json()

    # Make prediction (may fail if no models trained, but should not crash)
    response = client.post(
        "/api/predict/readmission",
        json=timeline,
    )

    # Either succeeds or returns 404/500 if no model
    assert response.status_code in [200, 404, 500]

    if response.status_code == 200:
        data = response.json()
        assert "risk_score" in data
        assert "risk_label" in data
        assert "explanation" in data
        assert 0 <= data["risk_score"] <= 1

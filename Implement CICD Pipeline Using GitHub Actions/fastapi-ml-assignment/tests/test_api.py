import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint(client):
    response = client.get("/health")
    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "healthy"
    assert result["model_loaded"] is True


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    result = response.json()
    assert response.status_code == 200
    assert "model_type" in result
    assert "model_version" in result
    assert "feature_names" in result
    assert "target_names" in result
    assert "test_accuracy" in result
    assert len(result["target_names"]) == 3


def test_valid_prediction(client):
    request_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post(
        "/predict",
        json=request_data
    )
    result = response.json()
    assert response.status_code == 200
    assert result["predicted_species"] in [
        "setosa",
        "versicolor",
        "virginica"
    ]
    assert 0 <= result["confidence"] <= 1
    assert "probabilities" in result
    assert result["model_version"] == "1.0.0"


def test_negative_feature_value(client):
    request_data = {
        "sepal_length": -5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post(
        "/predict",
        json=request_data
    )
    assert response.status_code == 422


def test_missing_feature(client):
    request_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
    }
    response = client.post(
        "/predict",
        json=request_data
    )
    assert response.status_code == 422


def test_incorrect_data_type(client):
    request_data = {
        "sepal_length": "invalid_number",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post(
        "/predict",
        json=request_data
    )
    assert response.status_code == 422


def test_unknown_endpoint(client):
    response = client.get("/unknown")
    assert response.status_code == 404

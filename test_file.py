import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    """Test health check root route."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_prediction_validation():
    """Test response validation with invalid input length."""
    response = client.post("/predict", json={"features": [1.0, 2.0]})
    assert response.status_code == 400
from fastapi.testclient import TestClient
from .app import app 
import numpy as np

client = TestClient(app)

def test_predict_endpoint_success():
    response = client.post(
        "/predict/aapl",
        json={"data": [150.0] * 60} 
    )
    assert response.status_code == 200
    data = response.json()
    assert data["stock_symbol"] == "aapl"
    assert "prediction" in data
    assert isinstance(data["prediction"], float)

def test_predict_endpoint_not_found():
    # Test for a stock symbol that doesn't exist
    response = client.post(
        "/predict/nonexistentstock",
        json={"data": [150.0] * 60}
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "Model or scaler for nonexistentstock not found."}
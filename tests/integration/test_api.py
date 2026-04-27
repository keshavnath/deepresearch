import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_research_endpoint_invalid_input():
    # Test empty query
    response = client.post("/research", json={"query": ""})
    assert response.status_code == 400
    
    # Test missing payload
    response = client.post("/research", json={})
    assert response.status_code == 422

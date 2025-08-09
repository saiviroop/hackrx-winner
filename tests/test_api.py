import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_query_endpoint():
    response = client.post(
        "/api/v1/query",
        json={
            "query": "What is the coverage limit for hospitalization?",
            "query_type": "coverage",
            "enable_rerank": True
        }
    )
    assert response.status_code == 200
    assert "answer" in response.json()

def test_invalid_query():
    response = client.post(
        "/api/v1/query",
        json={"query": ""}
    )
    assert response.status_code == 422
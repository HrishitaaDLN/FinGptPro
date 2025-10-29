from fastapi.testclient import TestClient
from api.server import app

client = TestClient(app)

def test_query_endpoint(monkeypatch):
    mock_result = {"query": "interest rates", "answer": "Rates are stable.", "context": []}
    monkeypatch.setattr("api.routes.rag_routes.pipeline.query", lambda q: mock_result)

    response = client.post("/rag/query", json={"query": "interest rates"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data

from unittest.mock import MagicMock
from app.retriever import Retriever

def test_search_returns_results(monkeypatch):
    retriever = Retriever()

    # Mock SentenceTransformer.encode
    mock_embed = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
    retriever.model.encode = mock_embed

    # Mock QdrantClient.search
    mock_search = MagicMock(return_value=[
        MagicMock(payload={"sentence": "The market is stable", "sentiment": "neutral"}, score=0.95)
    ])
    retriever.client.search = mock_search

    results = retriever.search("market sentiment")
    assert isinstance(results, list)
    assert "sentence" in results[0]
    assert results[0]["sentiment"] == "neutral"

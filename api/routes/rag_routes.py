from fastapi import APIRouter
from app.pipeline import RAGPipeline

router = APIRouter(prefix="/rag", tags=["RAG"])
pipeline = RAGPipeline()

@router.post("/query")
def query_rag(payload: dict):
    query = payload.get("query", "")
    if not query:
        return {"error": "Query text missing"}
    result = pipeline.query(query)
    return result

from fastapi import APIRouter
from app.retriever import Retriever

router = APIRouter(prefix="/ingest", tags=["Ingestion"])
retriever = Retriever()

@router.post("/build")
def build_index():
    retriever.build_index()
    return {"status": "Index built successfully!"}

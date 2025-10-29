import os

# 🔐 Secure config (reads from Streamlit Secrets or environment vars)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "finrag_docs")
DATA_PATH = os.getenv("DATA_PATH", "data/financial_phrasebank_50agree.csv")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cpu"  # Streamlit Cloud doesn't support GPU

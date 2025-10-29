import os
import streamlit as st

# =====================================
#  Configuration — Secrets + Defaults
# =====================================

def get_secret(key: str, default=None):
    """Helper to read Streamlit secrets safely."""
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

# -----------------------------
#  API Keys & Model Settings
# -----------------------------
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
QDRANT_URL = get_secret("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = get_secret("QDRANT_API_KEY", None)
COLLECTION_NAME = get_secret("COLLECTION_NAME", "finrag_docs")
DATA_PATH = get_secret("DATA_PATH", "data/financial_phrasebank_50agree.csv")

# -----------------------------
#  Embedding Model Settings
# -----------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cpu"

# -----------------------------
#  LLM Model Setting
# -----------------------------
LLM_MODEL = get_secret("LLM_MODEL", "gemini-2.5-flash")

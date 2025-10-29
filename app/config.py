import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# === Qdrant Configuration ===
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "finrag_docs")

# === Embedding & Model Configuration ===
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # fast, accurate
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")   # Gemini flash = super fast
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Data Source ===
DATA_PATH = os.getenv("DATA_PATH", "data/financial_phrasebank_50agree.csv")

# === Gemini API Key ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found. Please set it in your .env file or environment variables.")

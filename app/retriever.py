from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import pandas as pd
from functools import lru_cache
from app.config import *
from textblob import TextBlob
import re


# ======================================
# 🔹 Embedding Encoder (cached for speed)
# ======================================
@lru_cache(maxsize=1)
def get_encoder():
    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    model.max_seq_length = 256
    return model


# ======================================
# 🔹 Label Normalization Helper
# ======================================
def _normalize_label(lbl: str) -> str:
    """Map noisy or inconsistent labels to {positive, negative, neutral}."""
    if not lbl:
        return "neutral"
    s = re.sub(r"[^a-z]", "", str(lbl).lower())
    if s.startswith("pos"):
        return "positive"
    if s.startswith("neg"):
        return "negative"
    if s.startswith("neu"):
        return "neutral"
    if "posit" in s:
        return "positive"
    if "negat" in s:
        return "negative"
    if "neutral" in s or "neutr" in s:
        return "neutral"
    return "neutral"


# ======================================
# 🔹 Retriever Class
# ======================================
class Retriever:
    def __init__(self):
        """Initialize Qdrant connection and model."""
        try:
            # ✅ Use secure Qdrant Cloud endpoint + API key
            self.client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
            )
            print(f"🔗 Connected to Qdrant Cloud at {QDRANT_URL}")
        except Exception as e:
            print(f"❌ Qdrant connection failed: {e}")
            raise

        self.model = get_encoder()
        self.collection = COLLECTION_NAME
        self._init_collection()

    # ------------------------------
    # 🗂️ Ensure collection existence
    # ------------------------------
    def _init_collection(self):
        """Ensure collection exists with correct vector size."""
        vector_size = self.model.get_sentence_embedding_dimension()
        try:
            collections = [c.name for c in self.client.get_collections().collections]
        except Exception as e:
            print(f"⚠️ Could not fetch collections: {e}")
            collections = []

        if self.collection not in collections:
            print(f"📁 Creating new collection: {self.collection}")
            try:
                self.client.recreate_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
                print(f"✅ Created collection '{self.collection}' successfully.")
            except Exception as e:
                print(f"❌ Error creating collection: {e}")
        else:
            print(f"✅ Collection '{self.collection}' already exists")

    # ------------------------------
    # 🔄 Force clean collection
    # ------------------------------
    def _recreate_collection_hard(self):
        """Always start from a clean slate before indexing."""
        vector_size = self.model.get_sentence_embedding_dimension()
        try:
            self.client.delete_collection(self.collection)
            print(f"🧹 Old collection '{self.collection}' deleted.")
        except Exception as e:
            print(f"⚠️ Could not delete old collection: {e}")
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    # ------------------------------
    # ⚡ Build / Rebuild Index
    # ------------------------------
    def build_index(self, data_path=DATA_PATH):
        """
        Build or rebuild embeddings index from CSV data.
        - Loads Financial PhraseBank
        - Normalizes labels
        - Uploads embeddings + payloads to Qdrant
        """
        print(f"📂 Loading dataset from: {data_path}")
        df = pd.read_csv(data_path)
        df.columns = [c.strip().lower() for c in df.columns]
        print("📊 Columns detected:", df.columns.tolist())

        # Detect column layout
        if "sentence" in df.columns and "label" in df.columns:
            raw_labels = df["sentence"].astype(str).tolist()
            sentences = df["label"].astype(str).tolist()
        else:
            # fallback: find sentence and sentiment columns
            sentence_col = next((c for c in df.columns if "sentence" in c or "text" in c), None)
            if not sentence_col:
                raise ValueError("❌ No sentence/text column found.")
            label_col = next((c for c in df.columns if "sentiment" in c or "label" in c), None)
            if label_col:
                raw_labels = df[label_col].astype(str).tolist()
            else:
                print("⚠️ No sentiment column found — auto-labeling with TextBlob.")
                raw_labels = [self.auto_sentiment(t) for t in df[sentence_col].astype(str).tolist()]
            sentences = df[sentence_col].astype(str).tolist()

        # Normalize sentiment labels
        sentiments = [_normalize_label(l) for l in raw_labels]
        pre_counts = pd.Series(sentiments).value_counts()
        print("📊 (pre-upload) sentiment distribution:")
        print(pre_counts)

        # Reset collection (fresh start)
        self._recreate_collection_hard()

        # Generate embeddings
        print("🧠 Generating embeddings (this may take a minute)...")
        embeddings = self.model.encode(
            sentences,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        # Prepare payload
        payload = [{"sentence": s, "sentiment": sentiments[i]} for i, s in enumerate(sentences)]

        # Upload to Qdrant Cloud
        print(f"🚀 Uploading {len(sentences)} sentences to Qdrant Cloud...")
        try:
            self.client.upload_collection(
                collection_name=self.collection,
                vectors=embeddings,
                payload=payload,
            )
            print("✅ Upload complete.")
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            raise

        # Verify uploaded labels
        try:
            items, _ = self.client.scroll(collection_name=self.collection, limit=100)
            uniq = sorted({(it.payload.get('sentiment') or 'NA') for it in items})
            print("🔎 (post-upload) unique labels found:", uniq)
        except Exception as e:
            print(f"⚠️ Verification failed: {e}")

    # ------------------------------
    # 🔍 Search
    # ------------------------------
    def search(self, query: str, top_k=5):
        """Search most similar sentences by semantic embedding."""
        query_vector = self.model.encode([query])[0]
        try:
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=top_k,
            )
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []

        out = []
        for r in results:
            p = dict(r.payload)
            p["sentiment"] = _normalize_label(p.get("sentiment", "neutral"))
            out.append(p)
        return out

    # ------------------------------
    # 💬 Auto Sentiment (fallback)
    # ------------------------------
    def auto_sentiment(self, text):
        score = TextBlob(text).sentiment.polarity
        if score > 0.1:
            return "positive"
        elif score < -0.1:
            return "negative"
        else:
            return "neutral"

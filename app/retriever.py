from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import pandas as pd
from functools import lru_cache
from app.config import *
from textblob import TextBlob
import re


@lru_cache(maxsize=1)
def get_encoder():
    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    model.max_seq_length = 256
    return model


def _normalize_label(lbl: str) -> str:
    """Map any noisy label to {positive, negative, neutral}."""
    if not lbl:
        return "neutral"
    s = re.sub(r"[^a-z]", "", str(lbl).lower())  # keep letters only
    # handle duplicates/typos like 'positveitive', 'neutraltral'
    if s.startswith("pos"):
        return "positive"
    if s.startswith("neg"):
        return "negative"
    if s.startswith("neu"):
        return "neutral"
    # last fallback: polarity word inside
    if "posit" in s:
        return "positive"
    if "negat" in s:
        return "negative"
    if "neutral" in s or "neutr" in s:
        return "neutral"
    return "neutral"


class Retriever:
    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.model = get_encoder()
        self.collection = COLLECTION_NAME
        self._init_collection()

    def _init_collection(self):
        """Ensure collection exists with correct vector size."""
        vector_size = self.model.get_sentence_embedding_dimension()
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection not in collections:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )

    def _recreate_collection_hard(self):
        """Always start from a clean slate before indexing."""
        vector_size = self.model.get_sentence_embedding_dimension()
        # delete if exists; then recreate
        try:
            self.client.delete_collection(self.collection)
        except Exception:
            pass
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def build_index(self, data_path=DATA_PATH):
        """
        Rebuild index from dataset.
        - Wipes the collection (hard recreate)
        - Detects column layout (PhraseBank: 'sentence'=label, 'label'=text)
        - Normalizes labels before upload
        - Verifies distribution post-upload
        """
        print(f"📂 Loading dataset from: {data_path}")
        df = pd.read_csv(data_path)
        df.columns = [c.strip().lower() for c in df.columns]
        print("📊 Columns detected:", df.columns.tolist())

        # PhraseBank layout: first col sentiment, second col text
        if "sentence" in df.columns and "label" in df.columns:
            raw_labels = df["sentence"].astype(str).tolist()
            sentences = df["label"].astype(str).tolist()
        else:
            # generic fallback
            sentence_col = next((c for c in df.columns if "sentence" in c or "text" in c), None)
            if not sentence_col:
                raise ValueError("❌ No sentence/text column found.")
            label_col = next((c for c in df.columns if "sentiment" in c or "label" in c), None)
            if label_col:
                raw_labels = df[label_col].astype(str).tolist()
            else:
                # derive via TextBlob if totally missing
                print("⚠️ No sentiment column found — auto-labeling with TextBlob.")
                raw_labels = [self.auto_sentiment(t) for t in df[sentence_col].astype(str).tolist()]
            sentences = df[sentence_col].astype(str).tolist()

        # Normalize labels
        sentiments = [_normalize_label(l) for l in raw_labels]

        # Show pre-upload distribution
        pre_counts = pd.Series(sentiments).value_counts()
        print("📊 (pre-upload) sentiment distribution:")
        print(pre_counts)

        # === Hard recreate so NO stale junk remains ===
        self._recreate_collection_hard()

        # Embeddings
        print("🧠 Generating embeddings...")
        embeddings = self.model.encode(
            sentences,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        payload = [{"sentence": s, "sentiment": sentiments[i]} for i, s in enumerate(sentences)]

        print(f"🚀 Uploading {len(sentences)} sentences to Qdrant...")
        self.client.upload_collection(
            collection_name=self.collection,
            vectors=embeddings,
            payload=payload,
        )
        print("✅ Upload done.")

        # Verify post-upload (scroll few and unique labels)
        items, _ = self.client.scroll(collection_name=self.collection, limit=200)
        uniq = sorted({(it.payload.get("sentiment") or "NA") for it in items})
        print("🔎 (post-upload) unique labels found:", uniq)
        if not set(uniq).issubset({"positive", "negative", "neutral"}):
            print("⚠️ Unexpected labels present:", uniq)

    def search(self, query: str, top_k=5):
        query_vector = self.model.encode([query])[0]
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
        )
        # Canonicalize any stray labels at read time as well (belt & suspenders)
        out = []
        for r in results:
            p = dict(r.payload)
            p["sentiment"] = _normalize_label(p.get("sentiment", "neutral"))
            out.append(p)
        return out

    def auto_sentiment(self, text):
        score = TextBlob(text).sentiment.polarity
        if score > 0.1:
            return "positive"
        elif score < -0.1:
            return "negative"
        else:
            return "neutral"

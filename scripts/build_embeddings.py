from app.retriever import Retriever
import pandas as pd

if __name__ == "__main__":
    r = Retriever()
    df = pd.read_csv("data/financial_phrasebank_50agree.csv")
    sentences = df["sentence"].tolist()
    print("🧮 Generating embeddings...")
    embeddings = r.model.encode(sentences, show_progress_bar=True)
    print(f"✅ Generated {len(embeddings)} embeddings")

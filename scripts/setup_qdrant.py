from app.retriever import Retriever

if __name__ == "__main__":
    print("⚙️ Setting up Qdrant...")
    r = Retriever()
    r.build_index()
    print("✅ Qdrant setup complete!")

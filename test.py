from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
client.delete_collection("finrag_docs")
print("🗑️ Deleted old collection: finrag_docs")

from app.retriever import Retriever
from app.generator import Generator
import statistics


class RAGPipeline:
    """
    Retrieval-Augmented Generation (RAG) pipeline for FinGPT-Pro.
    Retrieves financial context via Qdrant and uses Gemini to synthesize insights.
    """

    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()

    def query(self, question: str):
        """
        Main query function (synchronous).
        Retrieves context from retriever, enriches it, and generates a Gemini-based response.
        """
        try:
            # Step 1: Retrieve similar sentences from the index
            docs = self.retriever.search(question, top_k=12)
            if not docs:
                return {
                    "query": question,
                    "context": [],
                    "answer": "⚠️ No relevant financial data found in the index. Try rebuilding or broadening your query.",
                }

            # Step 2: Analyze sentiment distribution for Gemini prompt context
            sentiments = [d.get("sentiment", "neutral").lower() for d in docs]
            sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}

            # Simple weighted sentiment summary
            dominant = max(sentiment_counts, key=sentiment_counts.get)
            sentiment_summary = f"Most retrieved sentences are {dominant} in tone ({sentiment_counts})."

            # Step 3: Generate response using Gemini
            answer = self.generator.generate(
                question,
                docs + [{"sentence": sentiment_summary, "sentiment": "meta"}],
            )

            # Step 4: Return structured response
            return {
                "query": question,
                "context": docs,
                "sentiment_summary": sentiment_summary,
                "answer": answer,
            }

        except Exception as e:
            return {
                "query": question,
                "context": [],
                "answer": f"❌ Error running query: {str(e)}",
            }

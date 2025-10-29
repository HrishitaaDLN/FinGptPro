from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class Evaluator:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def evaluate(self, query, retrieved_docs, generated_answer):
        q_vec = self.model.encode([query])
        d_vecs = self.model.encode([d["sentence"] for d in retrieved_docs])
        sim_scores = cosine_similarity(q_vec, d_vecs)[0]
        retrieval_score = float(sim_scores.mean())

        a_vec = self.model.encode([generated_answer])
        answer_quality = float(cosine_similarity(q_vec, a_vec)[0][0])

        return {
            "retrieval_score": round(retrieval_score, 3),
            "answer_quality": round(answer_quality, 3)
        }

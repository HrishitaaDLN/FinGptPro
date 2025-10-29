# app/generator.py
import os
import google.generativeai as genai
from app.config import *

# Configure Gemini with your valid key
genai.configure(api_key=GOOGLE_API_KEY)
print(f"🔑 GOOGLE_API_KEY loaded: {GOOGLE_API_KEY[:10]}********")

class Generator:
    def __init__(self, model_name=None):
        self.model_name = model_name or LLM_MODEL or "gemini-2.5-flash"
        print(f"✅ Using Gemini model: {self.model_name}")
        self.model = genai.GenerativeModel(self.model_name)

    def generate(self, question, context):
        """Generate an answer using Gemini based on retrieved context."""
        context_text = "\n".join(
            [f"- {d['sentence']} ({d.get('sentiment','?')})"
             for d in context if isinstance(d, dict)]
        )

        prompt = f"""You are FinGPT-Pro, a concise and factual financial analyst.
Use the retrieved context below to answer the user's question clearly.

Question:
{question}

Context:
{context_text}

Answer:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip() if response and response.text else "⚠️ No response from Gemini."
        except Exception as e:
            print(f"⚠️ Gemini generation error: {e}")
            return f"⚠️ Gemini error: {e}"

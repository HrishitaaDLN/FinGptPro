# 💼 FinGPT-Pro — Financial RAG Assistant

> A local-first financial analyst powered by **Retrieval-Augmented Generation (RAG)** using **Gemini 2.5** and **Qdrant VectorDB**.

FinGPT-Pro helps analyze financial datasets (like the Financial PhraseBank) using semantic search and large-language-model reasoning to summarize **positive**, **negative**, and **neutral** sentiments from contextually relevant statements.

---

## 🌟 Features

- 🔍 **Retrieval-Augmented Generation (RAG)** — Combines local vector search with Gemini-based reasoning  
- 🧠 **Financial Sentiment Analysis** — Understands tone from real-world corporate reports  
- 📊 **Interactive Dashboard** — View sentiment breakdown with beautiful visualizations  
- ⚡ **Fast Local Embedding Search** using **Sentence-Transformers + Qdrant**  
- 🧾 **Supports CSV or Text Datasets** (Financial PhraseBank included)

---

## 🧩 Project Structure

FinGPT-Pro/
│
├── app/
│ ├── main.py # Streamlit UI
│ ├── retriever.py # Qdrant retriever logic
│ ├── generator.py # Gemini model integration
│ ├── pipeline.py # Full RAG pipeline
│ ├── config.py # Environment variables & model paths
│
├── data/
│ └── financial_phrasebank_50agree.csv
│
├── requirements.txt
├── .env.example
├── README.md
└── setup.sh (optional for local setup)

## Installation (Local)
1. Clone the repository
git clone https://github.com/YourUsername/FinGPT-Pro.git
cd FinGPT-Pro

2. Create a virtual environment
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Set up environment variables

Create a .env file in the root folder with the following contents:

GOOGLE_API_KEY=your_google_api_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=finrag_docs
DATA_PATH=data/financial_phrasebank_50agree.csv

## Run locally
1. Make sure Qdrant is running

If you don’t already have Qdrant running locally, start it via Docker:

docker run -p 6333:6333 qdrant/qdrant


This launches Qdrant at http://localhost:6333

2. Launch FinGPT-Pro

Once Qdrant is running, start the Streamlit app:

streamlit run app/main.py

3. Open in your browser

After a few seconds, the app will be available at:

http://localhost:8501
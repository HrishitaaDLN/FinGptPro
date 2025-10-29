import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from app.pipeline import RAGPipeline
from qdrant_client import QdrantClient
import time

# -------------------------------
# 💄 PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="💼 FinGPT-Pro — Financial RAG Assistant",
    layout="wide",
    page_icon="💼",
)

# -------------------------------
# 🎨 CUSTOM STYLES
# -------------------------------
st.markdown("""
    <style>
    .main-title {
        font-size: 36px !important;
        font-weight: 700;
        color: #1F4E79;
    }
    .subheader {
        font-size: 18px !important;
        color: #444;
    }
    .stButton>button {
        background-color: #1F4E79;
        color: white;
        border-radius: 10px;
        font-weight: 600;
        height: 3em;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #15537C;
        color: #fff;
    }
    .credit {
        font-size: 14px;
        color: #888;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 🧠 APP HEADER
# -------------------------------
st.markdown("<h1 class='main-title'>💼 FinGPT-Pro</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Your Fast Local Financial Analyst powered by RAG</p>", unsafe_allow_html=True)
st.divider()

# -------------------------------
# 🧩 SIDEBAR — SYSTEM STATUS
# -------------------------------
with st.sidebar:
    st.header("💙 System Status")

    try:
        client = QdrantClient(host="localhost", port=6333)
        collections = [c.name for c in client.get_collections().collections]
        if collections:
            st.success("🟢 Qdrant Connected")
        else:
            st.warning("🟠 Connected — but no collections found")
    except Exception as e:
        st.error(f"🔴 Qdrant Offline\n{e}")

    st.subheader("🧠 Model Status")
    start = time.time()
    pipeline = RAGPipeline()
    elapsed = time.time() - start
    st.success(f"✅ Model ready in {elapsed:.1f}s")

    st.markdown("---")
    st.markdown("""
    **👩‍💻 Developed by [Hrishitaa Dharmavarapu](https://www.linkedin.com/in/hrishitaa-dharmavarapu-ln-3420a8205)**  
    🌐 [GitHub — HrishitaaDLN](https://github.com/HrishitaaDLN)
    """, unsafe_allow_html=True)

# -------------------------------
# 🚀 MAIN INTERFACE
# -------------------------------
if "result" not in st.session_state:
    st.session_state["result"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("💬 Ask a financial question:", placeholder="e.g., What’s the sentiment on inflation?")
with col2:
    ask_clicked = st.button("🔎 Analyze")

build_clicked = st.button("🔄 Build / Rebuild Index")

# -------------------------------
# ⚡ BUILD INDEX
# -------------------------------
if build_clicked:
    with st.spinner("Indexing financial dataset into Qdrant..."):
        pipeline.retriever.build_index()
    st.success("✅ Index built successfully!")

# -------------------------------
# 💬 RUN QUERY
# -------------------------------
if ask_clicked and query.strip():
    with st.spinner("💭 Thinking with Gemini..."):
        result = pipeline.query(query)
        st.session_state["result"] = result
        st.session_state["history"].insert(0, result)  # Save to history

# -------------------------------
# 📊 DISPLAY RESULT
# -------------------------------
if st.session_state["result"]:
    r = st.session_state["result"]

    st.markdown("### 💡 **Answer:**")
    st.write(r["answer"])

    # 🔹 Show retrieved context
    with st.expander("📄 Retrieved Context", expanded=False):
        for doc in r["context"]:
            st.markdown(f"- **{doc['sentence']}** — _({doc.get('sentiment', '?')})_")

    # -------------------------------
    # 🎨 Enhanced Sentiment Chart
    # -------------------------------
    st.markdown("### 📊 Sentiment Distribution")

    sentiments = [d.get("sentiment", "neutral") for d in r["context"]]
    counts = pd.Series(sentiments).value_counts().reindex(["positive", "neutral", "negative"]).fillna(0)
    total = counts.sum()
    percentages = (counts / total * 100).round(1)

    colors = {
        "positive": "#4CAF50",
        "neutral": "#9E9E9E",
        "negative": "#F44336",
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[colors.get(s, "#9E9E9E") for s in counts.index],
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9
    )

    # Label bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.05,
            f"{int(height)} ({pct}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    # Styling
    ax.set_title("Context Sentiment Breakdown", fontsize=14, fontweight="bold", color="#1F4E79")
    ax.set_xlabel("Sentiment", fontsize=12, color="#333")
    ax.set_ylabel("Frequency", fontsize=12, color="#333")
    ax.set_facecolor("#F9FAFB")
    fig.patch.set_facecolor("#F9FAFB")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Remove frame
    for spine in ax.spines.values():
        spine.set_visible(False)

    st.pyplot(fig)

# -------------------------------
# 🗂️ RECENT QUESTIONS HISTORY
# -------------------------------
if st.session_state["history"]:
    st.markdown("### 🗂️ Recent Questions")
    for i, entry in enumerate(st.session_state["history"][:5], 1):
        st.markdown(f"**{i}. Q:** {entry['query']}")
        st.markdown(f"🟢 **A:** {entry['answer']}")
        st.divider()

# -------------------------------
# 🪶 FOOTER
# -------------------------------
st.markdown("<p class='credit'>FinGPT-Pro © 2025 — Powered by Gemini + Qdrant</p>", unsafe_allow_html=True)

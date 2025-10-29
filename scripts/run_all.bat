@echo off
echo 🚀 Starting FinGPT-Pro system...
start cmd /k "docker run -p 6333:6333 qdrant/qdrant"
timeout /t 5
start cmd /k "cd C:\projects\FinGPT-PRO && .venv\Scripts\activate && streamlit run app/main.py"

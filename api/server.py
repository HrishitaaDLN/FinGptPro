from fastapi import FastAPI
from api.routes import rag_routes, ingest_routes

app = FastAPI(title="FinGPT-Pro API", version="1.0")

app.include_router(rag_routes.router)
app.include_router(ingest_routes.router)

@app.get("/")
def root():
    return {"message": "Welcome to FinGPT-Pro API 🚀"}

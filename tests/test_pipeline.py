from app.pipeline import RAGPipeline

def test_pipeline_output(monkeypatch):
    pipe = RAGPipeline()

    monkeypatch.setattr(pipe.retriever, "search", lambda q: [{"sentence": "Stocks rose due to inflation optimism."}])
    monkeypatch.setattr(pipe.generator, "generate", lambda q, c: "The market reacted positively to inflation.")

    result = pipe.query("How is the market reacting to inflation?")
    assert "answer" in result
    assert isinstance(result["context"], list)
    assert "market" in result["answer"].lower()

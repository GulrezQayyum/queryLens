from fastapi import FASTAPI
from pydantic import BaseModel
from app.strategies.hyde import hyde_retrieve
from queryLens.app.retrieval import retrieve

app = FastAPI(title="QueryLens", description="Query transformation comparison lab")

class QueryRequest(BaseModel):
    query: str
    k: int =5

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/retrieve/baseline")
def baseline_endpoint(req: QueryRequest):
    return {"strategy": "baseline", "query": req.query, "results": retrieve(req.query, req.k)}

@app.post("/retrieve/hyde")
def hyde_endpoint(req: QueryRequest):
    return hyde_retrieve(req.query, req.k)
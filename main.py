from fastapi import FastAPI
from pydantic import BaseModel

from config import TOP_K
import retrieval
import transformations as tx

app = FastAPI(title="QueryLens")


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query/baseline")
def query_baseline(req: QueryRequest):
    results = retrieval.retrieve(req.query, top_k=TOP_K)
    return {"strategy": "baseline", "query": req.query, "results": results}


@app.post("/query/hyde")
def query_hyde(req: QueryRequest):
    hypothetical = tx.hyde(req.query)
    results = retrieval.retrieve(hypothetical, top_k=TOP_K)
    return {
        "strategy": "hyde",
        "query": req.query,
        "hypothetical_document": hypothetical,
        "results": results,
    }


@app.post("/query/multiquery")
def query_multiquery(req: QueryRequest):
    variants = tx.multi_query(req.query)
    ranked_lists = [retrieval.retrieve(v, top_k=TOP_K) for v in variants]
    fused = retrieval.reciprocal_rank_fusion(ranked_lists, top_k=TOP_K)
    return {
        "strategy": "multi_query",
        "query": req.query,
        "variants": variants,
        "results": fused,
    }


@app.post("/query/stepback")
def query_stepback(req: QueryRequest):
    general_query = tx.step_back(req.query)
    original_results = retrieval.retrieve(req.query, top_k=TOP_K)
    general_results = retrieval.retrieve(general_query, top_k=TOP_K)
    fused = retrieval.reciprocal_rank_fusion([original_results, general_results], top_k=TOP_K)
    return {
        "strategy": "step_back",
        "query": req.query,
        "general_query": general_query,
        "results": fused,
    }


@app.post("/query/route")
def query_route(req: QueryRequest):
    strategy = tx.route(req.query)
    if strategy == "hyde":
        return query_hyde(req) | {"routed_to": "hyde"}
    elif strategy == "multi_query":
        return query_multiquery(req) | {"routed_to": "multi_query"}
    elif strategy == "step_back":
        return query_stepback(req) | {"routed_to": "step_back"}
    else:
        return query_baseline(req) | {"routed_to": "direct"}
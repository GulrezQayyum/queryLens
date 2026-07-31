import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL, TOP_K, RRF_K

_model = None
_collection = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model  # <--- Moved this OUTSIDE the if block
    
def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection
    
def retrieve(query_text: str, top_k: int = TOP_K):
    model = _get_model()
    collection = _get_collection()
    
    query_embedding = model.encode([query_text]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    
    ranked = []
    for doc_id, text, distance in zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    ):
        ranked.append({"id": doc_id, "text": text, "distance": distance})
    return ranked
    
def reciprocal_rank_fusion(ranked_list: list[list[dict]], k: int = RRF_K, top_k: int = TOP_K):
    scores: dict[str, float] = {}
    doc_lookup: dict[str, dict] = {}
    
    # Iterate through each ranked list (from each variant)
    for ranked in ranked_list:
        # Iterate through the items within this specific list
        for rank, item in enumerate(ranked):  # <--- CHANGE: use 'ranked' here, not 'ranked_list'
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            doc_lookup[doc_id] = item

    fused_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
    fused = [
        {**doc_lookup[doc_id], "rrf_score": scores[doc_id]}
        for doc_id in fused_ids[:top_k]
    ]
    return fused
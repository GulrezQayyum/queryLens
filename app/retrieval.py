from app.db import get_collection, embed


def retrieve(query: str, k: int = 5) -> list[dict]:
    collection = get_collection()
    q_emb = embed([query])[0]
    results = collection.query(query_embeddings=[q_emb], n_results=k)

    return [
        {
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]


def retrieve_by_text_query(query_text: str, k: int = 5) -> list[dict]:
    """Same as retrieve() — explicit alias used by strategies that generate
    a substitute query string (HyDE's hypothetical doc, step-back's abstract
    query, etc.) and just want to run it through the same retrieval path."""
    return retrieve(query_text, k=k)
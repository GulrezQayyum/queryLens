import os
from dotenv import load_dotenv

load_dotenv()

import json
import retrieval
import transformations as tx
from config import TOP_K

STRATEGIES = ["baseline", "hyde", "multi_query", "step_back"]
 
def run_strategy(strategy: str, query: str):
    if strategy == "baseline":
        return retrieval.retrieve(query, top_k = TOP_K)
    elif strategy == "hyde":
        hypothetical = tx.hyde(query)
        return retrieval.retrieve(hypothetical, top_k = TOP_K)
    elif strategy == "multi_query":
        variants = tx.multi_query(query)
        ranked_list = [retrieval.retrieve(variants, top_k = TOP_K)]
        return retrieval.reciprocal_rank_fusion(ranked_list, top_k = TOP_K)
    elif strategy == "step_back":
        general_query = tx.step_back(query)
        original_results = retrieval.retrieve(query, top_k=TOP_K)
        general_results = retrieval.retrieve(general_query, top_k=TOP_K)
        return retrieval.reciprocal_rank_fusion([original_results, general_results], top_k=TOP_K)
    raise ValueError(strategy)

def base_doc_id(chunk_id: str) -> str:
    parts = chunk_id.split("_")
    return "_".join(parts[:2]) if len(parts) > 2 else chunk_id
    
def recall_at_k(results: list[dict], relevant_ids: list[str]) -> float:
    retrieved = {base_doc_id(r["id"]) for r in results}
    hit = len(retrieved & set(relevant_ids))
    return hit / len(relevant_ids) if relevant_ids else 0.0

def mrr(results: list[dict], relevant_ids: list[str]) -> float:
    relevant_set = set(relevant_ids)
    for rank, r in enumerate(results, start=1):
        if base_doc_id(r["id"]) in relevant_set:
            return 1.0 / rank
    return 0.0

def main():
    with open("eval_queries.json") as f:
        eval_set = json.load(f)

    scores = {s: {"recall": [], "mrr": []} for s in STRATEGIES}
    per_query_log = []

    for item in eval_set:
        query, relevant_ids = item["query"], item["relevant_ids"]
        row = {"query": query}
        for strategy in STRATEGIES:
            results = run_strategy(strategy, query)
            r = recall_at_k(results, relevant_ids)
            m = mrr(results, relevant_ids)
            scores[strategy]["recall"].append(r)
            scores[strategy]["mrr"].append(m)
            row[strategy] = {"recall": r, "mrr": m, "top_ids": [x["id"] for x in results]}
        per_query_log.append(row)
        print(f"done: {query[:60]}...")

    print("\n" + "=" * 60)
    print(f"{'Strategy':<15}{'Recall@'+str(TOP_K):<15}{'MRR':<15}")
    print("=" * 60)
    summary = {}
    for strategy in STRATEGIES:
        avg_recall = sum(scores[strategy]["recall"]) / len(scores[strategy]["recall"])
        avg_mrr = sum(scores[strategy]["mrr"]) / len(scores[strategy]["mrr"])
        summary[strategy] = {"recall": round(avg_recall, 3), "mrr": round(avg_mrr, 3)}
        print(f"{strategy:<15}{avg_recall:<15.3f}{avg_mrr:<15.3f}")
    print("=" * 60)

    with open("eval_results.json", "w") as f:
        json.dump({"summary": summary, "per_query": per_query_log}, f, indent=2)
    print("\nSaved full results to eval_results.json")


if __name__ == "__main__":
    main()
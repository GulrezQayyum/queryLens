import json
import os
from app.retrieval import retrieve
from app.strategies.hyde import hyde_retrieve

GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")

STRATEGIES = {
    "baseline": lambda q, k: retrieve(q, k),
    "hyde": lambda q, k: hyde_retrieve(q, k)["results"],
}

def recall_at_k(retrieved_ids: list[list], relevant_ids: list[list], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids) & set(relevant_ids))
    return hits / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0

def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0

def load_ground_truth():
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run(k: int = 5):
    ground_truth = load_ground_truth()
    if any("REPLACE_WITH_REAL_QUERY" in item["query"] for item in ground_truth):
        print("[eval] ground_truth.json still has placeholder queries — "
              "fill it in with real queries + relevant chunk_ids first.")
        return

    scores = {name: {"recall": [], "mrr": []} for name in STRATEGIES}

    for item in ground_truth:
        query = item["query"]
        relevant = item["relevant_chunk_ids"]

        for name, fn in STRATEGIES.items():
            results = fn(query, k)
            retrieved_ids = [r["id"] for r in results]
            scores[name]["recall"].append(recall_at_k(retrieved_ids, relevant))
            scores[name]["mrr"].append(mrr(retrieved_ids, relevant))

    print(f"\n{'Strategy':<15} {'Recall@' + str(k):<12} {'MRR':<10}")
    print("-" * 37)
    for name, s in scores.items():
        avg_recall = sum(s["recall"]) / len(s["recall"])
        avg_mrr = sum(s["mrr"]) / len(s["mrr"])
        print(f"{name:<15} {avg_recall:<12.3f} {avg_mrr:<10.3f}")

if __name__ == "__main__":
    run()


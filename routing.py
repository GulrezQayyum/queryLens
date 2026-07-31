import json

import transformations as tx
from eval import run_strategy, recall_at_k, mrr, STRATEGIES


def best_strategy_for(query: str, relevant_ids: list[str]):
    scored = []
    for strategy in STRATEGIES:
        results = run_strategy(strategy, query)
        scored.append((strategy, mrr(results, relevant_ids), recall_at_k(results, relevant_ids)))
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return scored  

def main():
    with open("eval_queries.json") as f:
        eval_set = json.load(f)

    rows = []
    matches = 0
    near_matches = 0  

    for item in eval_set:
        query, relevant_ids = item["query"], item["relevant_ids"]

        routed_to = tx.route(query)
        scored = best_strategy_for(query, relevant_ids)
        best_name, best_mrr, best_recall = scored[0]

        routed_score = next((s for s in scored if s[0] == routed_to), None)
        routed_mrr = routed_score[1] if routed_score else 0.0

        is_match = routed_to == best_name
        is_near = (best_mrr - routed_mrr) <= 0.1
        matches += int(is_match)
        near_matches += int(is_near)

        rows.append({
            "query": query,
            "routed_to": routed_to,
            "best_strategy": best_name,
            "best_mrr": round(best_mrr, 3),
            "routed_mrr": round(routed_mrr, 3),
            "exact_match": is_match,
            "near_match": is_near,
        })
        flag = "✓" if is_match else ("~" if is_near else "✗")
        print(f"{flag} routed={routed_to:<12} best={best_name:<12} "
              f"routed_mrr={routed_mrr:.2f} best_mrr={best_mrr:.2f}  {query[:50]}")

    n = len(eval_set)
    print("\n" + "=" * 60)
    print(f"Exact match rate : {matches}/{n} ({100*matches/n:.0f}%)")
    print(f"Near match rate  : {near_matches}/{n} ({100*near_matches/n:.0f}%)  "
          f"(routed strategy within 0.1 MRR of the best strategy)")
    print("=" * 60)
    print(
        "\nRead this as: exact match means the router picked the single best "
        "strategy for that query. Near match means it picked something almost "
        "as good, even if not the top strategy. A low near-match rate is the "
        "signal to actually look at the routing prompt, not the metric formula."
    )

    with open("routing_eval_results.json", "w") as f:
        json.dump({
            "exact_match_rate": matches / n,
            "near_match_rate": near_matches / n,
            "per_query": rows,
        }, f, indent=2)
    print("\nSaved to routing_eval_results.json")


if __name__ == "__main__":
    main()
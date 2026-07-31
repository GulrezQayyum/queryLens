import sys
import json
from dotenv import load_dotenv


load_dotenv()                    

import retrieval
import transformations as tx
from config import TOP_K



def show_multi_query(query: str):
    print("\n--- DAY 3: multi-query expansion ---")
    variants = tx.multi_query(query)
    print(f"Original : {query}")
    for i, v in enumerate(variants, 1):
        print(f"Variant {i}: {v}")

    ranked_lists = [retrieval.retrieve(v, top_k=TOP_K) for v in variants]
    baseline = retrieval.retrieve(query, top_k=TOP_K)
    fused = retrieval.reciprocal_rank_fusion(ranked_lists, top_k=TOP_K)

    print(f"\nBaseline top-{TOP_K} ids : {[r['id'] for r in baseline]}")
    print(f"Fused top-{TOP_K} ids    : {[r['id'] for r in fused]}")
    if all(r["id"] == baseline[0]["id"] for r in [ranked_lists[i][0] for i in range(len(ranked_lists))]):
        print("Note: every variant's top hit matches baseline's top hit — "
              "multi-query added no new signal for this query.")


def show_step_back(query: str):
    print("\n--- DAY 4: step-back prompting ---")
    general_query = tx.step_back(query)
    print(f"Original      : {query}")
    print(f"Step-back to  : {general_query}")
    original_results = retrieval.retrieve(query, top_k=TOP_K)
    general_results = retrieval.retrieve(general_query, top_k=TOP_K)
    fused = retrieval.reciprocal_rank_fusion([original_results, general_results], top_k=TOP_K)

    print(f"\nOriginal-query top-{TOP_K} ids  : {[r['id'] for r in original_results]}")
    print(f"Step-back-query top-{TOP_K} ids : {[r['id'] for r in general_results]}")
    print(f"Fused top-{TOP_K} ids           : {[r['id'] for r in fused]}")
    
    if general_results and original_results and general_results[0]["id"] == original_results[0]["id"]:
        print("Note: step-back query's top hit matches the original's top hit — "
              "check whether the generated question is actually more general, "
              "or just a reworded version of the original.")


def run_one(query: str):
    show_multi_query(query)
    show_step_back(query)


def run_all():
    with open("eval_queries.json") as f:
        eval_set = json.load(f)
    for item in eval_set:
        print("\n" + "=" * 70)
        print(f"QUERY: {item['query']}")
        print(f"Ground truth: {item['relevant_ids']}")
        run_one(item["query"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:\n  python explore_script.py \"your question\"\n  python explore_script.py --all")
        sys.exit(1)

    if sys.argv[1] == "--all":
        run_all()
    else:
        run_one(" ".join(sys.argv[1:]))


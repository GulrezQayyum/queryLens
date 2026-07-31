import json
import os

import matplotlib.pyplot as plt

from eval import STRATEGIES


def main():
    if not os.path.exists("eval_results.json"):
        raise SystemExit("eval_results.json not found — run `python eval.py` first.")

    with open("eval_results.json") as f:
        eval_results = json.load(f)

    summary = eval_results["summary"]
    recalls = [summary[s]["recall"] for s in STRATEGIES]
    mrrs = [summary[s]["mrr"] for s in STRATEGIES]

    x = range(len(STRATEGIES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_recall = ax.bar([i - width / 2 for i in x], recalls, width, label="Recall@5")
    bars_mrr = ax.bar([i + width / 2 for i in x], mrrs, width, label="MRR")

    ax.set_xticks(list(x))
    ax.set_xticklabels(STRATEGIES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Query Transformation Strategy Comparison (QueryLens)")
    ax.legend()
    ax.bar_label(bars_recall, fmt="%.2f", padding=2)
    ax.bar_label(bars_mrr, fmt="%.2f", padding=2)

    fig.tight_layout()
    fig.savefig("eval_comparison.png", dpi=150)
    print("Saved eval_comparison.png")

    # Optional second panel: routing accuracy, only if that eval was run
    if os.path.exists("routing_eval_results.json"):
        with open("routing_eval_results.json") as f:
            routing = json.load(f)

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        rates = [routing["exact_match_rate"], routing["near_match_rate"]]
        labels = ["Exact match\n(picked best strategy)", "Near match\n(within 0.1 MRR)"]
        bars = ax2.bar(labels, rates, color=["#4C72B0", "#55A868"])
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Rate")
        ax2.set_title("Query Router Accuracy")
        ax2.bar_label(bars, fmt="%.2f", padding=2)
        fig2.tight_layout()
        fig2.savefig("routing_accuracy.png", dpi=150)
        print("Saved routing_accuracy.png")
    else:
        print("routing_eval_results.json not found — skipping routing chart "
              "(run explore_day5_routing.py first if you want it).")


if __name__ == "__main__":
    main()
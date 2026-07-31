# QueryLens

A small, focused project for learning **query transformation** techniques in
Retrieval-Augmented Generation (RAG): HyDE, multi-query expansion, step-back
prompting, and query routing — each measured against a no-transformation
baseline on the same corpus, using the same retriever.

This is the third project in an Advanced RAG Engineering roadmap, following
`ChunkLab` (chunking strategies) and a hybrid search implementation (dense +
sparse fusion). It's deliberately smaller in scope than ChunkLab: one corpus,
one embedding model, one ChromaDB collection. The only variable that changes
between runs is **what happens to the query before it reaches the retriever.**

---

## Why this project exists

Dense retrieval ranks documents by embedding similarity to the raw query.
That works well when the query's vocabulary overlaps with the source
documents' vocabulary, and degrades when it doesn't. Four strategies attempt
to close that gap *before* retrieval happens:

| Strategy | Idea |
|---|---|
| **HyDE** | Ask an LLM to hallucinate a plausible answer passage, then embed *that* instead of the raw query — answer-shaped text tends to sit closer to real answer passages in embedding space than a short question does. |
| **Multi-query expansion** | Ask an LLM to rephrase the query several different ways, retrieve for each, then fuse the ranked lists with Reciprocal Rank Fusion (RRF). |
| **Step-back prompting** | Ask an LLM to generalize the query to a broader question first, retrieve on both, then fuse. The idea is that background context helps even narrow questions. |
| **Query routing** | Classify the query and pick which of the above strategies (or none) it actually needs, instead of always paying for the expensive ones. |

The goal of this project isn't to prove these techniques work — it's to
measure **whether, and when, they actually help**, on a real (if small)
retrieval task, and to build the instinct for reading an eval result
honestly rather than assuming a fancier pipeline is automatically better.

---

## Project structure

```
queryLens/
├── config.py                  # Constants: embedding model, Groq model, paths, top-k, RRF k
├── corpus.py                  # 20 single-topic toy passages (the dataset)
├── ingest.py                  # Chunk corpus.py + embed + write to ChromaDB
│
├── retrieval.py                # Shared engine: embed+search, and RRF fusion
├── transformations.py          # hyde(), multi_query(), step_back(), route()
├── routing.py                  # Query router — classifies a query, dispatches to a strategy
├── main.py                     # FastAPI app, one endpoint per strategy
│
├── eval_queries.json           # 20 ground-truth queries, paraphrased from corpus wording
├── eval.py                     # Runs all 4 strategies vs baseline → Recall@5 / MRR
├── eval_results.json           # Saved output of eval.py (checked in — see Results below)
│
├── explore_script.py           # Inspect intermediate output (variants, step-back questions) by eye
├── routing_eval_results.json   # Saved output of the routing accuracy check
├── visualize_eval.py           # Turns the two result JSONs into bar charts
├── eval_comparison.png         # Strategy comparison chart (checked in)
├── routing_accuracy.png        # Routing accuracy chart (checked in)
│
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in GROQ_API_KEY
```

## Ingest the corpus

```bash
python ingest.py
```

Chunking is intentionally the simplest possible strategy (fixed-size,
word-based, with overlap). Chunking isn't what this project teaches — that
was ChunkLab's job. Keeping it dumb here keeps it out of the results.

## Run the API

```bash
uvicorn main:app --reload --port 8002
```

| Endpoint | Description |
|---|---|
| `POST /query/baseline` | Raw query, no transformation |
| `POST /query/hyde` | HyDE |
| `POST /query/multiquery` | Multi-query expansion + RRF |
| `POST /query/stepback` | Step-back prompting + RRF |
| `POST /query/route` | Routes to whichever strategy the classifier picks |

Body: `{"query": "your question"}`

## Run the eval

```bash
python eval.py                 # strategy comparison → eval_results.json
python explore_script.py       # inspect intermediate output for a query, or --all
python visualize_eval.py       # → eval_comparison.png, routing_accuracy.png
```

---

## Results (from `eval_results.json`)

| Strategy | Recall@5 | MRR |
|---|---|---|
| **baseline** | 1.0 | **0.950** |
| hyde | 1.0 | 0.917 |
| multi_query | 1.0 | 0.900 |
| step_back | 1.0 | 0.875 |

**Baseline won.** Every transformation strategy scored *lower* MRR than
doing nothing. Recall@5 is a perfect 1.0 across the board, so on this
corpus every strategy always finds the right document somewhere in the top
5 — the only metric that actually differentiates them is MRR (how high the
correct document ranks), and on that measure, transformation added more
noise than signal.

### Query-level breakdown

Out of 20 eval queries:

- **7 queries got worse** with at least one transformation. Example:
  *"How do you check if a generated answer is actually grounded in what
  was retrieved?"* — baseline ranked the correct doc #1 (MRR 1.0), but
  HyDE's hallucinated answer pulled it down to rank 3 (MRR 0.33), and
  step-back dropped it to rank 2 (MRR 0.5).
- **2 queries got better** with a transformation. Example: *"How can
  questions requiring multiple connected facts be answered?"* — baseline
  only ranked the correct doc #2 (MRR 0.5); step-back promoted it to #1
  (MRR 1.0).
- **11 queries were unaffected** — every strategy landed the correct
  document at rank 1 regardless.

### Why baseline won here

This isn't a bug in the transformation logic — it's a property of the eval
setup:

1. **The eval queries are paraphrased, not adversarial.** They reword a
   few terms (e.g. "keyword frequency scoring" instead of "term
   frequency") but still share enough vocabulary with the source passage
   that dense retrieval, which is already good at semantic matching,
   doesn't struggle. Query transformation earns its value when there's a
   genuine vocabulary or framing gap — a light paraphrase isn't that.
2. **The corpus is small and single-topic-per-passage** (20 well-separated
   docs). With so little topic overlap, there's very little for a query to
   miss in the first place, so there's little room for a transformation to
   improve on. In the 7 "hurt" cases, the transformation's generated
   variant pulled in a topically-adjacent-but-wrong passage that
   outranked the correct one — a failure mode that gets *worse*, not
   better, as the corpus gets easier.
3. **Practical takeaway:** on an easy retrieval task, transformation is a
   net cost — extra LLM calls for no MRR gain, and occasional harm. These
   techniques are built for genuinely hard queries, which is exactly the
   case that query routing is meant to detect and handle differently from
   the easy ones.

### Known issue: the routing eval numbers are misleading

`routing_eval_results.json` reports `exact_match_rate: 0.0` and
`near_match_rate: 0.05`, which looks like the router is badly broken. It
isn't — there's a naming mismatch between `routing.py`'s route labels
(`"direct"`) and `eval.py`'s strategy labels (`"baseline"`) for the same
thing. The eval script's lookup:

```python
routed_score = next((s for s in scored if s[0] == routed_to), None)
```

never matches `"direct"` against `"baseline"` in the scored list, so
`routed_mrr` silently falls back to `0.0` every time the router picked
direct retrieval — which, given the baseline-wins result above, is almost
always the *correct* choice on this corpus. Fix:

```python
ROUTE_ALIASES = {"direct": "baseline"}
routed_key = ROUTE_ALIASES.get(routed_to, routed_to)
routed_score = next((s for s in scored if s[0] == routed_key), None)
```

After this fix, the exact-match rate should land much closer to the
router's actual behavior — it mostly chose direct retrieval, and direct
retrieval mostly was the best strategy.

---

## What I'd change with more time

- **Harder eval queries.** The current set paraphrases lightly; a stronger
  test would include queries with genuine vocabulary mismatch (different
  domain terms for the same concept) or ambiguous phrasing where dense
  retrieval alone struggles — the conditions transformation is actually
  built for.
- **A bigger, noisier corpus.** 20 well-separated single-topic passages is
  close to a best case for plain dense retrieval. A few hundred passages
  with topical overlap and near-duplicates would create more room for
  transformation strategies to differentiate themselves.
- **Fix the routing eval naming bug** (above) before drawing conclusions
  about routing accuracy.
- **Cost/latency tracking.** Right now the comparison is purely
  Recall@5/MRR. Since every transformation strategy costs at least one
  extra LLM call, a fair comparison should weigh accuracy gain against
  added latency and cost — especially since baseline already wins on
  accuracy here.

---

## Reusing this for a different corpus

Swap `CORPUS` in `corpus.py` for your own `(id, text)` pairs, rewrite
`eval_queries.json` with matching ground truth, then re-run `ingest.py` and
`eval.py`. Everything else — the transformation logic, RRF fusion, and the
eval math — stays the same.
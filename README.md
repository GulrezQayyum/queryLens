# QueryLens

A small, focused project for learning **query transformation** techniques in
RAG: HyDE, multi-query expansion, step-back prompting, and query routing —
each compared against a no-transformation baseline on the same corpus.

Deliberately small. Chunking is fixed-size and boring on purpose (that's
ChunkLab's job, not this project's). The only thing that changes between
runs is what happens to the query before it hits the vector store.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in GROQ_API_KEY
```

## Ingest a corpus

Reuse a text file from ChunkLab (e.g. the Meditations text) or any plain
`.txt` file:

```bash
python -m app.ingest data/corpus.txt
```

## Run the API

```bash
uvicorn app.main:app --reload --port 8002
```

Endpoints:
- `POST /retrieve/baseline` — raw query, no transformation
- `POST /retrieve/hyde` — HyDE

Body: `{"query": "your question", "k": 5}`

## Build order

- [x] Day 1 — scaffold, baseline retrieval, ground truth template
- [x] Day 2 — HyDE (`app/strategies/hyde.py`)
- [ ] Day 3 — multi-query expansion + RRF fusion (`app/strategies/multi_query.py`)
      — reuse the RRF implementation from your `hybrid_search.py`
- [ ] Day 4 — step-back prompting (`app/strategies/step_back.py`)
- [ ] Day 5 — query routing (`app/strategies/router.py`)
- [ ] Day 6 — fill in `app/eval/ground_truth.json` with 15-20 real queries,
      run `python -m app.eval.run_eval`
- [ ] Day 7 — write up findings (good candidate for the academic-writing
      practice — this doubles as a mini paper: motivation, method,
      results table, discussion)

## Why this design

Every strategy funnels through `app/retrieval.py::retrieve_by_text_query`,
so the only thing each strategy file does is **produce a better query
string (or set of them)** before handing off to the same retrieval path.
That's the whole concept of query transformation in one line of code —
worth sitting with before you build the next strategy.
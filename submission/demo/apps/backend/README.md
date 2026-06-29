# OCEAN Pipeline Demo — Backend

FastAPI server that runs one essay through the full OCEAN-prediction pipeline
and streams all four pipeline stages to the browser via NDJSON.

## Quick start

```bash
# From the repo root:
make install   # uv sync  (reads pyproject.toml at repo root)
make dev       # starts backend + opens browser at http://127.0.0.1:8000
```

## Manual start

```bash
PYTHONPATH=packages:apps/backend \
  uvicorn app:app --app-dir apps/backend --host 127.0.0.1 --port 8000
```

## Resolved design decisions (from demo_webapp_requirements.md)

| Decision | Choice |
|----------|--------|
| Input mode | **C** — pick a labelled test essay *or* paste free text |
| Profiler model | `gpt-4o-mini` |
| Prediction model | `gpt-4o-mini` (5 reasoned calls/run) |
| top_k | `3` |
| temperature | `0.0` (deterministic) |
| Response | **streaming** NDJSON (stage-by-stage reveal) |
| Retrieval panel | all 5 traits |
| Port | `8000` (override with `DEMO_PORT` env var) |
| Dropdown source | `data/split/essays/test50.csv` |
| Pinned default essay | row `0` (★ in the dropdown) |
| Input length | 200 – 12 000 chars for free text |

## Project layout

```
apps/
  backend/          ← you are here
    app.py          FastAPI endpoints
    pipeline.py     Thin orchestration (wraps packages/ without re-implementing)
    pyproject.toml  Project metadata + editor path config
    logs/           demo_run.log appended per run (git-ignored)
  frontend/
    index.html      Single-page UI
    app.js          Vanilla JS — streams NDJSON, renders 4 panels
    style.css       Design system (dark, glass cards, aurora, timeline)
packages/           Shared pipeline code (ptd_model/, rag/, utils/)
data/               Essays CSVs + vector index
Makefile            make dev / install / check / clean
requirements.txt    All Python deps
```

## Prerequisites

1. **[uv](https://docs.astral.sh/uv/)** installed — see https://docs.astral.sh/uv/getting-started/installation/
2. **Python ≥ 3.10** — uv will manage the venv automatically.
   ```
   make install   # runs: uv sync  (reads pyproject.toml)
   ```
2. **`.env`** at repo root with `OPENAI_API_KEY=sk-...` — never commit it.
3. **Vector index** at `data/vector_db/essays_dual/` (already in repo):
   - `vectors.faiss`, `vectors_meta.jsonl`, `facet_vectors.npy`

   The embedder uses the public `nomic-ai/nomic-embed-text-v1.5` model
   (downloaded on first use, ~270 MB). No local model artifacts are required.

## Defense-day checklist

- [ ] `OPENAI_API_KEY` set in `.env`; one successful test run completed.
- [ ] Default essay (★, row 0) predicts correctly — preselected on load.
- [ ] **Record a screen capture of a full run as a fallback video.**
- [ ] Projector check: running state visible per stage, fonts readable.
- [ ] Avoid spamming **Run** — each run is ~6 OpenAI calls (~15–30 s).

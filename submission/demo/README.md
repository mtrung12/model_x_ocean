# OCEAN Pipeline Demo

A small local web app that runs one essay through the full OCEAN (Big-Five) prediction
pipeline and visualises every stage: **Profile → Retrieval → Reasoned prediction → Verdict**.

- **Backend:** FastAPI (`apps/backend/`) — wraps the existing pipeline in `packages/`.
- **Frontend:** single static page (`apps/frontend/`) — vanilla JS, no build step.
- **Pipeline:** profiler + retriever + reasoned predictor (`packages/rag/`, `packages/ptd_model/`).

Runs locally only. Every request calls the OpenAI API live (no cache).

---

## Prerequisites

1. **[uv](https://docs.astral.sh/uv/)** installed — https://docs.astral.sh/uv/getting-started/installation/
   (uv manages the Python ≥ 3.10 venv automatically).
2. **`make`** — on Windows use Git Bash, WSL, or run the manual commands below.
3. **`.env`** at the repo root with your OpenAI key:
   ```
   OPENAI_API_KEY=sk-...
   ```
   Never commit it.
4. **Vector index** at `data/vector_db/essays_dual/` (already in the repo):
   `vectors.faiss`, `vectors_meta.jsonl`, `facet_vectors.npy`.
   The embedder downloads `nomic-ai/nomic-embed-text-v1.5` (~270 MB) on first use.

---

## Run

```bash
make install     # uv sync — install Python deps from pyproject.toml
make dev         # start backend + open browser
```

`make dev` serves on **http://127.0.0.1:3000** (Makefile `PORT` default) and opens the
browser when ready. Override the port:

```bash
make PORT=8000 dev
```

### Manual start (no make)

```bash
uv sync

# bash / WSL
PYTHONPATH=packages:apps/backend \
  uv run python -m uvicorn app:app --app-dir apps/backend --host 127.0.0.1 --port 8000
```

```powershell
# PowerShell
uv sync
$env:PYTHONPATH = "packages;apps/backend"
uv run python -m uvicorn app:app --app-dir apps/backend --host 127.0.0.1 --port 8000
```

Run directly via `python apps/backend/app.py` and it defaults to port **8000**
(override with the `DEMO_PORT` env var).

---

## Use

1. Open the page.
2. Pick a labelled test essay from the dropdown (★ default, row 0) **or** paste free text
   (200 – 12 000 chars).
3. Click **Run**. Stages stream in one by one (~15–30 s, ~6 OpenAI calls per run).
4. The verdict panel shows the 5 predictions; for a labelled essay it adds ✓/✗ vs. ground truth.

---

## Other make targets

| Command        | What it does                              |
|----------------|-------------------------------------------|
| `make help`    | List all targets                          |
| `make install` | `uv sync` — install deps                  |
| `make dev`     | Backend + auto-open browser (hot-reload)  |
| `make dev-be`  | Backend only (hot-reload)                 |
| `make check`   | Syntax-check Python + JS sources          |
| `make clean`   | Remove `__pycache__`, `*.pyc`, logs       |

---

## Layout

```
apps/
  backend/    FastAPI app (app.py) + orchestration (pipeline.py)
  frontend/   index.html + app.js + style.css (served by backend)
packages/     Shared pipeline: rag/ (profiler, retriever, embedder), ptd_model/, utils/
data/         Essay CSVs (data/split/essays/) + vector index (data/vector_db/)
pyproject.toml  Deps + project metadata
Makefile        dev / install / check / clean
```

See [apps/backend/README.md](apps/backend/README.md) for backend design decisions and the
defense-day checklist, and [demo_webapp_requirements.md](demo_webapp_requirements.md) for full requirements.

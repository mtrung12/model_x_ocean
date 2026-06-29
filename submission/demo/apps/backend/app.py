"""FastAPI backend for the OCEAN-pipeline demo.

Endpoints
---------
GET  /            -> the single static page (index.html)
GET  /essays      -> labelled test essays for the dropdown (mode A/C)
POST /predict     -> NDJSON stream of pipeline stage events (mode A/B/C)

Run from repo root (PYTHONPATH set by Makefile / pyproject):
    make dev-be
or manually:
    PYTHONPATH=packages:apps/backend uvicorn app:app --app-dir apps/backend --host 127.0.0.1 --port 8000

Always-live: every /predict hits the OpenAI API in real time (no cache),
per the requirements.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

log = logging.getLogger("app")

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import pipeline  # apps/backend/pipeline.py — on PYTHONPATH via Makefile

# apps/backend/ → apps/ → repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_APPS_DIR = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_APPS_DIR)

# Frontend static files live in apps/frontend/
_FRONTEND_DIR = os.path.join(_APPS_DIR, "frontend")

# Source of dropdown essays. test50 is the smaller, defense-friendly slice.
ESSAYS_CSV = os.path.join(
    _REPO_ROOT, "data", "split", "essays", "test50.csv"
)

# Input length guards (profiler prompt assumes essay-length text).
MIN_TEXT_CHARS = 200
MAX_TEXT_CHARS = 12000

# Pinned default essay (reproducible first demo). Row index into ESSAYS_CSV.
# Row 0 is the known sorority/premed essay (labels high,high,low,high,high).
DEFAULT_ESSAY_ID = 0

app = FastAPI(title="OCEAN Pipeline Demo", version="1.0")


# --------------------------------------------------------------------------- #
# Essay store
# --------------------------------------------------------------------------- #

def _load_essays() -> pd.DataFrame:
    if not os.path.exists(ESSAYS_CSV):
        return pd.DataFrame()
    df = pd.read_csv(ESSAYS_CSV)
    df = df.reset_index(drop=True)
    return df


_ESSAYS_DF = _load_essays()


def _ground_truth_for_row(row: pd.Series) -> Dict[str, str]:
    """Map dataset label columns to full-trait-name -> high/low."""
    gt: Dict[str, str] = {}
    for col, trait in pipeline.COLUMN_TO_TRAIT.items():
        if col in row.index and pd.notna(row[col]):
            val = str(row[col]).strip().lower()
            if val in ("high", "low"):
                gt[trait] = val
            elif val in ("1", "1.0"):
                gt[trait] = "high"
            elif val in ("0", "0.0"):
                gt[trait] = "low"
    return gt


def _label_summary(gt: Dict[str, str]) -> str:
    order = ["Openness", "Conscientiousness", "Extraversion",
             "Agreeableness", "Neuroticism"]
    short = {"Openness": "O", "Conscientiousness": "C", "Extraversion": "E",
             "Agreeableness": "A", "Neuroticism": "N"}
    return "  ".join(
        f"{short[t]}:{gt[t]}" for t in order if t in gt
    )


@app.get("/essays")
def list_essays() -> List[Dict]:
    """Return dropdown rows: {id, label_summary, preview}."""
    out = []
    for idx, row in _ESSAYS_DF.iterrows():
        gt = _ground_truth_for_row(row)
        text = str(row.get("text", ""))
        out.append(
            {
                "id": int(idx),
                "label_summary": _label_summary(gt),
                "preview": pipeline._snippet(text, 140),
                "full_text": text,
                "is_default": int(idx) == DEFAULT_ESSAY_ID,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Predict (streaming)
# --------------------------------------------------------------------------- #


class PredictRequest(BaseModel):
    text: Optional[str] = None
    essay_id: Optional[int] = None
    top_k: int = pipeline.DEFAULT_TOP_K
    model_name: str = pipeline.DEFAULT_MODEL


def _resolve_input(req: PredictRequest):
    """Return (text, ground_truth) for a request, or raise HTTPException."""
    if req.essay_id is not None:
        if _ESSAYS_DF.empty or req.essay_id not in _ESSAYS_DF.index:
            raise HTTPException(404, f"essay_id {req.essay_id} not found")
        row = _ESSAYS_DF.loc[req.essay_id]
        return str(row["text"]), _ground_truth_for_row(row)

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(400, "Provide non-empty `text` or a valid `essay_id`.")
    if len(text) < MIN_TEXT_CHARS:
        raise HTTPException(
            400,
            f"Text too short ({len(text)} chars); need >= {MIN_TEXT_CHARS} "
            "for the profiler to work well.",
        )
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            400, f"Text too long ({len(text)} chars); max {MAX_TEXT_CHARS}."
        )
    return text, None  # free-text: no ground truth


_PIPELINE_EXECUTOR = ThreadPoolExecutor(max_workers=4)
_SENTINEL = object()


@app.post("/predict")
async def predict(req: PredictRequest):
    """Stream pipeline stage events as newline-delimited JSON (NDJSON)."""
    text, ground_truth = _resolve_input(req)
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run_pipeline():
        try:
            for event in pipeline.run_pipeline(
                text=text,
                ground_truth=ground_truth,
                top_k=req.top_k,
                model_name=req.model_name,
            ):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:
            log.error("Unhandled pipeline exception:\n%s", traceback.format_exc())
            loop.call_soon_threadsafe(
                queue.put_nowait, {"stage": "error", "message": str(exc)}
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    async def event_stream():
        loop.run_in_executor(_PIPELINE_EXECUTOR, _run_pipeline)
        while True:
            event = await queue.get()
            if event is _SENTINEL:
                break
            # SSE format: browsers flush on each "data: ...\n\n" frame.
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- #
# Static page
# --------------------------------------------------------------------------- #


_NO_CACHE = {"Cache-Control": "no-store"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"), headers=_NO_CACHE)


@app.get("/static/app.js")
def app_js() -> FileResponse:
    return FileResponse(os.path.join(_FRONTEND_DIR, "app.js"), headers=_NO_CACHE)


@app.get("/static/style.css")
def style_css() -> FileResponse:
    return FileResponse(os.path.join(_FRONTEND_DIR, "style.css"), headers=_NO_CACHE)


app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")


@app.on_event("startup")
def _startup() -> None:
    # Surface the §5.1 artifact blocker at boot, not mid-demo.
    try:
        pipeline.warm_up()
        print("[app] Retriever + index warmed up. Ready.")
    except Exception as exc:  # noqa: BLE001
        print(f"[app] WARNING: retriever failed to warm up: {exc}")
        print("[app] The app will start, but /predict will fail until the "
              "vector index/model artifacts are available (see README §5.1).")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=int(os.getenv("DEMO_PORT", "8000")),
        reload=False,
    )

"""Thin orchestration over the existing OCEAN-prediction pipeline.

Wraps (does NOT re-implement) the profiler, retriever, and reasoned predictor
already in `ptd_model/` and `rag/`. Exposes one generator, `run_pipeline`,
that yields a sequence of stage events so the FastAPI layer can stream them
to the browser. A non-streaming caller can simply drain the generator and
keep the final "done" event.

Stage event shape (each is a plain dict, JSON-serialisable):

    {"stage": "profile",   "status": "running"|"done", ...}
    {"stage": "retrieval", "trait": "...", ...}
    {"stage": "predict",   "trait": "...", "index": 1, "total": 5, ...}
    {"stage": "done",      "result": {<full API response object>}}
    {"stage": "error",     "message": "..."}

The full result object mirrors the API contract in
demo_webapp_requirements.md §4.
"""

from __future__ import annotations

import logging
import os
import traceback
from typing import Dict, Iterator, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

from ptd_model.predict import (
    TRAIT_TO_COLUMN,
    TRAIT_TO_RAG_NAME,
    _call_profiler_for_text,
    _profile_to_full_text,
    build_prompt,
    predict_text,
)
from rag.profiler.prompts import FACETS, LINGUISTIC_LINES, TRAIT_FULL_NAME
from utils.parser import extract_reasoned_full

# Demo configuration (the requirements' resolved TODOs).
PROMPT_MODE = "reasoned_rag_def_oneshot_30f"
DEFAULT_TOP_K = 3
DEFAULT_MODEL = "gpt-4o-mini"
PROFILER_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 1024

# Where predict_text's prompt/response transcript is appended (one rolling file
# for the whole demo session). Kept off git; handy for debugging a live run.
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILEPATH = os.path.join(_LOG_DIR, "demo_run.log")

# Trait display order (N/E/O/A/C — same grouping the profiler uses).
TRAIT_ORDER = [
    "Neuroticism",
    "Extraversion",
    "Openness",
    "Agreeableness",
    "Conscientiousness",
]

# dataset column <-> full trait name, for ground-truth lookup.
COLUMN_TO_TRAIT = {
    "cNEU": "Neuroticism",
    "cEXT": "Extraversion",
    "cOPN": "Openness",
    "cAGR": "Agreeableness",
    "cCON": "Conscientiousness",
}

# Facet code -> human-readable name, for the profile panel.
_FACET_NAME = {code: name for code, name, *_ in FACETS}
_FACET_TRAIT = {code: trait for code, name, trait, *_ in FACETS}


# A single shared retriever instance. Loading the FAISS index + embedding
# model is the expensive part; we do it once and reuse across requests.
_retriever = None


def get_retriever():
    """Lazily build (and cache) the RAG retriever.

    Raises a clear error if no vector index is present — this is the §5.1
    blocker, surfaced early so the app fails loudly at startup rather than
    mid-demo.
    """
    global _retriever
    if _retriever is None:
        from rag.retriever import FeatureRAGRetriever

        _retriever = FeatureRAGRetriever()
    return _retriever


def warm_up() -> None:
    """Force retriever + index + embedding model to load now.

    Call at server startup so the first live request doesn't pay the
    one-time load cost (and so artifact problems surface before the defense).
    """
    r = get_retriever()
    r._load_index()
    r._load_facets()


def _snippet(text: str, n: int = 320) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= n else text[:n].rstrip() + "…"


def _profile_to_panel(profile: Dict) -> Dict:
    """Reshape a parsed profile into the Stage-1 panel structure.

    Groups the 30 facets by trait and carries the linguistic fingerprint
    in its fixed key order.
    """
    facets = profile.get("facets", {})
    ling = profile.get("linguistic", {})

    grouped: Dict[str, list] = {full: [] for full in TRAIT_FULL_NAME.values()}
    for code, name, trait_code, _defn in FACETS:
        f = facets.get(code, {})
        grouped[TRAIT_FULL_NAME[trait_code]].append(
            {
                "code": code,
                "name": name,
                "signal": f.get("signal", ""),
                "evidence": f.get("evidence", ""),
            }
        )

    linguistic = [
        {"key": key, "value": ling.get(key, "")} for key, _ in LINGUISTIC_LINES
    ]

    return {
        "valid": bool(profile.get("valid")),
        "by_trait": grouped,
        "linguistic": linguistic,
    }


def run_pipeline(
    text: str,
    ground_truth: Optional[Dict[str, str]] = None,
    top_k: int = DEFAULT_TOP_K,
    model_name: str = DEFAULT_MODEL,
) -> Iterator[Dict]:
    """Run one text through the full pipeline, yielding stage events.

    Mirrors the per-record logic of `ptd_model.predict.predict` but for a
    single text, and emits intermediate artifacts for every stage instead of
    only writing final labels to a CSV.
    """
    text = (text or "").strip()
    if not text:
        yield {"stage": "error", "message": "Empty input text."}
        return

    result: Dict = {
        "input_text": text,
        "ground_truth": ground_truth,
        "profile": None,
        "traits": {},
    }

    # ---- Stage 1: profiler -------------------------------------------------
    log.info("Stage 1 — profiling text (%d chars) with %s", len(text), PROFILER_MODEL)
    yield {"stage": "profile", "status": "running"}

    query_profile_text = None
    query_profile_dict = None
    profile_note = None
    try:
        profile = _call_profiler_for_text(text, model_name=PROFILER_MODEL)
        if profile.get("valid"):
            query_profile_text = _profile_to_full_text(profile)
            query_profile_dict = profile
            log.info("Stage 1 — profile valid")
        else:
            profile_note = "profile invalid → raw fallback"
            log.warning("Stage 1 — profile invalid, falling back to raw text")
        panel = _profile_to_panel(profile)
    except Exception as exc:
        profile_note = f"profiler error → raw fallback ({exc})"
        log.error("Stage 1 — profiler failed:\n%s", traceback.format_exc())
        panel = {"valid": False, "by_trait": {}, "linguistic": []}

    panel["note"] = profile_note
    result["profile"] = panel
    yield {"stage": "profile", "status": "done", "profile": panel}

    # ---- Stages 2 & 3: per-trait retrieval + reasoned prediction -----------
    retriever = get_retriever()

    for i, trait_name in enumerate(TRAIT_ORDER, start=1):
        rag_trait = TRAIT_TO_RAG_NAME[trait_name]

        # Stage 2: retrieval
        log.info("Stage 2 — retrieving top-%d for trait %s (%d/%d)", top_k, trait_name, i, len(TRAIT_ORDER))
        yield {
            "stage": "retrieval",
            "status": "running",
            "trait": trait_name,
            "index": i,
            "total": len(TRAIT_ORDER),
        }

        try:
            retrieved = retriever.retrieve(
                posts=text,
                trait=rag_trait,
                top_k=top_k,
                profile_text=query_profile_text,
                query_profile_dict=query_profile_dict,
            )
            log.info("Stage 2 — retrieved %d results for %s", len(retrieved), trait_name)
        except Exception as exc:
            log.error("Stage 2 — retrieval failed for %s:\n%s", trait_name, traceback.format_exc())
            yield {"stage": "error", "message": f"Retrieval failed for {trait_name}: {exc}"}
            return

        retrieved_view = [
            {
                "user_id": r.get("user_id", "?"),
                "label": r.get("label", "?"),
                "scores": r.get("scores", {}),
                "snippet": _snippet(r.get("posts_raw", "")),
            }
            for r in retrieved
        ]
        yield {
            "stage": "retrieval",
            "status": "done",
            "trait": trait_name,
            "retrieved": retrieved_view,
        }

        # Stage 3: reasoned prediction for this trait.
        log.info("Stage 3 — predicting %s (%d/%d) with %s", trait_name, i, len(TRAIT_ORDER), model_name)
        yield {
            "stage": "predict",
            "status": "running",
            "trait": trait_name,
            "index": i,
            "total": len(TRAIT_ORDER),
        }

        try:
            usr_prompt, sys_prompt = build_prompt(
                prompt_mode=PROMPT_MODE,
                trait_name=trait_name,
                retriever=retriever,
                query_text=text,
                query_profile_text=query_profile_text,
                query_profile_dict=query_profile_dict,
                top_k=top_k,
            )
            os.makedirs(_LOG_DIR, exist_ok=True)
            output = predict_text(
                text=text,
                trait_name=trait_name,
                model_name=model_name,
                usr_prompt=usr_prompt,
                sys_prompt=sys_prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                record_idx=0,
                log_filepath=LOG_FILEPATH,
                temperature=TEMPERATURE,
            )
            log.info("Stage 3 — got response for %s (%d chars)", trait_name, len(output or ""))
        except Exception as exc:
            log.error("Stage 3 — prediction failed for %s:\n%s", trait_name, traceback.format_exc())
            yield {"stage": "error", "message": f"Prediction failed for {trait_name}: {exc}"}
            return

        parsed = extract_reasoned_full(output)

        trait_obj = {
            "retrieved": retrieved_view,
            "evidence": parsed.get("evidence"),
            "facet_check": parsed.get("facet_check"),
            "example_alignment": parsed.get("example_alignment"),
            "verdict": parsed.get("verdict"),
            "label": parsed.get("label"),
        }
        result["traits"][trait_name] = trait_obj

        yield {
            "stage": "predict",
            "status": "done",
            "trait": trait_name,
            "trait_result": trait_obj,
        }

    # ---- Stage 4: verdict --------------------------------------------------
    yield {"stage": "done", "result": result}

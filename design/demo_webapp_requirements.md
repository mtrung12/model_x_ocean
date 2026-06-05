# Demo Web App — Requirements

**Purpose:** A small local web app for the bachelor thesis presentation that runs one
piece of text through the full OCEAN-prediction pipeline and visualises every
intermediate stage (profile → retrieval → reasoned prediction → verdict).

**Status:** Draft requirements only. No code written yet. Open decisions are marked `TODO`.

---

## 1. Goal & scope

### 1.1 What it must do
- Take a single essay/text as input.
- Run it through the existing pipeline (no re-implementation of model logic — wrap
  `ptd_model/predict.py` and `rag/` code).
- Show the **4 pipeline stages** as separate, readable panels:
  1. **Profile** — the 30-facet NEO-PI-R profile the profiler LLM emits.
  2. **Retrieval** — the top-k similar labeled training essays + their fused/dense/facet scores.
  3. **Reasoned prediction** — per-trait evidence / facet_check / verdict / label.
  4. **Verdict** — the 5 Big-Five predictions (high/low), shown vs. ground truth when available.
- Be reliable enough to run live during a thesis defense.

### 1.2 Explicit non-goals
- ❌ **No web crawling / scraping.** Decided against: real text+OCEAN-label data is not
  crawlable (it only exists in static research datasets), there'd be no ground-truth label
  to compare against, and live scraping is fragile + raises consent/ethics questions in a defense.
- ❌ No training, no index building, no batch evaluation in the app (that stays in notebooks).
- ❌ No authentication, multi-user support, or persistence/database.
- ❌ No deployment to a public server (runs locally on the presenter's laptop).

---

## 2. Architecture

```
Browser (single static page)            FastAPI backend (localhost)
┌─────────────────────────┐            ┌────────────────────────────────────┐
│ - text input / essay pick│  POST /predict   │ 1. profiler  (rag.profiler.prompts) │
│ - "Run" button           │ ───────────────▶ │ 2. retriever (rag.retriever)        │
│ - 4 stage panels         │            │ 3. reasoned predict (ptd_model)      │
│ - verdict table          │ ◀─────────────── │    → returns JSON of all stages      │
└─────────────────────────┘   stream/JSON    └────────────────────────────────────┘
```

- **Backend:** FastAPI (Python). Wraps existing pipeline functions directly.
- **Frontend:** single static `index.html` + vanilla JS + CSS. No npm/build step.
- **Runtime:** local only (`uvicorn` on `127.0.0.1:<port>`). `TODO: choose port (e.g. 8000).`
- **Mode:** **always live** — every request hits the OpenAI API in real time
  (no result cache). See risks in §7.

### 2.1 Suggested file layout (proposed, not final)
```
demo/
  app.py            # FastAPI app + endpoints
  pipeline.py       # thin orchestration calling existing predict/profiler/retriever code
  static/
    index.html
    app.js
    style.css
  README.md         # how to run the demo
```
`TODO: confirm folder name "demo/" and whether it lives at repo root.`

---

## 3. Functional requirements

### 3.1 Input
`TODO (UNDECIDED): input mode not chosen yet.` Candidate modes — pick one or both:
- **(A) Pick a labeled test essay** from a dropdown sourced from `data/split/essays/test.csv`
  (or `test50.csv` / `minor_test.csv`). These rows have ground-truth columns
  `cEXT,cNEU,cAGR,cCON,cOPN` → enables predicted-vs-true ✓/✗ display.
- **(B) Free-text paste** — examiner/presenter pastes any writing. No ground-truth label;
  verdict panel shows predictions only.
- **(C) Both** — dropdown + textarea.

`TODO: decide A / B / C. Recommendation on record: C (both), as it is most flexible.`

Input constraints to define:
- `TODO: min / max text length (profiler prompt assumes essay-length text).`
- `TODO: behavior on empty input (disable Run button).`

### 3.2 Processing (per request)
For the submitted text, the backend must:
1. Call the **profiler** once (`_call_profiler_for_text`, model `gpt-4o-mini` by default)
   → 30-facet profile dict. On invalid profile, fall back to raw-text path (mirror
   existing `predict.py` behavior) and surface a "profile invalid → raw fallback" note.
2. For **each of the 5 traits**, build the prompt via `build_prompt(...)` with
   `prompt_mode="reasoned_rag_def_oneshot_30f"`, which internally calls the **retriever**
   (top_k similar essays + `scores` dict).
3. Run the **reasoned LLM prediction** per trait (`predict_text`), parse with
   `extract_reasoned_full` → label + evidence/facet_check/verdict.
4. Return all intermediate artifacts (not just final labels) to the frontend.

- `TODO: choose prediction model_name for the demo (e.g. "gpt-4o" vs "gpt-4o-mini").`
- `TODO: choose top_k (existing default 3).`
- `TODO: confirm temperature (existing runs use 0.0).`

### 3.3 Output panels
- **Panel 1 — Profile:** render the 30 facets grouped by trait (N/E/O/A/C), each row =
  `code | facet name | signal (high/mod/low/none/n/e) | evidence paraphrase`, plus the
  linguistic fingerprint lines. (Schema: `FACETS` and `LINGUISTIC_LINES` in
  `rag/profiler/prompts.py`.)
- **Panel 2 — Retrieval:** for the selected/each trait, list top_k retrieved essays:
  user_id, label, and the `scores` dict (`fused`, `dense`, `facet`). Show a short snippet
  of each retrieved essay. `TODO: show retrieval for all 5 traits or one selected trait?`
- **Panel 3 — Reasoned prediction:** per trait, show `evidence`, `facet_check`,
  `example_alignment`, `verdict`, and final `label`.
- **Panel 4 — Verdict:** table of 5 traits → predicted high/low. If a labeled essay was
  used, add a "true" column and ✓/✗ per trait + overall accuracy for that essay.

### 3.4 Progressive reveal (nice-to-have)
- Stream stages so the UI fills in as each completes ("Profiling… → Retrieving… →
  Predicting trait 1/5…"). Implementation via FastAPI `StreamingResponse` (SSE/NDJSON).
- `TODO: decide streaming vs. single JSON response. Streaming improves perceived latency
  for the ~6 API calls/run but adds frontend complexity. Acceptable to ship single-JSON
  first and add streaming later.`

---

## 4. API contract (draft)

`POST /predict`
```jsonc
// request
{ "text": "...", "essay_id": null,        // essay_id set when picked from dropdown
  "top_k": 3, "model_name": "gpt-4o" }
// response
{
  "input_text": "...",
  "ground_truth": { "Openness": "high", ... } | null,
  "profile": { "facets": {...}, "linguistic": {...}, "valid": true },
  "traits": {
    "Openness": {
      "retrieved": [ { "user_id": "...", "label": "high",
                       "scores": {"fused":..,"dense":..,"facet":..},
                       "snippet": "..." } ],
      "evidence": "...", "facet_check": "...", "verdict": "...",
      "label": "high"
    }
    // ... 4 more traits
  }
}
```
`GET /essays` → list of `{id, label_summary, preview}` for the dropdown (mode A/C only).
`TODO: finalize field names; align with what predict.py already returns.`

---

## 5. Dependencies & environment
- Reuses existing code: `ptd_model/predict.py`, `rag/profiler/*`, `rag/retriever.py`,
  `utils/gpt_client.py` (reads `OPENAI_API_KEY` from `.env`).
- New deps: `fastapi`, `uvicorn`. (`openai`, `faiss`, `pandas`, `sentence-transformers`,
  `python-dotenv` already used by the pipeline.)
- `TODO: create a requirements file — repo currently has NO requirements.txt/pyproject.`
- **`OPENAI_API_KEY` must be set in `.env` on the presentation laptop.** Never commit it.

### 5.1 ⚠️ Hard blocker — missing model artifacts on this machine
- `models/` is **empty** here. The RAG path needs the finetuned SBERT model
  (`models/sbert_essays_finetuned`) and/or a FAISS index. `rag/retriever.py` looks for:
  - dual index: `data/vector_db/essays_dual/vectors.faiss`
  - finetuned: `models/rag_artifacts/train_index.faiss`
  `data/vector_db/essays_dual/` exists but the `models/` artifacts do not.
- `TODO (BLOCKER): confirm which vector index + embedding model the demo will use, and
  ensure those artifacts are present on the presentation laptop. Verify retriever
  initialises before building the app.`

---

## 6. UX / presentation requirements
- Single screen, large readable fonts (projector-friendly).
- Clear stage separation with labeled headers (Stage 1…4).
- A visible "Running…" state per stage (each run takes ~15–30s for 1 profiler + 5 reasoned calls).
- One pinned **default essay** (known-good, predicts correctly) preloaded so the first demo
  is reproducible. `TODO: pick the specific essay id once index is available.`
- `TODO: branding — thesis title / name in header? optional.`

---

## 7. Risks & mitigations (always-live mode)
| Risk | Impact | Mitigation |
|------|--------|-----------|
| OpenAI API latency/timeout mid-demo | Stall during defense | Progressive reveal so wait is visible/intentional; have a screen-recording backup |
| Network failure at venue | Demo dead | **Strongly recommended:** record a full successful run beforehand as fallback video |
| API cost | Minor | ~6 calls/run; negligible, but avoid spamming Run |
| Missing model artifacts (§5.1) | Pipeline can't init | Resolve BLOCKER before building |
| Non-deterministic output | Wrong prediction live | Use temperature 0.0; pin a known-good default essay |
| API key leak | Security | Keep key in `.env`, never commit, clear terminal scrollback |

> Note: user chose **always-live** (no cache). A cached/"demo-mode" fallback was advised
> but declined. Recording a backup run is the minimum safety net and is strongly recommended.

---

## 8. Open decisions (TODO summary)
1. Input mode A / B / C (§3.1). *(undecided)*
2. Resolve missing model artifacts BLOCKER (§5.1).
3. Prediction model_name, top_k, temperature for the demo (§3.2).
4. Retrieval panel: per-trait vs. single selected trait (§3.3).
5. Streaming vs. single JSON response (§3.4).
6. Port, folder name/location, requirements file (§2, §5).
7. Pinned default essay id (§6).
8. Min/max input length + empty-input handling (§3.1).
```
```

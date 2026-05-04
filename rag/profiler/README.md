# `rag.profiler` — 30-facet psychological profile generator

Generates a structured psychological profile (~1000 tokens) per essay,
keyed to the 30 NEO-PI-R facets (Costa & McCrae, 1992) plus a short
LIWC-style linguistic fingerprint (Pennebaker & King, 1999; Schwartz
et al., 2013).

The profile is the retrieval substrate for a profile-based RAG pipeline:
profiles get embedded and indexed, retrieval at inference time happens
in profile-vs-profile space rather than raw-text-vs-raw-text space.

## Why this exists

Raw-text RAG on this dataset retrieves topically-similar essays
(school, sorority, football) rather than trait-similar essays. Per-sample
label match rate sits at 51-53%, barely above random. Profile-based
embedding aims to put trait-similar essays close in vector space.

## Usage

```python
import pandas as pd
from rag.profiler.runner import build_profiles

train = pd.read_csv("data/split/essays/train.csv")
build_profiles(
    data=train,
    output_dir="data/profile_db/essays",
    model_name="gpt-4o",        # strong model — generates once, used forever
    use_labels=True,            # labels used as loose anchors
)
```

## Files

| File | Purpose |
|---|---|
| `prompts.py` | Schema (30 facets, 7 linguistic lines), prompt templates, output parser |
| `store.py`   | Append-safe JSONL persistence (`ProfileStore`) |
| `runner.py`  | Driver that profiles a DataFrame, with checkpoint + retry logic |

## Bias-reduction in the profiler prompt

The profiler is **label-aware** — the writer's five trait labels are
shown to it. This is a deliberate calibration choice, not leakage,
because the prompt explicitly forbids label-justified reasoning.
Concrete techniques layered into the system prompt:

1. **Counterfactual scanning per facet.** For every facet, the model
   must scan the text for evidence on both the high side and the low
   side before deciding a signal. This prevents the "label says high so
   write high" collapse that label-conditioned generation usually
   produces.
2. **Anti-genre-bias clause.** The prompt names the specific failure
   mode that produces the Conscientiousness mode-collapse seen in raw
   classification: informal student-journal register pushes C2/C5/C6
   toward "low" and N1 toward "high" by *style alone*. The clause
   demands content-level evidence for these facets.
3. **First-class `n/e` (not-evidenced) value.** A1 trust, A3 altruism,
   E1 warmth, E2 gregariousness are typically unobservable in solo
   introspective text. Allowing `n/e` removes the pressure to fabricate.
4. **Mandatory paraphrase anchoring.** Evidence cell must be a 12-20
   word paraphrase of concrete behaviour or stated belief. Abstract
   trait inferences are forbidden.
5. **No-justification-by-label rule.** The prompt explicitly forbids
   writing "because the label is X" as evidence.
6. **Closing self-check.** After emitting all 30 lines, the model
   demotes any line whose evidence would not survive a skeptical reader.

## Output schema

Each profile is a dict written to JSONL:

```json
{
  "user_id": "user_42",
  "trait_labels": {"cOPN": "high", "cCON": "low", ...},
  "facets": {
    "N1": {"signal": "high", "evidence": "..."},
    "N2": {"signal": "n/e",  "evidence": ""},
    ...
    "C6": {"signal": "low",  "evidence": "..."}
  },
  "linguistic": {
    "pronouns":  "...",
    "emotion":   "...",
    "tense":     "...",
    "cognitive": "...",
    "social":    "...",
    "register":  "...",
    "length":    "..."
  },
  "raw":   "<full profiler output>",
  "valid": true,
  "model": "gpt-4o-..."
}
```

`signal` ∈ `{high, mod, low, none, n/e}` (controlled vocabulary):
- `high` / `low` — clear on-text evidence
- `mod`          — mixed or weak evidence
- `none`         — facet observable but writer is neutral / absent on it
- `n/e`          — facet not observable in this writing genre

## Validation before committing to the full corpus

After profiling ~100 essays, run these three cheap checks before
profiling the rest of the corpus:

1. **Logistic-regression sanity check.** Encode signals as numeric
   (high=+1, mod=+0.5, none=0, low=-0.5, n/e=0) and train a
   per-trait logistic regression on facet signals to predict the
   binary trait label. If you can't beat 0.55 accuracy, the profiles
   aren't capturing trait signal; revise the schema before scaling up.
2. **Self-consistency.** Profile 20 essays twice with `temperature=0`.
   Facet-signal agreement should be ≥85%. If it isn't, tighten the
   prompt or use a stronger model.
3. **Class-balance per facet.** If C5 self-discipline comes out "low"
   in >85% of profiles, the profiler is mode-collapsed the same way
   the classifier was. Stop and fix before propagating.

## Inference-time slicing

Don't show the LLM the full 30-facet grid at inference. Use
`prompts.slice_profile_for_trait(profile, trait_code)` to emit a
~120-180 token brief with only the trait-relevant facets and
linguistic lines.

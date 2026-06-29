# Personality Trait Detection from Text with Profile-Based RAG

Graduation thesis project. The system predicts the **Big Five / OCEAN**
personality traits (Openness, Conscientiousness, Extraversion,
Agreeableness, Neuroticism) from a person's writing, using large language
models augmented with a **profile-based retrieval-augmented generation
(RAG)** pipeline, and compares them against classical ML/DL baselines.

The central idea: instead of retrieving topically-similar text (raw-text
RAG, which retrieves essays about *the same subject* rather than *the same
personality*), each text is first converted into a structured 30-facet
psychological profile (NEO-PI-R facets), and retrieval happens in
profile-vs-profile space. See [rag/profiler/README.md](rag/profiler/README.md)
for the rationale and prompt design.

## Repository layout

| Path | Contents |
|---|---|
| [ptd_model/](ptd_model/) | Main model: reasoned RAG prediction pipeline (`predict`, `evaluate`, prompts) |
| [rag/](rag/) | Retrieval stack — embedder, FAISS index, dual/facet retrievers, profiler |
| [rag/profiler/](rag/profiler/) | 30-facet psychological profile generator (the retrieval substrate) |
| [baselines/](baselines/) | LLM prompting baselines (zero/one-shot, CoT) |
| [baseline_notebook_ML_DL/](baseline_notebook_ML_DL/) | Classical baselines: SVM, LSTM, BERT |
| [utils/](utils/) | LLM clients (OpenAI / HF / Ollama), parsing, logging |
| [notebook/](notebook/) | All experiments: ablations, RAG comparisons, index building, per-model runs |
| [scripts/](scripts/) | Standalone analysis scripts (e.g. false-prediction extraction) |
| [data/](data/) | Datasets, splits, profile stores, and prebuilt vector DBs (see below) |
| [stats/](stats/) | F1 / retrieval-accuracy result trackers (`.xlsx`) |
| [design/](design/) | Pipeline diagrams and demo web-app requirements |
| [related_paper+bachelor_ref/](related_paper+bachelor_ref/) | Reference papers |

## Data

The primary corpus is the **Essays (stream-of-consciousness)** dataset (2,468
student essays, five binary OCEAN labels from a median split); **myPersonality**
is a secondary corpus. The Essays corpus is split into three non-overlapping
partitions under [data/split/essays/](data/split/essays/):

| Split | File | n | Role |
|---|---|---|---|
| Train | `train.csv` | 1,974 | retrieval pool / exemplars (index is built from this) |
| Validation | `val.csv` | 247 | retrieval-strategy ablation; a 50-essay subset drives per-trait gamma selection |
| Test | `test.csv` | 247 | final classification evaluation |

The committed `data/` tree also ships the **prebuilt profile stores**
([data/profile_db/](data/profile_db/)) and **FAISS vector databases**
([data/vector_db/](data/vector_db/)), so the prediction/evaluation pipelines run
out of the box — **Stages 1–2 below can be skipped** unless you want to rebuild
from scratch.

## Models used

| Role | Model | Notes |
|---|---|---|
| Profiler | `gpt-4o-mini` | label-aware on train, **label-blind on val/test** (no leakage) |
| Sentence encoder | `nomic-ai/nomic-embed-text-v1.5` | dual fusion `α=0.5` of raw-essay + profile embeddings (downloaded on first use, ~270 MB) |
| Classifier | `gpt-4o-mini` (main) / `meta-llama/Meta-Llama-3-8B-Instruct` (alt) | 5 reasoned XML calls per essay (one per trait) |

Defaults: `prompt_mode="reasoned_rag_def_oneshot_30f"`, `top_k=5`,
`temperature=0.0`. The proposed retriever is **hybrid-facet** (dense dual
embedding + per-trait trait-masked facet re-ranking), backed by
`data/vector_db/essays_dual/`.

## Setup

```bash
# 1. Create an environment (Python 3.10+ recommended)
python -m venv .venv && source .venv/bin/activate   # or conda

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
#   then edit .env and add your OPENAI_API_KEY (and HF_TOKEN if running LLaMA)
```

## Running the pipeline end-to-end

The pipeline has two phases — an **index-building phase** (Stages 1–2, run once
over the training corpus) and an **inference phase** (Stage 3, run per test
essay). All steps are driven from Python or the experiment notebooks.

### Stage 1 — Create the 30-facet profiles

Profile the train split (labels shown as loose anchors) and the val/test splits
**label-blind**. See [rag/profiler/](rag/profiler/) for prompt design.

```python
import pandas as pd
from rag.profiler.runner import build_profiles

# Training profiles (label-aware) -> data/profile_db/essays/
build_profiles(
    data=pd.read_csv("data/split/essays/train.csv"),
    output_dir="data/profile_db/essays",
    model_name="gpt-4o-mini",
    use_labels=True,
)

# Test profiles (LABEL-BLIND, to avoid leakage) -> data/profile_db/essays_test/
build_profiles(
    data=pd.read_csv("data/split/essays/test.csv"),
    output_dir="data/profile_db/essays_test",
    model_name="gpt-4o-mini",
    use_labels=False,
)
```

### Stage 2 — Build the index

Build the dual-embedding FAISS index plus the 30-d facet-vector matrix (the
hybrid-facet runtime index). See [rag/runners/build_features.py](rag/runners/build_features.py).

```python
from rag.runners.build_features import build_index

build_index(
    data=pd.read_csv("data/split/essays/train.csv"),
    profile_store_path="data/profile_db/essays/profile_store.jsonl",
    output_dir="data/vector_db/essays_dual",   # writes vectors.faiss, vectors_meta.jsonl, facet_vectors.npy
)
```

Notebook equivalents under [notebook/gpt/](notebook/gpt/): `build_dual_index.ipynb`
(proposed / hybrid), `build_sliced_dual_index.ipynb`, `build_hybrid_index.ipynb`
(other retrieval strategies used in the ablation).

### Stage 3 — Run the main pipeline (predict + evaluate)

```python
import pandas as pd
from ptd_model.predict import predict
from ptd_model.evaluate import evaluate

test = pd.read_csv("data/split/essays/test.csv")

run_id, run_time, prediction_csv = predict(
    text_df=test,
    model_name="gpt-4o-mini",
    prompt_mode="reasoned_rag_def_oneshot_30f",   # hybrid RAG + XML reasoning
    vector_db_dir="data/vector_db/essays_dual",
    profiler_model="gpt-4o-mini",                 # profiles each test essay on the fly
    top_k=5,
    max_new_tokens=1024,
    temperature=0.0,
    log_dir="log",
    res_dir="result",
)

evaluate(
    prediction_csv=prediction_csv,
    model_name="gpt-4o-mini",
    prompt_mode="reasoned_rag_def_oneshot_30f",
    run_id=run_id,
    run_time=run_time,
    res_dir="result",
)
```

Results are written under `result/<model>/<prompt_mode>/<run_id>/`
(`predictions.csv`, `evaluation_summary.csv`, per-trait reports,
`reasoning_log.jsonl`) and logs under `log/` (both gitignored). The runnable
notebook version is [notebook/ablation/predict_ablation.ipynb](notebook/ablation/predict_ablation.ipynb)
(main run + analysis) or [notebook/gpt/rag_profile_half2_predict.ipynb](notebook/gpt/rag_profile_half2_predict.ipynb).

## Reproducing the thesis experiments

| Experiment | Where | Notes |
|---|---|---|
| **LLM prompting baselines** (zero/one-shot, CoT) | [baselines/](baselines/) `predict.py` (`prompt_mode` ∈ `zeroshot`/`oneshot`/`cot`); notebooks [notebook/gpt/](notebook/gpt/) & [notebook/llama/](notebook/llama/) (`zeroshot`,`oneshot`,`cot`) | no retrieval |
| **Classical / neural baselines** | [baseline_notebook_ML_DL/](baseline_notebook_ML_DL/) — `SVM.ipynb`, `LSTM.ipynb`, `BERT.ipynb` (BERT/RoBERTa/DeBERTa) | supervised, per-trait |
| **Retrieval ablation** (5 strategies, MMR/HR) | [notebook/rag_compare/](notebook/rag_compare/) `rag_retrieve_accuracy_*.ipynb` | on validation split |
| **Retrieval → downstream F1** per strategy | [notebook/ablation/](notebook/ablation/) `rag_ablation_*.ipynb` | raw-post / profile / sliced-dual / dual / hybrid-facet |
| **Gamma sweep** (per-trait `γ` selection) | [notebook/ablation/gamma_sweep.ipynb](notebook/ablation/gamma_sweep.ipynb) | `γ ∈ {0.1,0.3,0.5,0.7,0.9}` on 50-essay subset |
| **Component ablation** (no-RAG / no-reasoning / no-antibias) | [notebook/ablation_main_pipe/](notebook/ablation_main_pipe/) | keeps everything else fixed |
| **Classifier backbone comparison** (GPT-4o-mini vs LLaMA-3-8B) | [notebook/llama/rag_profile_half2_predict.ipynb](notebook/llama/rag_profile_half2_predict.ipynb) | same pipeline, swapped classifier |

## Environment variables

| Variable | Needed for |
|---|---|
| `OPENAI_API_KEY` | All GPT pipelines (profiler + classifier) |
| `HF_TOKEN` | Hugging Face / LLaMA-3-8B pipelines ([notebook/llama/](notebook/llama/)) |

## License

See [LICENSE](LICENSE).

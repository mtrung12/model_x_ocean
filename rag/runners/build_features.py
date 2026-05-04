import gc
import os
import time
import pandas as pd
import numpy as np

from utils.gpt_client import gpt_call
from utils.hf_client import hf_call
from utils.log import log_to_file

from rag.extractor import build_extractor_prompts, parse_extractor_output, CATEGORIES, EXTRACTOR_MAX_TOKENS
from rag.store import FeatureStore


LLAMA_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

TRAIT_MAP = {
    "cEXT": "Extraversion",
    "cNEU": "Neuroticism",
    "cAGR": "Agreeableness",
    "cCON": "Conscientiousness",
    "cOPN": "Openness to Experience",
}


def _call_extractor(text, model_name, log_filepath=None):
    sys_p, usr_p = build_extractor_prompts(text)
    if not model_name.startswith("gpt"):
        try:
            raw = hf_call(usr_p, sys_p, model_name, EXTRACTOR_MAX_TOKENS, 0.0)
            result = parse_extractor_output(raw)
            if log_filepath:
                log_to_file(log_filepath, sys_p, usr_p, raw, "extractor-llama")
            return result
        except Exception:
            pass
    try:
        raw = gpt_call(usr_p, sys_p, "gpt-4o-mini", EXTRACTOR_MAX_TOKENS, 0.0)
        result = parse_extractor_output(raw)
        if log_filepath:
            log_to_file(log_filepath, sys_p, usr_p, raw, "extractor-gpt")
        return result
    except Exception:
        return {cat: "" for cat in CATEGORIES}


def _extract_trait_labels(row):
    labels = {}
    for col, trait_name in TRAIT_MAP.items():
        if col in row.index:
            raw = str(row[col]).strip().lower()
            if raw in ("high", "low"):
                labels[trait_name] = raw
    return labels


def build_features(data, output_dir="data/vector_db/essays", model_name=LLAMA_MODEL, log_dir=None,
                   embed_model_dir=None):
    """Build feature store + FAISS index from ``data``.

    Parameters
    ----------
    embed_model_dir:
        Optional path to a fine-tuned SentenceTransformer model directory.
        When provided, overrides ``rag.embedder.FINETUNED_MODEL_DIR`` so the
        index is built with the same model used at retrieval time.
    """
    # Lazy import: sentence_transformers only needed when actually running
    import rag.embedder as _embedder
    if embed_model_dir is not None:
        _embedder.FINETUNED_MODEL_DIR = embed_model_dir
    from rag.embedder import get_embedding

    os.makedirs(output_dir, exist_ok=True)

    if log_dir is None:
        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        log_dir = os.path.join(_root, "log", "vector_db")
    os.makedirs(log_dir, exist_ok=True)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_filepath = os.path.join(log_dir, f"{run_id}.log")

    store = FeatureStore(os.path.join(output_dir, "feature_store.jsonl"))
    store.load()

    print(f"[{run_id}] Building features -- output: {output_dir}, model: {model_name}")
    print(f"[{run_id}] Logs: {log_filepath}")

    n = len(data)
    t0 = time.time()
    errors = 0

    for i, row in data.iterrows():
        user_id = f"user_{i}"
        if user_id in store.entries:
            continue

        text = str(row["text"])
        try:
            features = _call_extractor(text, model_name, log_filepath)
        except Exception:
            features = {cat: "" for cat in CATEGORIES}
            errors += 1

        trait_labels = _extract_trait_labels(row)
        store.add(user_id, trait_labels, features)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 1
            remaining = (n - i - 1) / rate
            print(f"  [build_features] {i + 1}/{n} done. {errors} errors. ETA {remaining:.0f}s")

        if (i + 1) % 100 == 0:
            store.save()

        gc.collect()

    store.save()
    total = time.time() - t0
    print(f"[build_features] Done. {n} records in {total:.1f}s. {errors} errors.")

    print("[build_features] Building FAISS index...")
    _build_index(data, store, output_dir)
    print("[build_features] Index ready.")


def _build_index(data, store, output_dir):
    # Lazy imports: heavy deps only needed at index-build time
    from rag.faiss_index import FAISSIndex
    from rag.embedder import get_embedding

    index_path = os.path.join(output_dir, "vectors.faiss")
    meta_path = os.path.join(output_dir, "vectors_meta.jsonl")

    if os.path.exists(index_path) and os.path.exists(meta_path):
        print("  [index] Already exists, skipping rebuild.")
        return

    print("  [index] Embedding all texts...")
    texts = data["text"].astype(str).tolist()
    embeddings = get_embedding(texts)
    embeddings = np.array(embeddings, dtype="float32")

    meta = []
    for i, row in data.iterrows():
        user_id = f"user_{i}"
        fd = store.get(user_id)
        trait_labels = _extract_trait_labels(row)
        meta.append({
            "user_id": user_id,
            "posts_raw": str(row["text"]),
            "trait_labels": trait_labels,
            "features": fd["features"] if fd else {},
        })

    idx = FAISSIndex(dimension=embeddings.shape[1])
    idx.build(embeddings, meta)
    idx.save(index_path, meta_path)
    print(f"  [index] Saved ({len(meta)} entries).")

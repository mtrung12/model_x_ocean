"""Build the dual-embedding FAISS index from a training DataFrame.

Each vector is:
    normalise( ALPHA * embed(raw_text) + (1-ALPHA) * embed(profile_text) )

The profile text is loaded from a ProfileStore (built by rag/profiler/runner.py).
If a user_id has no valid profile the raw embedding is used as fallback (alpha=1.0).

Usage
-----
    import pandas as pd
    from rag.runners.build_features import build_index

    train = pd.read_csv("data/split/essays/train.csv")
    build_index(
        data=train,
        profile_store_path="data/profile_db/essays/profile_store.jsonl",
        output_dir="data/vector_db/essays_dual",
    )
"""

import os
import time
import numpy as np
import pandas as pd

from rag.faiss_index import FAISSIndex
from rag.profiler.store import ProfileStore
from rag.profiler.prompts import FACETS
from rag.facet_vector import build_facet_matrix, save_facet_matrix

TRAIT_MAP = {
    "cEXT": "Extraversion",
    "cNEU": "Neuroticism",
    "cAGR": "Agreeableness",
    "cCON": "Conscientiousness",
    "cOPN": "Openness to Experience",
}


def _extract_trait_labels(row):
    labels = {}
    for col, trait_name in TRAIT_MAP.items():
        if col in row.index:
            raw = str(row[col]).strip().lower()
            if raw in ("high", "low"):
                labels[trait_name] = raw
            elif raw in ("1", "1.0"):
                labels[trait_name] = "high"
            elif raw in ("0", "0.0"):
                labels[trait_name] = "low"
    return labels


def _profile_to_full_text(profile):
    """Serialise all 30 facets + linguistic block to a plain string for embedding."""
    facets = profile.get("facets", {})
    ling   = profile.get("linguistic", {})
    lines  = []
    for code, _, _, _ in FACETS:
        f = facets.get(code)
        if f and f.get("signal"):
            ev = f.get("evidence", "")
            lines.append(f"{code} | {f['signal']} | {ev}")
    for key, val in ling.items():
        if val:
            lines.append(f"{key}: {val}")
    return "\n".join(lines)


def build_index(
    data,
    profile_store_path,
    output_dir="data/vector_db/essays_dual",
    alpha=None,
    force_rebuild=False,
):
    """Build (or skip if already built) a dual-embedding FAISS index.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain a 'text' column + OCEAN label columns.
    profile_store_path : str
        Path to a ProfileStore JSONL built by rag/profiler/runner.py.
    output_dir : str
        Destination for vectors.faiss + vectors_meta.jsonl.
    alpha : float or None
        Fusion weight for raw text (0-1). None uses embedder.ALPHA default.
    force_rebuild : bool
        Rebuild even if the index already exists.
    """
    from rag.embedder import get_dual_embedding, ALPHA as DEFAULT_ALPHA
    _alpha = DEFAULT_ALPHA if alpha is None else alpha

    index_path = os.path.join(output_dir, "vectors.faiss")
    meta_path  = os.path.join(output_dir, "vectors_meta.jsonl")

    if not force_rebuild and os.path.exists(index_path) and os.path.exists(meta_path):
        print(f"[build_index] Index already exists at {output_dir!r}, skipping rebuild.")
        return

    os.makedirs(output_dir, exist_ok=True)

    store = ProfileStore(profile_store_path)
    store.load()
    print(f"[build_index] Loaded {len(store)} profiles from {profile_store_path!r}")

    n = len(data)
    print(f"[build_index] Embedding {n} essays (alpha={_alpha}) ...")
    t0 = time.time()

    embeddings  = []
    meta        = []
    profile_seq = []
    no_profile  = 0

    for i, row in data.iterrows():
        user_id  = f"user_{i}"
        raw_text = str(row["text"])
        entry    = store.get(user_id)

        if entry and entry.get("valid"):
            profile_text = _profile_to_full_text(entry)
        else:
            profile_text = raw_text
            no_profile  += 1

        vec = get_dual_embedding(raw_text, profile_text, alpha=_alpha)
        embeddings.append(vec)

        trait_labels = _extract_trait_labels(row)
        meta.append({
            "user_id":      user_id,
            "posts_raw":    raw_text,
            "trait_labels": trait_labels,
            "profile":      entry or {},
        })
        profile_seq.append(entry or {})

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (n - i - 1) / rate
            print(f"  {i + 1}/{n}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"[build_index] Embedded {n} essays in {elapsed:.1f}s  "
          f"(no-profile fallbacks: {no_profile})")

    embeddings = np.array(embeddings, dtype="float32")
    idx = FAISSIndex(dimension=embeddings.shape[1])
    idx.build(embeddings, meta)
    idx.save(index_path, meta_path)
    print(f"[build_index] Saved -> {output_dir}/  ({n} vectors, dim={embeddings.shape[1]})")

    facet_path = os.path.join(output_dir, "facet_vectors.npy")
    facet_mat  = build_facet_matrix(profile_seq, normalize=True)
    save_facet_matrix(facet_mat, facet_path)
    print(f"[build_index] Saved facet matrix -> {facet_path}  shape={facet_mat.shape}")

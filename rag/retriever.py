"""RAG retriever backed by a dual-embedding FAISS index.

Supports dense-only and hybrid (dense + 30-d facet cosine) retrieval.
final_score = beta * cos_dense + gamma * cos_facet  (per trait)
"""

from __future__ import annotations

import os
import numpy as np

from .faiss_index import FAISSIndex
from .facet_vector import (
    facet_vector,
    facet_cosine,
    hybrid_score,
    load_facet_matrix,
)

import os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

DUAL_INDEX_DIR = _os.path.join(_PROJECT_ROOT, "data", "vector_db", "essays_dual")
FINETUNED_ARTIFACTS_DIR = _os.path.join(_PROJECT_ROOT, "models", "rag_artifacts")

DEFAULT_TRAIT_WEIGHTS = {
    "Openness to Experience": (0.4, 0.6),
    "Conscientiousness":      (0.3, 0.7),
    "Extraversion":           (0.4, 0.6),
    "Agreeableness":          (0.4, 0.6),
    "Neuroticism":            (0.3, 0.7),
}


class FeatureRAGRetriever:
    """Retriever that uses dual (raw + profile) embeddings.

    Falls back to the finetuned raw-text index when the dual index has not
    been built yet.

    If `use_hybrid=True` and the index directory has a `facet_vectors.npy`
    file, retrieval also incorporates a structured facet-vector cosine
    score per trait. The dense FAISS index still drives a coarse
    candidate fetch; the facet score then re-ranks within that pool.
    """

    def __init__(self, db_dir=None, use_hybrid: bool = True,
                 trait_weights: dict | None = None,
                 hybrid_fetch_multiplier: int = 8):
        dual_faiss = os.path.join(DUAL_INDEX_DIR, "vectors.faiss")

        if db_dir is not None:
            self._mode   = "legacy"
            self._db_dir = db_dir
        elif os.path.exists(dual_faiss):
            self._mode   = "dual"
            self._db_dir = DUAL_INDEX_DIR
        elif os.path.exists(os.path.join(FINETUNED_ARTIFACTS_DIR, "train_index.faiss")):
            self._mode   = "finetuned"
            self._db_dir = FINETUNED_ARTIFACTS_DIR
        else:
            raise FileNotFoundError(
                "No FAISS index found. Run rag/runners/build_features.py first."
            )

        facet_path = os.path.join(self._db_dir, "facet_vectors.npy")
        self._use_hybrid    = bool(use_hybrid and os.path.exists(facet_path))
        self._facet_matrix  = None  # lazy-loaded
        self._facet_path    = facet_path if self._use_hybrid else None
        self._trait_weights = dict(trait_weights or DEFAULT_TRAIT_WEIGHTS)
        self._hybrid_fetch_multiplier = max(1, int(hybrid_fetch_multiplier))

        print(f"[retriever] mode={self._mode!r}  dir={self._db_dir!r}  hybrid={self._use_hybrid}")
        self._index    = None
        self._ft_index = None
        self._ft_meta  = None

    def _load_facets(self):
        if self._facet_matrix is not None or not self._use_hybrid:
            return
        self._facet_matrix = load_facet_matrix(self._facet_path)
        print(f"[retriever] Facet matrix loaded ({self._facet_matrix.shape}).")

    def _load_index(self):
        if self._index is not None or self._ft_index is not None:
            return

        if self._mode == "dual":
            self._index = FAISSIndex(dimension=0)
            self._index.load(
                os.path.join(self._db_dir, "vectors.faiss"),
                os.path.join(self._db_dir, "vectors_meta.jsonl"),
            )
            print(f"[retriever] Dual index loaded ({self._index._index.ntotal} vectors).")

        elif self._mode == "finetuned":
            import faiss
            import pandas as pd
            self._ft_index = faiss.read_index(
                os.path.join(self._db_dir, "train_index.faiss")
            )
            self._ft_meta = pd.read_csv(
                os.path.join(self._db_dir, "train_metadata.csv")
            )
            print(f"[retriever] Finetuned index loaded ({self._ft_index.ntotal} vectors).")

        else:  # legacy
            self._index = FAISSIndex(dimension=0)
            self._index.load(
                os.path.join(self._db_dir, "vectors.faiss"),
                os.path.join(self._db_dir, "vectors_meta.jsonl"),
            )
            print(f"[retriever] Legacy index loaded ({self._index._index.ntotal} vectors).")

    def _embed_query(self, raw_text, profile_text=None):
        if self._mode == "dual" and profile_text:
            from .embedder import get_dual_embedding
            vec = get_dual_embedding(raw_text, profile_text)
        else:
            from .embedder import get_embedding
            vec = get_embedding(raw_text)
        return np.array(vec, dtype="float32")

    def _search(self, query_emb, fetch_n):
        self._load_index()

        if self._mode == "finetuned":
            q = query_emb.reshape(1, -1)
            scores, indices = self._ft_index.search(q, fetch_n)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                row = self._ft_meta.iloc[idx]
                trait_labels = {}
                for col, name in [
                    ("cOPN", "Openness to Experience"),
                    ("cCON", "Conscientiousness"),
                    ("cEXT", "Extraversion"),
                    ("cAGR", "Agreeableness"),
                    ("cNEU", "Neuroticism"),
                ]:
                    if col in row.index:
                        val = str(row[col]).strip().lower()
                        if val in ("high", "low"):
                            trait_labels[name] = val
                        elif val in ("1", "1.0"):
                            trait_labels[name] = "high"
                        elif val in ("0", "0.0"):
                            trait_labels[name] = "low"
                results.append({
                    "user_id":      str(row.get("user_id", f"user_{idx}")),
                    "posts_raw":    str(row.get("text", "")),
                    "trait_labels": trait_labels,
                    "profile":      {},
                    "distance":     float(score),
                })
            return results

        _, results = self._index.search(query_emb, fetch_n)
        return results

    def _hybrid_rerank(self, query_emb, query_profile, trait, top_k):
        self._load_index()
        self._load_facets()

        fetch_n = max(top_k * self._hybrid_fetch_multiplier, 50)

        if self._mode == "finetuned":
            q = np.asarray(query_emb, dtype=np.float32).reshape(1, -1)
            scores, indices = self._ft_index.search(q, fetch_n)
            cand_idx   = [int(i) for i in indices[0] if i >= 0]
            cand_dense = [float(s) for s, i in zip(scores[0], indices[0]) if i >= 0]
        else:
            q = np.asarray(query_emb, dtype=np.float32).reshape(1, -1)
            k = min(fetch_n, self._index._index.ntotal)
            d, i = self._index._index.search(q, k)
            cand_idx = [int(j) for j in i[0] if j >= 0]
            cand_dense = [-float(dd) for dd, jj in zip(d[0], i[0]) if jj >= 0]

        if not cand_idx:
            return []

        full_to_code = {
            "Openness to Experience": "cOPN",
            "Conscientiousness":      "cCON",
            "Extraversion":           "cEXT",
            "Agreeableness":          "cAGR",
            "Neuroticism":            "cNEU",
        }
        trait_code   = full_to_code.get(trait)

        q_facet = facet_vector(query_profile or {}, normalize=True)
        sub_mat = self._facet_matrix[cand_idx]
        cand_facet = facet_cosine(q_facet, sub_mat, trait_code=trait_code)

        dense_arr = np.asarray(cand_dense, dtype=np.float32)
        if dense_arr.size > 1:
            d_min, d_max = float(dense_arr.min()), float(dense_arr.max())
            if d_max > d_min:
                dense_arr = 2.0 * (dense_arr - d_min) / (d_max - d_min) - 1.0
            else:
                dense_arr = np.zeros_like(dense_arr)

        beta, gamma = self._trait_weights.get(trait, (0.5, 0.5))
        fused = hybrid_score(dense_arr, cand_facet, beta=beta, gamma=gamma)

        order = np.argsort(-fused)
        return [
            (cand_idx[k], float(fused[k]), float(dense_arr[k]), float(cand_facet[k]))
            for k in order
        ]

    def retrieve(self, posts, trait, top_k=5, profile_text=None,
                 query_profile_dict=None):
        """Return the top_k nearest training essays for the given trait.

        Parameters
        ----------
        posts              : raw essay text of the test instance.
        trait              : full trait name, e.g. "Conscientiousness".
        top_k              : number of results to return.
        profile_text       : serialised profile summary for the test essay
                             (used by the dual-embedding path).
        query_profile_dict : the parsed test profile dict (with 'facets').
                             Required when use_hybrid=True. If None and
                             hybrid is on, we fall back to dense-only.
        """
        query_emb = self._embed_query(posts, profile_text)

        if self._use_hybrid and query_profile_dict is not None:
            ranked = self._hybrid_rerank(
                query_emb=query_emb,
                query_profile=query_profile_dict,
                trait=trait,
                top_k=top_k,
            )
            self._load_index()
            meta = (
                self._index._meta if self._mode != "finetuned"
                else None
            )
            enriched = []
            for row_idx, fused_s, dense_s, facet_s in ranked:
                if self._mode == "finetuned":
                    row = self._ft_meta.iloc[row_idx]
                    trait_labels = {}
                    for col, name in [
                        ("cOPN", "Openness to Experience"),
                        ("cCON", "Conscientiousness"),
                        ("cEXT", "Extraversion"),
                        ("cAGR", "Agreeableness"),
                        ("cNEU", "Neuroticism"),
                    ]:
                        if col in row.index:
                            v = str(row[col]).strip().lower()
                            if v in ("high", "low"):                trait_labels[name] = v
                            elif v in ("1", "1.0"):                  trait_labels[name] = "high"
                            elif v in ("0", "0.0"):                  trait_labels[name] = "low"
                    if trait not in trait_labels:
                        continue
                    enriched.append({
                        "user_id":   str(row.get("user_id", f"user_{row_idx}")),
                        "posts_raw": str(row.get("text", "")),
                        "label":     trait_labels[trait],
                        "profile":   {},
                        "distance":  fused_s,
                        "scores":    {"fused": fused_s, "dense": dense_s, "facet": facet_s},
                    })
                else:
                    entry = meta[row_idx]
                    if trait not in entry.get("trait_labels", {}):
                        continue
                    enriched.append({
                        "user_id":   entry["user_id"],
                        "posts_raw": entry.get("posts_raw", ""),
                        "label":     entry["trait_labels"][trait],
                        "profile":   entry.get("profile", {}),
                        "distance":  fused_s,
                        "scores":    {"fused": fused_s, "dense": dense_s, "facet": facet_s},
                    })
                if len(enriched) >= top_k:
                    break
            return enriched

        all_results = self._search(query_emb, top_k * 6)
        enriched = []
        for r in all_results:
            if trait not in r.get("trait_labels", {}):
                continue
            enriched.append({
                "user_id":   r["user_id"],
                "posts_raw": r.get("posts_raw", ""),
                "label":     r["trait_labels"][trait],
                "profile":   r.get("profile", {}),
                "distance":  r.get("distance", 0.0),
            })
            if len(enriched) >= top_k:
                break
        return enriched

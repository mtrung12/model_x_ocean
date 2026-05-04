import os
import numpy as np

from .faiss_index import FAISSIndex
from .store import FeatureStore

# ---------------------------------------------------------------------------
# PATHS — update these after running finetune_sbert_essays_kaggle.ipynb
# Paths are relative to the project root (the folder containing the rag/ package).
# ---------------------------------------------------------------------------
import os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

# Path to the RAG artifacts folder saved by the notebook (ARTIFACTS_DIR).
# The folder must contain:
#   train_index.faiss   — FAISS index built from fine-tuned embeddings
#   train_metadata.csv  — CSV with essay texts + OCEAN labels
#   config.json         — training config snapshot (informational)
# Set to None to use the legacy vectors.faiss / vectors_meta.jsonl layout
# produced by rag/runners/build_features.py instead.
FINETUNED_ARTIFACTS_DIR = _os.path.join(_PROJECT_ROOT, "models", "rag_artifacts")
# ---------------------------------------------------------------------------


class FeatureRAGRetriever:
    """Retriever that supports both the fine-tuned notebook artifacts layout
    (FINETUNED_ARTIFACTS_DIR) and the legacy build_features layout (db_dir).

    Priority:
      1. If FINETUNED_ARTIFACTS_DIR is set and the directory exists on disk,
         use the fine-tuned FAISS index + CSV metadata from that folder.
      2. Otherwise fall back to the legacy db_dir layout
         (vectors.faiss + vectors_meta.jsonl + feature_store.jsonl).
    """

    def __init__(self, db_dir="data/vector_db/essays"):
        self.db_dir = db_dir
        self._use_finetuned = (
            FINETUNED_ARTIFACTS_DIR is not None
            and os.path.isdir(FINETUNED_ARTIFACTS_DIR)
            and os.path.exists(os.path.join(FINETUNED_ARTIFACTS_DIR, "train_index.faiss"))
        )

        if self._use_finetuned:
            print(f"[retriever] Using fine-tuned artifacts: {FINETUNED_ARTIFACTS_DIR}")
            self.feature_store = None  # not used in fine-tuned path
            self._train_meta = None   # loaded lazily
        else:
            print(f"[retriever] Fine-tuned artifacts not found; falling back to legacy db_dir: {db_dir}")
            self.feature_store = FeatureStore(os.path.join(db_dir, "feature_store.jsonl"))
            self.feature_store.load()

        self._index = None

    # ------------------------------------------------------------------
    # Index loading
    # ------------------------------------------------------------------

    def _load_index(self):
        if self._index is not None:
            return
        if self._use_finetuned:
            self._load_finetuned_index()
        else:
            self._load_legacy_index()

    def _load_finetuned_index(self):
        import faiss as _faiss
        import pandas as pd

        index_path = os.path.join(FINETUNED_ARTIFACTS_DIR, "train_index.faiss")
        meta_path = os.path.join(FINETUNED_ARTIFACTS_DIR, "train_metadata.csv")
        self._faiss_raw = _faiss.read_index(index_path)
        self._train_meta = pd.read_csv(meta_path)
        # Keep _index as a sentinel so _load_index() is not called again
        self._index = True
        print(f"[retriever] Loaded fine-tuned FAISS index ({self._faiss_raw.ntotal} vectors).")

    def _load_legacy_index(self):
        index_path = os.path.join(self.db_dir, "vectors.faiss")
        meta_path = os.path.join(self.db_dir, "vectors_meta.jsonl")
        self._index = FAISSIndex(dimension=0)
        self._index.load(index_path, meta_path)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, text):
        from .embedder import get_embedding
        return np.array(get_embedding(text), dtype="float32")

    # ------------------------------------------------------------------
    # Internal search helpers (return a uniform list-of-dicts regardless
    # of which backend is active)
    # ------------------------------------------------------------------

    def _search_finetuned(self, query_emb, fetch_n):
        """Search using the fine-tuned FAISS index + CSV metadata."""
        # FAISS IndexFlatIP expects 2-D input
        q = query_emb.reshape(1, -1)
        scores, indices = self._faiss_raw.search(q, fetch_n)
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            row = self._train_meta.iloc[idx]
            # Build trait_labels dict from CSV columns.
            # Supports both string labels ("high"/"low") and integer labels (1/0).
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
                "user_id": str(row.get("user_id", f"user_{idx}")),
                "posts_raw": str(row.get("text", "")),
                "trait_labels": trait_labels,
                "features": {},   # fine-tuned path has no pre-extracted features
                "distance": float(score),
            })
        return results

    def _search_legacy(self, query_emb, fetch_n):
        """Search using the legacy FAISSIndex wrapper."""
        _, results = self._index.search(query_emb, fetch_n)
        return results

    def _search(self, query_emb, fetch_n):
        self._load_index()
        if self._use_finetuned:
            return self._search_finetuned(query_emb, fetch_n)
        return self._search_legacy(query_emb, fetch_n)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, posts, trait, top_k=5):
        query_emb = self._embed(posts)
        all_results = self._search(query_emb, top_k * 6)
        enriched = []
        for r in all_results:
            if trait not in r.get("trait_labels", {}):
                continue
            enriched.append({
                "user_id": r["user_id"],
                "posts_raw": r.get("posts_raw", ""),
                "label": r["trait_labels"][trait],
                "features": r.get("features", {}),
                "distance": r.get("distance", 0.0),
            })
            if len(enriched) >= top_k:
                break
        return enriched

    def build_explainer_context(self, posts, trait, top_k=5):
        query_emb = self._embed(posts)
        all_results = self._search(query_emb, top_k * 6)

        high_lines, low_lines = [], []
        count_high, count_low = 0, 0

        for r in all_results:
            trait_labels = r.get("trait_labels", {})
            if trait not in trait_labels:
                continue
            label = trait_labels[trait]
            features = r.get("features", {})
            posts_raw = r.get("posts_raw", "")

            lines = []
            for cat in ["Emotion", "Cognition", "Sensory Perception", "Sociality"]:
                ev = features.get(cat, "").strip()
                if ev:
                    lines.append(f"  {cat}: {ev}")
            if not lines:
                lines.append(f'  (raw excerpt: "{posts_raw[:200]}...")')

            block = "\n".join(lines)

            if label == "high" and count_high < top_k:
                high_lines.append(f"[Example {count_high + 1}] Evidence:\n{block}")
                count_high += 1
            elif label == "low" and count_low < top_k:
                low_lines.append(f"[Example {count_low + 1}] Evidence:\n{block}")
                count_low += 1

            if count_high >= top_k and count_low >= top_k:
                break

        return "\n\n".join(high_lines), "\n\n".join(low_lines)

    def build_similar_context(self, posts, trait, top_k=5):
        query_emb = self._embed(posts)
        all_results = self._search(query_emb, top_k * 4)

        blocks = []
        seen = 0
        for r in all_results:
            if trait not in r.get("trait_labels", {}):
                continue
            label = r["trait_labels"][trait]
            posts_raw = r.get("posts_raw", "")
            features = r.get("features", {})

            excerpt = posts_raw[:300].strip()
            if len(posts_raw) > 300:
                excerpt += "..."

            ev_lines = []
            for cat in ["Emotion", "Cognition", "Sensory Perception", "Sociality"]:
                ev = features.get(cat, "").strip()
                if ev:
                    ev_lines.append(f"  {cat}: {ev}")
            ev_block = "\n".join(ev_lines) if ev_lines else "  (no features extracted)"

            block = (
                f"[Similar Text {seen + 1}] (label: {label})\n"
                f"Text excerpt:\n{excerpt}\n"
                f"Psychological evidence:\n{ev_block}"
            )
            blocks.append(block)
            seen += 1
            if seen >= top_k:
                break

        return "\n\n".join(blocks)

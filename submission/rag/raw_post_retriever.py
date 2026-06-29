"""RAG retriever backed by a raw-text-only FAISS index.

No profile fusion, no facet vectors — each vector is simply:
    normalise( embed(raw_post) )

Usage
-----
    retriever = RawPostRetriever(db_dir="data/vector_db/essays_rawpost")
    results   = retriever.retrieve(text, trait="Conscientiousness", top_k=5)
"""

from __future__ import annotations

import os
import numpy as np

from .faiss_index import FAISSIndex

import os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

TRAIT_MAP = {
    "cEXT": "Extraversion",
    "cNEU": "Neuroticism",
    "cAGR": "Agreeableness",
    "cCON": "Conscientiousness",
    "cOPN": "Openness to Experience",
}


class RawPostRetriever:
    """Dense-only retriever using raw-text embeddings (no profile, no facet)."""

    def __init__(self, db_dir: str):
        index_path = os.path.join(db_dir, "vectors.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"No FAISS index found at {db_dir!r}. "
                "Run the build_rawpost_index notebook first."
            )
        self._db_dir = db_dir
        self._index: FAISSIndex | None = None
        print(f"[RawPostRetriever] db_dir={db_dir!r}")

    def _load(self):
        if self._index is not None:
            return
        self._index = FAISSIndex(dimension=0)
        self._index.load(
            os.path.join(self._db_dir, "vectors.faiss"),
            os.path.join(self._db_dir, "vectors_meta.jsonl"),
        )
        print(f"[RawPostRetriever] Index loaded ({self._index._index.ntotal} vectors).")

    def retrieve(self, text: str, trait: str, top_k: int = 5) -> list[dict]:
        """Return up to top_k nearest training entries that have a label for `trait`.

        Parameters
        ----------
        text    : raw post text to embed as the query.
        trait   : full trait name, e.g. "Conscientiousness".
        top_k   : number of results with a matching trait label to return.

        Returns
        -------
        List of dicts with keys: user_id, posts_raw, label, distance.
        """
        from .embedder import get_embedding

        self._load()

        query_vec = np.array(get_embedding(text), dtype="float32")
        fetch_n   = top_k * 6

        _, results = self._index.search(query_vec, fetch_n)

        enriched = []
        for r in results:
            label_map = r.get("trait_labels", {})
            if trait not in label_map:
                continue
            enriched.append({
                "user_id":   r["user_id"],
                "posts_raw": r.get("posts_raw", ""),
                "label":     label_map[trait],
                "distance":  r.get("distance", 0.0),
            })
            if len(enriched) >= top_k:
                break

        return enriched

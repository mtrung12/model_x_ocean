import json
import numpy as np
import faiss


class FAISSIndex:
    """Thin wrapper around a flat L2 FAISS index with JSON-line metadata."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self._index: faiss.Index | None = None
        self._meta: list[dict] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, embeddings: np.ndarray, meta: list[dict]):
        """Build the index from a (N, D) float32 array and a parallel list of meta dicts."""
        embeddings = np.array(embeddings, dtype="float32")
        n, d = embeddings.shape
        self.dimension = d
        self._index = faiss.IndexFlatL2(d)
        self._index.add(embeddings)
        self._meta = meta

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def save(self, index_path: str, meta_path: str):
        if self._index is None:
            raise RuntimeError("Index has not been built yet.")
        faiss.write_index(self._index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            for entry in self._meta:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load(self, index_path: str, meta_path: str):
        self._index = faiss.read_index(index_path)
        self.dimension = self._index.d
        self._meta = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._meta.append(json.loads(line))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Search for the top_k nearest neighbours.

        Parameters
        ----------
        query   : 1-D float32 array of length D
        top_k   : number of results to return

        Returns
        -------
        distances : 1-D float32 array of shape (top_k,)
        results   : list of meta dicts, each augmented with a "distance" key
        """
        if self._index is None:
            raise RuntimeError("Index is not loaded.")

        query = np.array(query, dtype="float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)

        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(query, k)

        distances = distances[0]   # shape (k,)
        indices = indices[0]       # shape (k,)

        results = []
        for dist, idx in zip(distances, indices):
            if idx < 0 or idx >= len(self._meta):
                continue
            entry = dict(self._meta[idx])
            entry["distance"] = float(dist)
            results.append(entry)

        return distances, results

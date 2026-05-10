"""30-d trait-aligned facet vector for personality-aware retrieval.

Encoding per facet (ordered by FACETS):
    high -> +1.0  |  mod -> +0.5  |  none/n/e -> 0.0  |  low -> -1.0

Output is L2-normalised so cosine == dot product.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from rag.profiler.prompts import FACETS

SIGNAL_TO_VALUE: Dict[str, float] = {
    "high": 1.0,
    "mod":  0.5,
    "none": 0.0,
    "n/e":  0.0,
    "low": -1.0,
}

FACET_ORDER: List[str] = [code for code, *_ in FACETS]
FACET_DIM = len(FACET_ORDER)  # 30

TRAIT_FACET_SLICES: Dict[str, List[int]] = {}
for _trait_code in ("cNEU", "cEXT", "cOPN", "cAGR", "cCON"):
    TRAIT_FACET_SLICES[_trait_code] = [
        i for i, (_, _, t, _) in enumerate(FACETS) if t == _trait_code
    ]
assert all(len(v) == 6 for v in TRAIT_FACET_SLICES.values()), (
    "Expected 6 facets per trait in FACETS table"
)


def _encode_signal(signal: Optional[str]) -> float:
    if not signal:
        return 0.0
    return SIGNAL_TO_VALUE.get(signal.strip().lower(), 0.0)


def facet_vector(profile: Dict, normalize: bool = True) -> np.ndarray:
    """Encode a profile dict into a 30-d ordinal facet vector.

    Parameters
    ----------
    profile   : profile entry with a 'facets' key. Missing facets -> 0.0.
    normalize : L2-normalise so cosine == dot product. Default True.

    Returns
    -------
    np.ndarray of shape (30,) dtype float32.
    """
    facets = (profile or {}).get("facets", {}) or {}
    vec = np.zeros(FACET_DIM, dtype=np.float32)
    for i, code in enumerate(FACET_ORDER):
        f = facets.get(code) or {}
        vec[i] = _encode_signal(f.get("signal"))
    if normalize:
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
    return vec


def build_facet_matrix(profile_entries: List[Dict], normalize: bool = True) -> np.ndarray:
    """Stack facet vectors for a list of profile entries (row order matches input order)."""
    if not profile_entries:
        return np.zeros((0, FACET_DIM), dtype=np.float32)
    rows = [facet_vector(p, normalize=normalize) for p in profile_entries]
    return np.vstack(rows).astype(np.float32)


def facet_cosine(
    query_vec: np.ndarray,
    matrix: np.ndarray,
    trait_code: Optional[str] = None,
) -> np.ndarray:
    """Cosine similarity between one query and an (N, 30) matrix.

    If `trait_code` is given, both sides are masked to that trait's 6 facets
    and re-normalised before computing cosine — reduces cross-trait noise.
    """
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    M = np.asarray(matrix, dtype=np.float32)

    if trait_code is not None:
        idx = np.array(TRAIT_FACET_SLICES[trait_code], dtype=np.int64)
        q = q[idx]
        M = M[:, idx]
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0:
            q = q / q_norm
        row_norms = np.linalg.norm(M, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        M = M / row_norms

    return M @ q


def hybrid_score(
    dense_sim: np.ndarray,
    facet_sim: np.ndarray,
    beta: float = 0.5,
    gamma: float = 0.5,
) -> np.ndarray:
    """Linear fusion: score = beta * dense_sim + gamma * facet_sim."""
    dense_sim = np.asarray(dense_sim, dtype=np.float32).reshape(-1)
    facet_sim = np.asarray(facet_sim, dtype=np.float32).reshape(-1)
    if dense_sim.shape != facet_sim.shape:
        raise ValueError(
            f"Shape mismatch: dense_sim={dense_sim.shape} facet_sim={facet_sim.shape}"
        )
    return beta * dense_sim + gamma * facet_sim


def save_facet_matrix(matrix: np.ndarray, path: str) -> None:
    np.save(path, np.asarray(matrix, dtype=np.float32))


def load_facet_matrix(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)

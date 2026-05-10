import numpy as np
from sentence_transformers import SentenceTransformer

import os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

# Update after running finetune_sbert_essays_kaggle.ipynb
FINETUNED_MODEL_DIR = _os.path.join(_PROJECT_ROOT, "models", "sbert_essays_finetuned")
FALLBACK_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBED_PREFIX = "search_document: "
MAX_SEQ_LENGTH = 2048          # NomicBERT n_positions=2048
SLIDING_STRIDE = 1024
ALPHA = 0.5                    # final = ALPHA*embed(raw) + (1-ALPHA)*embed(profile)
BATCH_SIZE = 100

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        import os
        model_path = FINETUNED_MODEL_DIR if os.path.isdir(FINETUNED_MODEL_DIR) else FALLBACK_EMBEDDING_MODEL
        print(f"[embedder] Loading embedding model: {model_path}")
        _model = SentenceTransformer(model_path, trust_remote_code=True)
        _model.max_seq_length = MAX_SEQ_LENGTH
    return _model


def _embed_single(text: str) -> np.ndarray:
    """Embed one text with sliding-window mean pooling if it exceeds MAX_SEQ_LENGTH."""
    model = _get_model()
    tok = model.tokenizer
    prefixed = EMBED_PREFIX + text
    token_ids = tok.encode(prefixed, add_special_tokens=False, truncation=False)

    if len(token_ids) <= MAX_SEQ_LENGTH - 2:
        vec = model.encode(prefixed, show_progress_bar=False, normalize_embeddings=True)
        return np.array(vec, dtype="float32")

    chunk_vecs = []
    for start in range(0, len(token_ids), SLIDING_STRIDE):
        chunk_ids = token_ids[start : start + MAX_SEQ_LENGTH - 2]
        chunk_text = tok.decode(chunk_ids, skip_special_tokens=True)
        vec = model.encode(
            EMBED_PREFIX + chunk_text,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        chunk_vecs.append(vec)
        if start + MAX_SEQ_LENGTH - 2 >= len(token_ids):
            break

    avg = np.mean(chunk_vecs, axis=0).astype("float32")
    norm = np.linalg.norm(avg)
    return avg / norm if norm > 0 else avg


def get_embedding(text: str | list[str]) -> list[float] | list[list[float]]:
    """Embed one string or a list of strings (sliding-window for long inputs)."""
    if isinstance(text, list):
        return [_embed_single(t).tolist() for t in text]
    return _embed_single(text).tolist()


def get_dual_embedding(raw_text: str, profile_text: str, alpha: float = ALPHA) -> list[float]:
    """Fuse raw-essay and profile embeddings: normalise(alpha*embed(raw) + (1-alpha)*embed(profile))."""
    emb_raw     = np.array(_embed_single(raw_text),     dtype="float32")
    emb_profile = np.array(_embed_single(profile_text), dtype="float32")
    fused = alpha * emb_raw + (1.0 - alpha) * emb_profile
    norm = np.linalg.norm(fused)
    fused = fused / norm if norm > 0 else fused
    return fused.tolist()

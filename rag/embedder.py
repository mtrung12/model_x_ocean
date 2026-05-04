import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# PATHS — update these after running finetune_sbert_essays_kaggle.ipynb
# Paths are relative to the project root (the folder containing the rag/ package).
# ---------------------------------------------------------------------------
import os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

# Path to the fine-tuned SBERT model saved by the notebook.
# Set to MODEL_OUTPUT_DIR from the notebook (e.g. "/kaggle/working/sbert_essays_finetuned"
# when running on Kaggle, or a local path after downloading).
# Falls back to the generic model if the fine-tuned path does not exist.
FINETUNED_MODEL_DIR = _os.path.join(_PROJECT_ROOT, "models", "sbert_essays_finetuned")

# Fallback model used when FINETUNED_MODEL_DIR does not exist on disk.
FALLBACK_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"

# Prefix required by nomic-embed / fine-tuned model for document encoding.
# Set to "" if your model does not use a task prefix.
EMBED_PREFIX = "search_document: "

BATCH_SIZE = 100
# ---------------------------------------------------------------------------

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        import os
        model_path = FINETUNED_MODEL_DIR if os.path.isdir(FINETUNED_MODEL_DIR) else FALLBACK_EMBEDDING_MODEL
        print(f"[embedder] Loading embedding model: {model_path}")
        _model = SentenceTransformer(model_path, trust_remote_code=True)
    return _model


def get_embedding(text: str | list[str]) -> list[float] | list[list[float]]:
    model = _get_model()
    if isinstance(text, list):
        prefixed = [EMBED_PREFIX + t for t in text]
        result = model.encode(prefixed, show_progress_bar=False, normalize_embeddings=True)
        return result.tolist()
    result = model.encode(EMBED_PREFIX + text, show_progress_bar=False, normalize_embeddings=True)
    return result.tolist()

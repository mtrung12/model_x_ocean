import os
import time
from typing import Dict, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# PATHS — update these after running finetune_sbert_essays_kaggle.ipynb
# Paths are relative to the project root (the folder containing this package).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory that contains the fine-tuned SBERT model
# (MODEL_OUTPUT_DIR from the notebook).
FINETUNED_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models", "sbert_essays_finetuned")

# Directory that contains the RAG artifacts built by the notebook
# (ARTIFACTS_DIR from the notebook):
#   train_index.faiss, train_metadata.csv, config.json
FINETUNED_ARTIFACTS_DIR = os.path.join(_PROJECT_ROOT, "models", "rag_artifacts")

# Legacy vector-DB directory (used as fallback when the fine-tuned artifacts
# above are not present on disk).
LEGACY_VECTOR_DB_DIR = os.path.join(_PROJECT_ROOT, "data", "vector_db", "essays")
# ---------------------------------------------------------------------------

from principle_model.prompts import (
    SYS_PROMPT,
    SYS_PROMPT_COT,
    PRINCIPLE_PROMPT,
    ONESHOT_USR_PROMPT,
    RAG_PRINCIPLE_PROMPT,
    RAG_ONESHOT_USR_PROMPT,
    COT_PRINCIPLE_PROMPT,
    COT_ONESHOT_USR_PROMPT,
    COT_RAG_PRINCIPLE_PROMPT,
    COT_RAG_ONESHOT_USR_PROMPT,
)
from utils.gpt_client import gpt_call
from utils.hf_client import hf_call
from utils.log import log_to_file
from utils.parser import extract_direct, extract_cot


TRAIT_TO_COLUMN = {
    "Openness": "pred_cOPN",
    "Conscientiousness": "pred_cCON",
    "Extraversion": "pred_cEXT",
    "Agreeableness": "pred_cAGR",
    "Neuroticism": "pred_cNEU",
}

TRAIT_TO_LABEL_COLUMN = {
    "Openness": "cOPN",
    "Conscientiousness": "cCON",
    "Extraversion": "cEXT",
    "Agreeableness": "cAGR",
    "Neuroticism": "cNEU",
}

TRAIT_TO_RAG_NAME = {
    "Openness": "Openness to Experience",
    "Conscientiousness": "Conscientiousness",
    "Extraversion": "Extraversion",
    "Agreeableness": "Agreeableness",
    "Neuroticism": "Neuroticism",
}

# Prompt modes that require a RAG retriever
RAG_PROMPT_MODES = {
    "rag_zeroshot",
    "rag_oneshot",
    "cot_rag_zeroshot",
    "cot_rag_oneshot",
}

# Prompt modes that use chain-of-thought and extract_cot()
COT_PROMPT_MODES = {
    "cot_zeroshot",
    "cot_oneshot",
    "cot_rag_zeroshot",
    "cot_rag_oneshot",
}


def load_principles(principles_dir):
    principles = {}
    for trait_name in TRAIT_TO_COLUMN:
        path = os.path.join(principles_dir, f"principles_{trait_name}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing principles file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            principles[trait_name] = f.read().strip()
    return principles


def _llm_call(model_name, system_prompt, user_prompt, max_new_tokens, temperature):
    if model_name.startswith("gpt"):
        return gpt_call(user_prompt, system_prompt, model_name, max_new_tokens, temperature)
    return hf_call(user_prompt, system_prompt, model_name, max_new_tokens, temperature)


def predict_text(
    text, trait_name, model_name, usr_prompt, sys_prompt,
    max_new_tokens, record_idx, log_filepath, temperature,
):
    formatted_prompt = usr_prompt.replace("<text>", text)
    output = _llm_call(model_name, sys_prompt, formatted_prompt, max_new_tokens, temperature)
    log_to_file(log_filepath, sys_prompt, formatted_prompt, output, f"{record_idx}-{trait_name}")
    return output


def build_prompt(prompt_mode, trait_name, principles, example_df=None, retriever=None, query_text='', top_k=3):
    """Return (usr_prompt, sys_prompt) for the given prompt_mode."""
    rag_trait = TRAIT_TO_RAG_NAME[trait_name]

    # direct zeroshot
    if prompt_mode == "zeroshot":
        return (
            PRINCIPLE_PROMPT.format(trait_name=trait_name, principles=principles),
            SYS_PROMPT,
        )

    # direct oneshot
    if prompt_mode == "oneshot":
        if example_df is None:
            raise ValueError("oneshot requires example_df")
        sample = example_df.sample(1).iloc[0]
        return (
            ONESHOT_USR_PROMPT.format(
                trait_name=trait_name,
                principles=principles,
                example_text=sample["text"],
                example_label=sample[TRAIT_TO_LABEL_COLUMN[trait_name]],
            ),
            SYS_PROMPT,
        )

    # direct RAG modes
    if prompt_mode in ("rag_zeroshot", "rag_oneshot"):
        if retriever is None:
            raise ValueError(f"prompt_mode={prompt_mode!r} requires a retriever")
        similar_context = retriever.build_similar_context(
            posts=query_text, trait=rag_trait, top_k=top_k
        )
        if prompt_mode == "rag_zeroshot":
            return (
                RAG_PRINCIPLE_PROMPT.format(
                    trait_name=trait_name,
                    principles=principles,
                    top_k=top_k,
                    similar_context=similar_context,
                ),
                SYS_PROMPT,
            )
        # rag_oneshot: use only the RAG-retrieved samples as examples
        return (
            RAG_ONESHOT_USR_PROMPT.format(
                trait_name=trait_name,
                principles=principles,
                top_k=top_k,
                similar_context=similar_context,
            ),
            SYS_PROMPT,
        )

    # CoT zeroshot
    if prompt_mode == "cot_zeroshot":
        return (
            COT_PRINCIPLE_PROMPT.format(trait_name=trait_name, principles=principles),
            SYS_PROMPT_COT,
        )

    # CoT oneshot
    if prompt_mode == "cot_oneshot":
        if example_df is None:
            raise ValueError("cot_oneshot requires example_df")
        sample = example_df.sample(1).iloc[0]
        return (
            COT_ONESHOT_USR_PROMPT.format(
                trait_name=trait_name,
                principles=principles,
                example_text=sample["text"],
                example_label=sample[TRAIT_TO_LABEL_COLUMN[trait_name]],
            ),
            SYS_PROMPT_COT,
        )

    # CoT RAG modes
    if prompt_mode in ("cot_rag_zeroshot", "cot_rag_oneshot"):
        if retriever is None:
            raise ValueError(f"prompt_mode={prompt_mode!r} requires a retriever")
        similar_context = retriever.build_similar_context(
            posts=query_text, trait=rag_trait, top_k=top_k
        )
        if prompt_mode == "cot_rag_zeroshot":
            return (
                COT_RAG_PRINCIPLE_PROMPT.format(
                    trait_name=trait_name,
                    principles=principles,
                    top_k=top_k,
                    similar_context=similar_context,
                ),
                SYS_PROMPT_COT,
            )
        # cot_rag_oneshot: use only the RAG-retrieved samples as examples
        return (
            COT_RAG_ONESHOT_USR_PROMPT.format(
                trait_name=trait_name,
                principles=principles,
                top_k=top_k,
                similar_context=similar_context,
            ),
            SYS_PROMPT_COT,
        )

    raise ValueError(f"Unsupported prompt mode: {prompt_mode!r}")


def predict(
    text_df,
    model_name,
    log_dir,
    prompt_mode,
    max_new_tokens,
    res_dir,
    temperature,
    principles_dir,
    example_df=None,
    retriever_df=None,
    top_k=3,
    vector_db_dir=None,  # defaults to FINETUNED_ARTIFACTS_DIR (or LEGACY_VECTOR_DB_DIR)
):
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_filepath = os.path.join(log_dir, model_name, prompt_mode, f"{run_id}_log.txt")
    output_dir = os.path.join(res_dir, model_name, prompt_mode, run_id)
    os.makedirs(output_dir, exist_ok=True)

    principles_by_trait = load_principles(principles_dir)

    # Determine extractor: CoT modes use extract_cot, others use extract_direct
    extractor = extract_cot if prompt_mode in COT_PROMPT_MODES else extract_direct

    retriever = None
    if prompt_mode in RAG_PROMPT_MODES:
        import rag.embedder as _embedder
        from rag.retriever import FeatureRAGRetriever, FINETUNED_ARTIFACTS_DIR as _RAG_DIR
        from rag.runners.build_features import build_features

        # Sync embedder paths with the module-level constants here so a single
        # edit in this file (or in rag/embedder.py) is enough.
        _embedder.FINETUNED_MODEL_DIR = FINETUNED_MODEL_DIR

        # Resolve the effective vector-DB directory:
        #   1. Caller-supplied vector_db_dir (explicit override)
        #   2. Fine-tuned artifacts dir (if it exists on disk)
        #   3. Legacy fallback
        if vector_db_dir is not None:
            effective_db_dir = vector_db_dir
        elif os.path.isdir(FINETUNED_ARTIFACTS_DIR):
            effective_db_dir = FINETUNED_ARTIFACTS_DIR
        else:
            effective_db_dir = LEGACY_VECTOR_DB_DIR

        # For the legacy layout, make sure the index exists (build if needed).
        if effective_db_dir not in (FINETUNED_ARTIFACTS_DIR, _RAG_DIR):
            index_path = os.path.join(effective_db_dir, "vectors.faiss")
            if not os.path.exists(index_path):
                if retriever_df is None:
                    raise ValueError(
                        f"No pre-built index at {effective_db_dir!r} and retriever_df not provided."
                    )
                print(f"[predict] Building index from retriever_df ({len(retriever_df)} rows)...")
                build_features(data=retriever_df, output_dir=effective_db_dir)
                print("[predict] Index build complete.")

        retriever = FeatureRAGRetriever(db_dir=effective_db_dir)
        print(f"[predict] RAG retriever loaded (top_k={top_k}).")

    df = text_df.copy()
    for col in TRAIT_TO_COLUMN.values():
        df[col] = None

    n = len(df)
    print(f"[predict] {n} records | mode={prompt_mode} | model={model_name}")
    t0 = time.time()

    for idx, row in df.iterrows():
        text = row["text"]
        for trait_name, pred_col in TRAIT_TO_COLUMN.items():
            usr_prompt, sys_prompt = build_prompt(
                prompt_mode=prompt_mode,
                trait_name=trait_name,
                principles=principles_by_trait[trait_name],
                example_df=example_df,
                retriever=retriever,
                query_text=text,
                top_k=top_k,
            )
            output = predict_text(
                text=text,
                trait_name=trait_name,
                model_name=model_name,
                usr_prompt=usr_prompt,
                sys_prompt=sys_prompt,
                max_new_tokens=max_new_tokens,
                record_idx=idx,
                log_filepath=log_filepath,
                temperature=temperature,
            )
            df.at[idx, pred_col] = extractor(output.strip())

        if (idx + 1) % 10 == 0:
            print(f"  [predict] {idx + 1}/{n} done.")

    elapsed = time.time() - t0
    prediction_filepath = os.path.join(output_dir, "predictions.csv")
    df.to_csv(prediction_filepath, index=False)
    print(f"[predict] Finished in {elapsed:.2f}s -> {prediction_filepath}")
    return run_id, elapsed, prediction_filepath

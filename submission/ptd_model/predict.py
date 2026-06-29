import os
import time

import pandas as pd

from ptd_model.prompts import (
    SYS_PROMPT_REASONED,
    REASONED_RAG_DEF_ONESHOT_PROMPT,
    TRAITS,
    TRAIT_NOTES,
)
from utils.gpt_client import gpt_call
from utils.hf_client import hf_call
from utils.ollama_client import ollama_call
from utils.log import log_to_file
from utils.parser import extract_reasoned_full


TRAIT_TO_COLUMN = {
    "Openness":          "pred_cOPN",
    "Conscientiousness": "pred_cCON",
    "Extraversion":      "pred_cEXT",
    "Agreeableness":     "pred_cAGR",
    "Neuroticism":       "pred_cNEU",
}


TRAIT_TO_RAG_NAME = {
    "Openness":          "Openness to Experience",
    "Conscientiousness": "Conscientiousness",
    "Extraversion":      "Extraversion",
    "Agreeableness":     "Agreeableness",
    "Neuroticism":       "Neuroticism",
}


PROMPT_MODES = {
    "reasoned_rag_def_oneshot_30f",
}


def _call_profiler_for_text(text, model_name="gpt-4o-mini"):
    from rag.profiler.prompts import build_profiler_prompts, parse_profile_output
    sys_p, usr_p = build_profiler_prompts(text=text, trait_labels=None)
    raw = gpt_call(usr_p, sys_p, model_name, max_new_tokens=1024, temperature=0.0)
    return parse_profile_output(raw)


def _profile_to_full_text(profile):
    from rag.profiler.prompts import FACETS
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


def build_similar_context_full_30(retrieved, top_k):
    from rag.profiler.prompts import slice_profile_full_30, parse_profile_output
    blocks = []
    for i, r in enumerate(retrieved[:top_k]):
        label = r.get("label", "?")

        profile_dict = r.get("profile") or {}
        if not profile_dict.get("facets"):
            profile_dict = (r.get("features") or {}).get("profile") or {}

        if not profile_dict.get("facets"):
            raw_text = r.get("posts_raw", "")
            if raw_text.strip().startswith("[FACETS]"):
                parsed = parse_profile_output(raw_text)
                if parsed.get("valid"):
                    profile_dict = parsed

        if profile_dict.get("facets"):
            evidence = slice_profile_full_30(profile_dict)
        else:
            evidence = r.get("posts_raw", "").strip()

        blocks.append(f"[Similar Profile {i + 1}] (label: {label})\n{evidence}")
    return "\n\n".join(blocks)


def _llm_call(model_name, system_prompt, user_prompt, max_new_tokens, temperature):
    if model_name.startswith("gpt"):
        return gpt_call(user_prompt, system_prompt, model_name, max_new_tokens, temperature)
    if model_name.startswith("ollama:"):
        return ollama_call(user_prompt, system_prompt, model_name[len("ollama:"):], max_new_tokens, temperature)
    return hf_call(user_prompt, system_prompt, model_name, max_new_tokens, temperature)


def predict_text(
    text, trait_name, model_name, usr_prompt, sys_prompt,
    max_new_tokens, record_idx, log_filepath, temperature,
):
    formatted_prompt = usr_prompt.replace("<text>", text)
    output = _llm_call(model_name, sys_prompt, formatted_prompt, max_new_tokens, temperature)
    log_to_file(log_filepath, sys_prompt, formatted_prompt, output, f"{record_idx}-{trait_name}")
    return output


def build_prompt(
    prompt_mode,
    trait_name,
    retriever=None,
    query_text="",
    query_profile_text=None,
    query_profile_dict=None,
    top_k=3,
):
    rag_trait  = TRAIT_TO_RAG_NAME[trait_name]
    trait_defs = TRAITS.get(trait_name, {})

    if prompt_mode == "reasoned_rag_def_oneshot_30f":
        if retriever is None:
            raise ValueError(f"prompt_mode={prompt_mode!r} requires a retriever")
        retrieved = retriever.retrieve(
            posts=query_text,
            trait=rag_trait,
            top_k=top_k,
            profile_text=query_profile_text,
            query_profile_dict=query_profile_dict,
        )
        similar_context = build_similar_context_full_30(retrieved, top_k)
        trait_note = TRAIT_NOTES.get(trait_name, "")
        return (
            REASONED_RAG_DEF_ONESHOT_PROMPT.format(
                trait_name=trait_name,
                definition_high=trait_defs.get("high", ""),
                definition_low=trait_defs.get("low", ""),
                top_k=top_k,
                similar_context=similar_context,
                trait_note=trait_note,
            ),
            SYS_PROMPT_REASONED,
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
    retriever_df=None,
    top_k=3,
    vector_db_dir=None,
    profiler_model="gpt-4o-mini",
):
    run_id       = time.strftime("%Y%m%d-%H%M%S")
    safe_model   = model_name.replace(":", "_")
    log_filepath = os.path.join(log_dir, safe_model, prompt_mode, f"{run_id}_log.txt")
    output_dir   = os.path.join(res_dir, safe_model, prompt_mode, run_id)
    os.makedirs(output_dir, exist_ok=True)

    reasoning_log_path = (
        os.path.join(output_dir, "reasoning_log.jsonl")
        if prompt_mode in PROMPT_MODES
        else None
    )

    retriever = None
    if prompt_mode in PROMPT_MODES:
        from rag.retriever import FeatureRAGRetriever
        retriever = FeatureRAGRetriever(db_dir=vector_db_dir)
        print(f"[predict] RAG retriever ready (top_k={top_k}).")

    df = text_df.copy()
    for col in TRAIT_TO_COLUMN.values():
        df[col] = None

    n  = len(df)
    t0 = time.time()
    print(f"[predict] {n} records | mode={prompt_mode} | model={model_name}")

    for idx, row in df.iterrows():
        text = row["text"]

        query_profile_text = None
        query_profile_dict = None
        if prompt_mode in PROMPT_MODES:
            try:
                profile = _call_profiler_for_text(text, model_name=profiler_model)
                if profile.get("valid"):
                    query_profile_text = _profile_to_full_text(profile)
                    query_profile_dict = profile
                else:
                    print(f"  [predict] Profile invalid for record {idx}, using raw-text fallback.")
            except Exception as exc:
                print(f"  [predict] Profiler error for record {idx}: {exc}. Using raw-text fallback.")

        for trait_name, pred_col in TRAIT_TO_COLUMN.items():
            usr_prompt, sys_prompt = build_prompt(
                prompt_mode=prompt_mode,
                trait_name=trait_name,
                retriever=retriever,
                query_text=text,
                query_profile_text=query_profile_text,
                query_profile_dict=query_profile_dict,
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
            stripped = output.strip()

            import json as _json
            parsed = extract_reasoned_full(stripped)
            df.at[idx, pred_col] = parsed.get("label")
            rec = {
                "record_idx":        int(idx),
                "trait":             trait_name,
                "label":             parsed.get("label"),
                "evidence":          parsed.get("evidence"),
                "facet_check":       parsed.get("facet_check"),
                "example_alignment": parsed.get("example_alignment"),
                "verdict":           parsed.get("verdict"),
            }
            os.makedirs(os.path.dirname(reasoning_log_path), exist_ok=True)
            with open(reasoning_log_path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(rec, ensure_ascii=False) + "\n")

        if (idx + 1) % 10 == 0:
            print(f"  [predict] {idx + 1}/{n} done.")

    elapsed = time.time() - t0
    prediction_filepath = os.path.join(output_dir, "predictions.csv")
    df.to_csv(prediction_filepath, index=False)
    print(f"[predict] Finished in {elapsed:.2f}s -> {prediction_filepath}")
    return run_id, elapsed, prediction_filepath

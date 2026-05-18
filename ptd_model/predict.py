import json
import os
import random
import time

import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FINETUNED_MODEL_DIR     = os.path.join(_PROJECT_ROOT, "models", "sbert_essays_finetuned")
FINETUNED_ARTIFACTS_DIR = os.path.join(_PROJECT_ROOT, "models", "rag_artifacts")
LEGACY_VECTOR_DB_DIR    = os.path.join(_PROJECT_ROOT, "data", "vector_db", "essays")

from ptd_model.prompts import (
    SYS_PROMPT,
    SYS_PROMPT_REASONED,
    SYS_PROMPT_ALL_TRAITS,
    SYS_PROMPT_DEBATE,
    SYS_PROMPT_VERIFIER,
    DEF_ZEROSHOT_PROMPT,
    DEF_ONESHOT_PROMPT,
    REASONED_RAG_DEF_ONESHOT_PROMPT,
    ALL_TRAITS_PROMPT,
    DEBATE_PROBE_PROMPT,
    VERIFIER_PROMPT,
    TRAITS,
    _build_debate_template,
)
from utils.gpt_client import gpt_call
from utils.hf_client import hf_call
from utils.log import log_to_file
from utils.parser import extract_direct, extract_reasoned_full


TRAIT_TO_COLUMN = {
    "Openness":          "pred_cOPN",
    "Conscientiousness": "pred_cCON",
    "Extraversion":      "pred_cEXT",
    "Agreeableness":     "pred_cAGR",
    "Neuroticism":       "pred_cNEU",
}

TRAIT_TO_LABEL_COLUMN = {
    "Openness":          "cOPN",
    "Conscientiousness": "cCON",
    "Extraversion":      "cEXT",
    "Agreeableness":     "cAGR",
    "Neuroticism":       "cNEU",
}

TRAIT_TO_RAG_NAME = {
    "Openness":          "Openness to Experience",
    "Conscientiousness": "Conscientiousness",
    "Extraversion":      "Extraversion",
    "Agreeableness":     "Agreeableness",
    "Neuroticism":       "Neuroticism",
}

RAG_NAME_TO_TRAIT_CODE = {
    "Openness to Experience": "cOPN",
    "Conscientiousness":      "cCON",
    "Extraversion":           "cEXT",
    "Agreeableness":          "cAGR",
    "Neuroticism":            "cNEU",
}

RAG_PROMPT_MODES = {
    "def_zeroshot", "def_oneshot",
    "reasoned_rag_def_oneshot",
}

REASONED_PROMPT_MODES = {
    "reasoned_rag_def_oneshot",
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


def build_similar_context(retrieved, trait_code, top_k):
    from rag.profiler.prompts import slice_profile_for_trait
    blocks = []
    for i, r in enumerate(retrieved[:top_k]):
        label = r.get("label", "?")

        profile_dict = r.get("profile") or {}
        if not profile_dict.get("facets"):
            profile_dict = (r.get("features") or {}).get("profile") or {}

        if profile_dict.get("facets"):
            evidence = slice_profile_for_trait(profile_dict, trait_code)
        else:
            evidence = r.get("posts_raw", "").strip()

        blocks.append(f"[Similar Profile {i + 1}] (label: {label})\n{evidence}")
    return "\n\n".join(blocks)


def _build_label_only_context(shared_results, trait_rag_name, top_k):
    """Build label-only context for def_zeroshot: same retrieved results for all traits."""
    blocks, seen = [], 0
    for r in shared_results:
        label = r.get("trait_labels", {}).get(trait_rag_name)
        if label is None:
            continue
        blocks.append(f"[Similar Profile {seen + 1}] (label: {label})")
        seen += 1
        if seen >= top_k:
            break
    return "\n\n".join(blocks)


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


def build_prompt(
    prompt_mode,
    trait_name,
    retriever=None,
    query_text="",
    query_profile_text=None,
    query_profile_dict=None,
    top_k=3,
    pre_built_context=None,
):
    rag_trait  = TRAIT_TO_RAG_NAME[trait_name]
    trait_code = RAG_NAME_TO_TRAIT_CODE[rag_trait]
    trait_defs = TRAITS.get(trait_name, {})

    def _get_similar_context():
        if retriever is None:
            raise ValueError(f"prompt_mode={prompt_mode!r} requires a retriever")
        retrieved = retriever.retrieve(
            posts=query_text,
            trait=rag_trait,
            top_k=top_k,
            profile_text=query_profile_text,
            query_profile_dict=query_profile_dict,
        )
        return build_similar_context(retrieved, trait_code, top_k)

    if prompt_mode == "def_zeroshot":
        return (
            DEF_ZEROSHOT_PROMPT.format(
                trait_name=trait_name,
                definition_high=trait_defs.get("high", ""),
                definition_low=trait_defs.get("low", ""),
                top_k=top_k,
                similar_context=pre_built_context or "",
            ),
            SYS_PROMPT,
        )

    if prompt_mode == "def_oneshot":
        similar_context = _get_similar_context()
        return (
            DEF_ONESHOT_PROMPT.format(
                trait_name=trait_name,
                definition_high=trait_defs.get("high", ""),
                definition_low=trait_defs.get("low", ""),
                top_k=top_k,
                similar_context=similar_context,
            ),
            SYS_PROMPT,
        )

    if prompt_mode == "reasoned_rag_def_oneshot":
        similar_context = _get_similar_context()
        return (
            REASONED_RAG_DEF_ONESHOT_PROMPT.format(
                trait_name=trait_name,
                definition_high=trait_defs.get("high", ""),
                definition_low=trait_defs.get("low", ""),
                top_k=top_k,
                similar_context=similar_context,
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
    log_filepath = os.path.join(log_dir, model_name, prompt_mode, f"{run_id}_log.txt")
    output_dir   = os.path.join(res_dir, model_name, prompt_mode, run_id)
    os.makedirs(output_dir, exist_ok=True)

    if prompt_mode in REASONED_PROMPT_MODES:
        extractor = lambda r: (extract_reasoned_full(r).get("label"))
    else:
        extractor = extract_direct

    reasoning_log_path = (
        os.path.join(output_dir, "reasoning_log.jsonl")
        if prompt_mode in REASONED_PROMPT_MODES
        else None
    )

    retriever = None
    if prompt_mode in RAG_PROMPT_MODES:
        import rag.embedder as _embedder
        _embedder.FINETUNED_MODEL_DIR = FINETUNED_MODEL_DIR

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
        if prompt_mode in RAG_PROMPT_MODES and prompt_mode != "def_zeroshot":
            try:
                profile = _call_profiler_for_text(text, model_name=profiler_model)
                if profile.get("valid"):
                    query_profile_text = _profile_to_full_text(profile)
                    query_profile_dict = profile
                else:
                    print(f"  [predict] Profile invalid for record {idx}, using raw-text fallback.")
            except Exception as exc:
                print(f"  [predict] Profiler error for record {idx}: {exc}. Using raw-text fallback.")

        # def_zeroshot: retrieve once per essay (trait-agnostic), reuse for all 5 traits
        shared_results = None
        if prompt_mode == "def_zeroshot":
            query_emb = retriever._embed_query(text)
            shared_results = retriever._search(query_emb, top_k * 4)

        for trait_name, pred_col in TRAIT_TO_COLUMN.items():
            rag_trait = TRAIT_TO_RAG_NAME[trait_name]
            pre_context = (
                _build_label_only_context(shared_results, rag_trait, top_k)
                if prompt_mode == "def_zeroshot"
                else None
            )
            usr_prompt, sys_prompt = build_prompt(
                prompt_mode=prompt_mode,
                trait_name=trait_name,
                retriever=retriever,
                query_text=text,
                query_profile_text=query_profile_text,
                query_profile_dict=query_profile_dict,
                top_k=top_k,
                pre_built_context=pre_context,
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

            if reasoning_log_path is not None:
                parsed = extract_reasoned_full(stripped)
                df.at[idx, pred_col] = parsed.get("label")
                import json as _json
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
            else:
                df.at[idx, pred_col] = extractor(stripped)

        if (idx + 1) % 10 == 0:
            print(f"  [predict] {idx + 1}/{n} done.")

    elapsed = time.time() - t0
    prediction_filepath = os.path.join(output_dir, "predictions.csv")
    df.to_csv(prediction_filepath, index=False)
    print(f"[predict] Finished in {elapsed:.2f}s -> {prediction_filepath}")
    return run_id, elapsed, prediction_filepath


# ---------------------------------------------------------------------------
# Multi-LLM debate pipeline
# ---------------------------------------------------------------------------

# Per-step token budgets
_TOKENS_PREDICT = 80    # steps 2/3: exactly 5 XML lines (~40 tokens, 80 is generous)
_TOKENS_DEBATE  = 600   # step 5: 3 answers × up to 5 traits × ~40 tokens each
_TOKENS_VERIFY  = 400   # step 6: <reasoning> (~200 tokens) + <predictions> (~70 tokens)

# Canonical order of XML trait tags used in all-traits prompts
_TRAIT_TAGS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

# Full RAG name → short XML tag
_FULL_TO_TAG = {
    "Openness to Experience": "Openness",
    "Conscientiousness":      "Conscientiousness",
    "Extraversion":           "Extraversion",
    "Agreeableness":          "Agreeableness",
    "Neuroticism":            "Neuroticism",
}

# Short XML tag → prediction column
_TAG_TO_COL = {
    "Openness":          "pred_cOPN",
    "Conscientiousness": "pred_cCON",
    "Extraversion":      "pred_cEXT",
    "Agreeableness":     "pred_cAGR",
    "Neuroticism":       "pred_cNEU",
}


def _retrieve_for_debate(retriever, text, profile_text, top_k_total):
    """Dense retrieval without per-trait filtering; returns up to top_k_total results."""
    query_emb = retriever._embed_query(text, profile_text)
    results = retriever._search(query_emb, top_k_total * 4)
    seen, out = set(), []
    for r in results:
        uid = r.get("user_id", "")
        if uid in seen:
            continue
        if r.get("trait_labels"):
            out.append(r)
            seen.add(uid)
        if len(out) >= top_k_total:
            break
    return out


def _build_all_traits_context(retrieved, top_k):
    """Context block showing all 5 trait labels (and brief evidence) per example."""
    blocks = []
    for i, r in enumerate(retrieved[:top_k]):
        raw_labels = r.get("trait_labels", {})
        parts = []
        for full, tag in _FULL_TO_TAG.items():
            lbl = raw_labels.get(full) or raw_labels.get(tag)
            if lbl:
                parts.append(f"{tag}={lbl}")
        labels_str = ", ".join(parts) or "?"

        profile_dict = r.get("profile") or {}
        if profile_dict.get("facets"):
            from rag.profiler.prompts import FACETS
            ev_lines = []
            for code, _, _, _ in FACETS:
                f = profile_dict["facets"].get(code, {})
                if f.get("signal") and f["signal"] not in ("n/e", "none"):
                    ev_lines.append(f"{code}|{f['signal']}|{f.get('evidence', '')}")
            evidence = "\n".join(ev_lines[:10])
        else:
            evidence = r.get("posts_raw", "").strip()[:300]

        blocks.append(f"[Similar Profile {i + 1}] (labels: {labels_str})\n{evidence}")
    return "\n\n".join(blocks)


def _fmt_preds(preds):
    return "\n".join(f"  {t}: {preds.get(t) or '?'}" for t in _TRAIT_TAGS)


def _compare_preds(preds_a, preds_b):
    agree, disagree = [], []
    for t in _TRAIT_TAGS:
        a, b = preds_a.get(t), preds_b.get(t)
        if a and b and a == b:
            agree.append(t)
        else:
            disagree.append(t)
    return agree, disagree


def _make_all_traits_prompt(text, context, top_k):
    """Render ALL_TRAITS_PROMPT: format known vars, then inject raw text."""
    partial = ALL_TRAITS_PROMPT.format(
        openness_high=TRAITS["Openness"]["high"],
        openness_low=TRAITS["Openness"]["low"],
        conscientiousness_high=TRAITS["Conscientiousness"]["high"],
        conscientiousness_low=TRAITS["Conscientiousness"]["low"],
        extraversion_high=TRAITS["Extraversion"]["high"],
        extraversion_low=TRAITS["Extraversion"]["low"],
        agreeableness_high=TRAITS["Agreeableness"]["high"],
        agreeableness_low=TRAITS["Agreeableness"]["low"],
        neuroticism_high=TRAITS["Neuroticism"]["high"],
        neuroticism_low=TRAITS["Neuroticism"]["low"],
        top_k=top_k,
        similar_context=context,
    )
    return partial.replace("<text>", text.strip())


def _make_debate_probe_prompt(text, my_preds, their_preds, disagreed):
    disagree_lines = "\n".join(
        f"  {t}: you said {my_preds.get(t) or '?'}, they said {their_preds.get(t) or '?'}"
        for t in disagreed
    )
    return DEBATE_PROBE_PROMPT.format(
        your_predictions=_fmt_preds(my_preds),
        disagreements=disagree_lines,
        debate_template=_build_debate_template(disagreed),
    ).replace("<text>", text.strip())


def _make_verifier_prompt(text, model_a_name, model_b_name,
                          preds_a, preds_b, agree, disagree,
                          debate_a, debate_b):
    # Escape curly braces in free-text LLM outputs before calling .format()
    def _safe(s):
        return (s or "").replace("{", "{{").replace("}", "}}")

    rendered = VERIFIER_PROMPT.format(
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        model_a_predictions=_fmt_preds(preds_a),
        model_b_predictions=_fmt_preds(preds_b),
        agreements=", ".join(agree) if agree else "(none)",
        disagreements=", ".join(disagree) if disagree else "(none — all agree)",
        model_a_debate=_safe(debate_a) or "(no debate — all traits agreed)",
        model_b_debate=_safe(debate_b) or "(no debate — all traits agreed)",
    )
    return rendered.replace("<text>", text.strip())


def predict_debate(
    text_df,
    model_a_name,
    model_b_name,
    verifier_model_name,
    log_dir,
    res_dir,
    temperature=0.0,
    top_k=3,
    vector_db_dir=None,
    profiler_model="gpt-4o-mini",
):
    """Multi-LLM debate pipeline.

    Flow per record
    ---------------
    1. Retrieve k*2 examples; split randomly into k for Model A, k for Model B.
    2. Model A predicts all 5 traits with its context.       (~80 tokens)
    3. Model B predicts all 5 traits with its context.       (~80 tokens)
    4. Compare; identify agreements and disagreements.
    5. For every disagreement probe each model:              (~600 tokens)
       assumption / evidence / falsifier per trait.
    6. Verifier reads both predictions + debate, emits       (~400 tokens)
       <reasoning> + <predictions> final answer.
    """
    from utils.parser import extract_all_traits, extract_verifier_output

    run_id = time.strftime("%Y%m%d-%H%M%S")
    label_a = model_a_name.replace("/", "-")
    label_b = model_b_name.replace("/", "-")
    output_dir = os.path.join(res_dir, "debate", f"{label_a}_vs_{label_b}", run_id)
    os.makedirs(output_dir, exist_ok=True)

    log_base = os.path.join(log_dir, "debate", run_id)
    os.makedirs(log_base, exist_ok=True)
    log_a      = os.path.join(log_base, "model_a.txt")
    log_b      = os.path.join(log_base, "model_b.txt")
    log_verify = os.path.join(log_base, "verifier.txt")
    debate_log = os.path.join(output_dir, "debate_log.jsonl")

    import rag.embedder as _embedder
    _embedder.FINETUNED_MODEL_DIR = FINETUNED_MODEL_DIR
    from rag.retriever import FeatureRAGRetriever
    retriever = FeatureRAGRetriever(db_dir=vector_db_dir)
    print(f"[debate] RAG retriever ready (k_per_model={top_k}).")

    df = text_df.copy()
    for col in _TAG_TO_COL.values():
        df[col] = None

    n, t0 = len(df), time.time()
    print(f"[debate] {n} records | A={model_a_name} | B={model_b_name} | V={verifier_model_name}")

    for idx, row in df.iterrows():
        text = row["text"]

        # Profile the essay for better embeddings
        profile_text = None
        try:
            profile = _call_profiler_for_text(text, model_name=profiler_model)
            if profile.get("valid"):
                profile_text = _profile_to_full_text(profile)
        except Exception as exc:
            print(f"  [debate] profiler error record {idx}: {exc}")

        # Step 1 — retrieve k*2, split randomly
        pool = _retrieve_for_debate(retriever, text, profile_text, top_k_total=top_k * 2)
        random.shuffle(pool)
        retrieved_a = pool[:top_k]
        retrieved_b = pool[top_k:top_k * 2] or pool[:top_k]  # fallback if pool < 2k

        ctx_a = _build_all_traits_context(retrieved_a, top_k)
        ctx_b = _build_all_traits_context(retrieved_b, top_k)

        # Step 2 — Model A predicts all traits
        usr_a   = _make_all_traits_prompt(text, ctx_a, top_k)
        raw_a   = _llm_call(model_a_name, SYS_PROMPT_ALL_TRAITS, usr_a, _TOKENS_PREDICT, temperature)
        log_to_file(log_a, SYS_PROMPT_ALL_TRAITS, usr_a, raw_a, f"{idx}-step2-model_a")
        preds_a = extract_all_traits(raw_a)

        # Step 3 — Model B predicts all traits independently
        usr_b   = _make_all_traits_prompt(text, ctx_b, top_k)
        raw_b   = _llm_call(model_b_name, SYS_PROMPT_ALL_TRAITS, usr_b, _TOKENS_PREDICT, temperature)
        log_to_file(log_b, SYS_PROMPT_ALL_TRAITS, usr_b, raw_b, f"{idx}-step3-model_b")
        preds_b = extract_all_traits(raw_b)

        # Step 4 — Compare
        agree, disagree = _compare_preds(preds_a, preds_b)

        # Step 5 — Probe each model on its disagreements
        debate_a_raw = debate_b_raw = ""
        if disagree:
            probe_a      = _make_debate_probe_prompt(text, preds_a, preds_b, disagree)
            debate_a_raw = _llm_call(model_a_name, SYS_PROMPT_DEBATE, probe_a, _TOKENS_DEBATE, temperature)
            log_to_file(log_a, SYS_PROMPT_DEBATE, probe_a, debate_a_raw, f"{idx}-step5-model_a")

            probe_b      = _make_debate_probe_prompt(text, preds_b, preds_a, disagree)
            debate_b_raw = _llm_call(model_b_name, SYS_PROMPT_DEBATE, probe_b, _TOKENS_DEBATE, temperature)
            log_to_file(log_b, SYS_PROMPT_DEBATE, probe_b, debate_b_raw, f"{idx}-step5-model_b")

        # Step 6 — Verifier produces final answer
        usr_v  = _make_verifier_prompt(
            text, model_a_name, model_b_name,
            preds_a, preds_b, agree, disagree,
            debate_a_raw, debate_b_raw,
        )
        raw_v  = _llm_call(verifier_model_name, SYS_PROMPT_VERIFIER, usr_v, _TOKENS_VERIFY, temperature)
        log_to_file(log_verify, SYS_PROMPT_VERIFIER, usr_v, raw_v, f"{idx}-step6-verifier")
        verified = extract_verifier_output(raw_v)
        final    = verified["traits"]

        for tag, col in _TAG_TO_COL.items():
            df.at[idx, col] = final.get(tag)

        with open(debate_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "record_idx":  int(idx),
                "preds_a":     preds_a,
                "preds_b":     preds_b,
                "agree":       agree,
                "disagree":    disagree,
                "final":       final,
            }, ensure_ascii=False) + "\n")

        if (idx + 1) % 10 == 0:
            print(f"  [debate] {idx + 1}/{n} done.")

    elapsed = time.time() - t0
    pred_path = os.path.join(output_dir, "predictions.csv")
    df.to_csv(pred_path, index=False)
    print(f"[debate] Finished in {elapsed:.2f}s -> {pred_path}")
    return run_id, elapsed, pred_path

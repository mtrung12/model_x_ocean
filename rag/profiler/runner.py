"""Driver that profiles every essay in a training set and writes a ProfileStore.

Usage (from a notebook or script):

    import pandas as pd
    from rag.profiler.runner import build_profiles

    train = pd.read_csv("data/split/essays/train.csv")
    build_profiles(
        data=train,
        output_dir="data/profile_db/essays",
        model_name="gpt-4o",          # use a STRONG model here
        log_dir="log/profiler",
    )

The runner is append-safe: re-running it after a partial run will skip
user_ids that are already profiled and valid.
"""

from __future__ import annotations

import gc
import os
import time
from typing import Dict, Optional

import pandas as pd

from utils.gpt_client import gpt_call
from utils.log import log_to_file

from .prompts import (
    PROFILER_MAX_TOKENS,
    build_profiler_prompts,
    parse_profile_output,
)
from .store import ProfileStore


TRAIT_COLS = ("cOPN", "cCON", "cEXT", "cAGR", "cNEU")


def _extract_trait_labels(row: pd.Series) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for col in TRAIT_COLS:
        if col in row.index:
            raw = str(row[col]).strip().lower()
            if raw in ("high", "low"):
                labels[col] = raw
            elif raw in ("1", "1.0"):
                labels[col] = "high"
            elif raw in ("0", "0.0"):
                labels[col] = "low"
    return labels


def _call_profiler(
    text: str,
    trait_labels: Dict[str, str],
    model_name: str,
    log_filepath: Optional[str] = None,
    record_idx: Optional[int] = None,
    use_labels: bool = True,
) -> Dict:
    """Single profiler call. Retries once on parse failure with a tighter
    instruction. Returns a parsed-profile dict (see parse_profile_output).
    """
    sys_p, usr_p = build_profiler_prompts(
        text=text,
        trait_labels=trait_labels if use_labels else None,
    )
    raw = gpt_call(usr_p, sys_p, model_name, PROFILER_MAX_TOKENS, 0.0)
    profile = parse_profile_output(raw)

    if not profile["valid"]:
        # Retry once with an even more emphatic format reminder appended
        usr_retry = (
            usr_p
            + "\n\nFORMAT REMINDER: emit exactly 30 facet lines, each with a"
              " signal in {high, mod, low, none, n/e}. Do NOT skip lines or"
              " collapse cells. Output ONLY the [FACETS] and [LINGUISTIC]"
              " blocks."
        )
        raw_retry = gpt_call(usr_retry, sys_p, model_name, PROFILER_MAX_TOKENS, 0.0)
        profile_retry = parse_profile_output(raw_retry)
        if profile_retry["valid"] or _count_valid(profile_retry) > _count_valid(profile):
            profile = profile_retry
            raw = raw_retry

    if log_filepath is not None and record_idx is not None:
        log_to_file(log_filepath, sys_p, usr_p, raw, record_idx)

    return profile


def _count_valid(profile: Dict) -> int:
    from .prompts import SIGNAL_VOCAB
    return sum(
        1 for f in profile.get("facets", {}).values()
        if f.get("signal") in SIGNAL_VOCAB
    )


def build_profiles(
    data: pd.DataFrame,
    output_dir: str = "data/profile_db/essays",
    model_name: str = "gpt-4o",
    log_dir: Optional[str] = None,
    use_labels: bool = True,
    checkpoint_every: int = 25,
    progress_every: int = 10,
) -> ProfileStore:
    """Generate profiles for every row in ``data`` and persist them.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain a ``text`` column. Trait label columns (cOPN/cCON/
        cEXT/cAGR/cNEU) are optional but recommended (used only as loose
        anchors in the prompt; the prompt forbids label-justified
        reasoning).
    output_dir : str
        Folder for the JSONL profile store.
    model_name : str
        OpenAI model. Use a strong model -- profiles are generated once
        and bound the quality of downstream retrieval forever.
    log_dir : str
        Where to write per-record prompt/response logs.
    use_labels : bool
        Whether to expose the trait labels to the profiler. Default True.
        Set False to generate label-blind profiles (useful for evaluation
        of retrieval quality without leakage concerns).
    checkpoint_every : int
        Save the store to disk every N records.
    progress_every : int
        Print progress every N records.
    """
    if "text" not in data.columns:
        raise ValueError("`data` must contain a 'text' column")

    os.makedirs(output_dir, exist_ok=True)
    if log_dir is None:
        _root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        log_dir = os.path.join(_root, "log", "profiler")
    os.makedirs(log_dir, exist_ok=True)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_filepath = os.path.join(log_dir, f"{run_id}_log.txt")

    store_path = os.path.join(output_dir, "profile_store.jsonl")
    store = ProfileStore(store_path)
    store.load()

    n = len(data)
    print(f"[{run_id}] Profiling {n} essays -> {store_path}")
    print(f"[{run_id}] Model: {model_name}  | use_labels={use_labels}")
    print(f"[{run_id}] Logs : {log_filepath}")
    if len(store) > 0:
        print(f"[{run_id}] Resuming: {len(store)} essays already profiled.")

    t0 = time.time()
    errors = 0
    invalids = 0
    new_done = 0

    for i, row in data.iterrows():
        user_id = f"user_{i}"
        if store.has(user_id) and store.get(user_id).get("valid"):
            continue

        text = str(row["text"])
        trait_labels = _extract_trait_labels(row)

        try:
            profile = _call_profiler(
                text=text,
                trait_labels=trait_labels,
                model_name=model_name,
                log_filepath=log_filepath,
                record_idx=i,
                use_labels=use_labels,
            )
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"  [profiler] error on user_{i}: {exc}")
            profile = {
                "facets": {},
                "linguistic": {},
                "raw": "",
                "valid": False,
            }

        if not profile.get("valid"):
            invalids += 1

        store.add(user_id, trait_labels, profile, model_name)
        new_done += 1

        if new_done % progress_every == 0:
            elapsed = time.time() - t0
            rate = new_done / elapsed if elapsed > 0 else 1.0
            remaining_records = n - (i + 1)
            eta = remaining_records / rate if rate > 0 else 0.0
            print(
                f"  [profiler] {i + 1}/{n} done. "
                f"errors={errors} invalid={invalids} "
                f"rate={rate:.2f}/s ETA={eta:.0f}s"
            )

        if new_done % checkpoint_every == 0:
            store.save()

        gc.collect()

    store.save()
    total = time.time() - t0
    print(
        f"[profiler] Done. {new_done} new profiles in {total:.1f}s. "
        f"errors={errors} invalid={invalids}"
    )
    return store

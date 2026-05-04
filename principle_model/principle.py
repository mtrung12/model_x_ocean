import os
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from principle_model.prompts import (
    DISCOVERY_SYS_PROMPT,
    TURN1_LOW_PROMPT,
    TURN2_HIGH_PROMPT,
    TURN3_SYNTHESIS_PROMPT,
)
from utils.gpt_client import gpt_call
from utils.hf_client import hf_call
from utils.log import log_to_file


TRAIT_TO_COLUMN = {
    "Openness": "cOPN",
    "Conscientiousness": "cCON",
    "Extraversion": "cEXT",
    "Agreeableness": "cAGR",
    "Neuroticism": "cNEU",
}


def _llm_call(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    if model_name.startswith("gpt"):
        return gpt_call(
            user_prompt, system_prompt, model_name, max_new_tokens, temperature
        )
    return hf_call(
        user_prompt, system_prompt, model_name, max_new_tokens, temperature
    )


def sample_trait_examples(
    df: pd.DataFrame,
    trait_name: str,
    n_high: int,
    n_low: int,
    random_state: int = 42,
) -> Tuple[List[str], List[str]]:
    if trait_name not in TRAIT_TO_COLUMN:
        raise ValueError(f"Unknown trait: {trait_name}")

    column = TRAIT_TO_COLUMN[trait_name]
    labels = df[column].astype(str).str.lower()

    high_pool = df[labels == "high"]
    low_pool = df[labels == "low"]

    if len(high_pool) < n_high or len(low_pool) < n_low:
        raise ValueError(
            f"Not enough records for {trait_name}: "
            f"available high={len(high_pool)}, low={len(low_pool)}; "
            f"requested high={n_high}, low={n_low}"
        )

    high_examples = (
        high_pool.sample(n_high, random_state=random_state)["text"].tolist()
    )
    low_examples = (
        low_pool.sample(n_low, random_state=random_state)["text"].tolist()
    )

    return low_examples, high_examples


def format_examples(examples: List[str]) -> str:
    return "\n\n".join(f"{i + 1}. {text}" for i, text in enumerate(examples))


def discover_trait_principles(
    train_df: pd.DataFrame,
    trait_name: str,
    model_name: str,
    n_high: int = 5,
    n_low: int = 5,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    random_state: int = 42,
    log_filepath: Optional[str] = None,
) -> str:
    low_examples, high_examples = sample_trait_examples(
        train_df,
        trait_name=trait_name,
        n_high=n_high,
        n_low=n_low,
        random_state=random_state,
    )

    turn1_prompt = TURN1_LOW_PROMPT.format(
        trait_name=trait_name, examples=format_examples(low_examples)
    )
    low_patterns = _llm_call(
        model_name, DISCOVERY_SYS_PROMPT, turn1_prompt, max_new_tokens, temperature
    )

    turn2_prompt = TURN2_HIGH_PROMPT.format(
        trait_name=trait_name, examples=format_examples(high_examples)
    )
    high_patterns = _llm_call(
        model_name, DISCOVERY_SYS_PROMPT, turn2_prompt, max_new_tokens, temperature
    )

    turn3_prompt = TURN3_SYNTHESIS_PROMPT.format(
        trait_name=trait_name,
        low_patterns=low_patterns,
        high_patterns=high_patterns,
    )
    principles = _llm_call(
        model_name, DISCOVERY_SYS_PROMPT, turn3_prompt, max_new_tokens, temperature
    )

    if log_filepath:
        log_to_file(
            log_filepath, DISCOVERY_SYS_PROMPT, turn1_prompt, low_patterns,
            f"{trait_name}-turn1-low",
        )
        log_to_file(
            log_filepath, DISCOVERY_SYS_PROMPT, turn2_prompt, high_patterns,
            f"{trait_name}-turn2-high",
        )
        log_to_file(
            log_filepath, DISCOVERY_SYS_PROMPT, turn3_prompt, principles,
            f"{trait_name}-turn3-synthesis",
        )

    return principles


def discover_principles(
    train_df: pd.DataFrame,
    model_name: str,
    output_dir: str,
    n_high: int = 5,
    n_low: int = 5,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    random_state: int = 42,
    traits: Optional[Iterable[str]] = None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    log_filepath = os.path.join(output_dir, "discovery_log.txt")

    target_traits = list(traits) if traits else list(TRAIT_TO_COLUMN.keys())
    principles = {}

    for trait_name in target_traits:
        print(f"[{trait_name}] discovering principles...")

        trait_principles = discover_trait_principles(
            train_df=train_df,
            trait_name=trait_name,
            model_name=model_name,
            n_high=n_high,
            n_low=n_low,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            random_state=random_state,
            log_filepath=log_filepath,
        )

        principles[trait_name] = trait_principles

        out_path = os.path.join(output_dir, f"principles_{trait_name}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(trait_principles)

        print(f"[{trait_name}] saved -> {out_path}")

    return principles

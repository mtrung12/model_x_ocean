from dotenv import load_dotenv
import os
import math
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

load_dotenv()

client = None


def get_client():
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set."
            )
        client = OpenAI(api_key=api_key)
    return client

def create_message(sys_prompt: str, usr_prompt: str):
    return [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': usr_prompt}
    ]

def gpt_call(
    user_prompt: str,
    system_prompt: str,
    model: str,
    max_new_tokens: int,
    temperature: float,
):
    message = create_message(system_prompt, user_prompt)
    response = get_client().responses.create(
        model=model,
        temperature=temperature,
        max_output_tokens=max_new_tokens,
        input=message
    )

    return response.output_text


def gpt_call_with_logprobs(
    user_prompt: str,
    system_prompt: str,
    model: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_logprobs: int = 5,
) -> Tuple[str, List[Dict]]:
    """
    Chat-completions call that asks the API to return per-token log
    probabilities for the next-token alternatives.

    Returns:
        text       : the full assistant completion (str)
        token_info : list of dicts, one per generated token, each with
                     keys "token", "logprob", and "top": a dict of
                     {token_str: logprob} for the top-k alternatives.
    """
    message = create_message(system_prompt, user_prompt)
    response = get_client().chat.completions.create(
        model=model,
        messages=message,
        temperature=temperature,
        max_tokens=max_new_tokens,
        logprobs=True,
        top_logprobs=top_logprobs,
    )

    choice = response.choices[0]
    text = choice.message.content or ""

    token_info: List[Dict] = []
    if choice.logprobs and choice.logprobs.content:
        for tok in choice.logprobs.content:
            top_map = {
                t.token: t.logprob for t in (tok.top_logprobs or [])
            }
            token_info.append({
                "token": tok.token,
                "logprob": tok.logprob,
                "top": top_map,
            })

    return text, token_info


def extract_high_low_probability(
    token_info: List[Dict],
    high_token: str = "high",
    low_token: str = "low",
    fallback_logprob: float = -20.0,
) -> Optional[float]:
    """
    Walk the generated tokens from the END backwards and find the
    first token position where either 'high' or 'low' (case-
    insensitive, surrounding whitespace ignored) is the chosen token
    or appears in top_logprobs. From that position read the logprob
    of "high" and "low" (whichever is missing gets fallback_logprob),
    convert to a binary normalized probability:

        p_high = exp(lp_high) / (exp(lp_high) + exp(lp_low))

    Returns p_high in [0, 1], or None if neither token appears at
    any position.
    """
    if not token_info:
        return None

    target_high = high_token.strip().lower()
    target_low = low_token.strip().lower()

    def _norm(s: str) -> str:
        return s.strip().lower()

    for pos in range(len(token_info) - 1, -1, -1):
        tok = token_info[pos]
        chosen = _norm(tok["token"])
        top_norm = {_norm(k): v for k, v in tok["top"].items()}

        if chosen in (target_high, target_low) \
                or target_high in top_norm \
                or target_low in top_norm:
            lp_h = top_norm.get(target_high, fallback_logprob)
            lp_l = top_norm.get(target_low, fallback_logprob)

            if chosen == target_high:
                lp_h = max(lp_h, tok["logprob"])
            if chosen == target_low:
                lp_l = max(lp_l, tok["logprob"])

            eh = math.exp(lp_h)
            el = math.exp(lp_l)
            denom = eh + el
            if denom <= 0:
                return None
            return eh / denom

    return None


def gpt_embed(
    texts: List[str],
    model: str = "text-embedding-3-large",
    batch_size: int = 64,
) -> List[List[float]]:
    """Embed a list of texts. Returns a list of float vectors."""
    out: List[List[float]] = []
    cli = get_client()
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        clean = [t if t and t.strip() else " " for t in batch]
        resp = cli.embeddings.create(model=model, input=clean)
        out.extend([d.embedding for d in resp.data])
    return out

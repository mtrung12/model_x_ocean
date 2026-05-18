import re
from typing import Dict, Optional


def extract_direct(response: str) -> str | None:
    match = re.search(r"\b(high|low)\b", response, re.IGNORECASE)
    return match.group(1).lower() if match else None


def extract_cot(response: str) -> str | None:
    match = re.search(r"Label\s*:\s*(high|low)", response, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    for line in reversed(response.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        tokens = re.findall(r"\b(high|low)\b", line, re.IGNORECASE)
        if len(tokens) == 1:
            return tokens[0].lower()

    return None


_REASONED_SECTIONS = (
    "evidence",
    "facet_check",
    "example_alignment",
    "verdict",
    "label",
)


def _extract_tag(response: str, tag: str) -> Optional[str]:
    """Return the contents of the first <tag>...</tag> block, or None."""
    pattern = rf"<\s*{tag}\s*>(.*?)<\s*/\s*{tag}\s*>"
    m = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


def extract_reasoned_full(response: str) -> Dict[str, Optional[str]]:
    """Parse a reasoned (XML-tagged) response into its component sections.

    Returns a dict with keys: evidence, facet_check, example_alignment,
    verdict, label, raw. `label` is normalised to "high"/"low"/None.
    Falls back to extract_cot if <label> is missing or malformed.
    """
    out: Dict[str, Optional[str]] = {key: None for key in _REASONED_SECTIONS}
    out["raw"] = response

    if not response:
        return out

    for tag in _REASONED_SECTIONS:
        out[tag] = _extract_tag(response, tag)

    label_text = out.get("label") or ""
    label_match = re.search(r"\b(high|low)\b", label_text, re.IGNORECASE)
    if label_match:
        out["label"] = label_match.group(1).lower()
    else:
        out["label"] = extract_cot(response)

    return out


def extract_reasoned(response: str) -> Optional[str]:
    """Drop-in label extractor for the reasoned prompt mode. Returns 'high', 'low', or None."""
    return extract_reasoned_full(response).get("label")


# ---------------------------------------------------------------------------
# All-traits (multi-LLM debate) parsers
# ---------------------------------------------------------------------------

TRAIT_TAGS = ("Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism")

# Text that looks like a label but is actually a template placeholder
_TEMPLATE_NOISE = re.compile(r"\bhigh\b.{0,10}\blow\b|\blow\b.{0,10}\bhigh\b", re.IGNORECASE)


def _extract_single_trait(text: str, tag: str) -> Optional[str]:
    """Extract one high/low label for *tag* from *text* using three strategies.

    Strategy 1 — XML tag  <Openness>high</Openness>
    Strategy 2 — key-value pattern  "Openness: high" / "Openness = high"
    Strategy 3 — first line that contains the trait name and a label word
    """
    # ── Strategy 1: XML tag ─────────────────────────────────────────────────
    xml_pat = rf"<\s*{re.escape(tag)}\s*>(.*?)<\s*/\s*{re.escape(tag)}\s*>"
    m = re.search(xml_pat, text, re.IGNORECASE | re.DOTALL)
    if m:
        inner = m.group(1).strip()
        # Reject template placeholders like "high or low"
        if not _TEMPLATE_NOISE.search(inner):
            lm = re.search(r"\b(high|low)\b", inner, re.IGNORECASE)
            if lm:
                return lm.group(1).lower()

    # ── Strategy 2: key-value  "Openness: high" ─────────────────────────────
    # Allow up to 20 non-alpha characters between the trait name and the label
    kv_pat = rf"\b{re.escape(tag)}\b[^a-zA-Z\n]{{0,20}}(high|low)\b"
    m = re.search(kv_pat, text, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # ── Strategy 3: scan line by line ───────────────────────────────────────
    for line in text.splitlines():
        if re.search(rf"\b{re.escape(tag)}\b", line, re.IGNORECASE):
            if _TEMPLATE_NOISE.search(line):
                continue  # skip "Openness: high or low" template lines
            lm = re.search(r"\b(high|low)\b", line, re.IGNORECASE)
            if lm:
                return lm.group(1).lower()

    return None


def extract_all_traits(response: str) -> Dict[str, Optional[str]]:
    """Parse all-traits output. Returns {trait_tag -> 'high'/'low'/None}.

    Tries XML-tag extraction first, then falls back to key-value and
    line-level patterns so that mildly malformed responses still parse.
    """
    return {tag: _extract_single_trait(response, tag) for tag in TRAIT_TAGS}


def extract_verifier_output(response: str) -> Dict:
    """Parse the verifier's response.

    Extracts traits ONLY from the <predictions> block so that trait XML tags
    quoted inside <reasoning> do not cause false matches.
    Falls back to searching the response with the <reasoning> block stripped.
    """
    reasoning = _extract_tag(response, "reasoning")

    # Primary: look inside <predictions> block only
    predictions_block = _extract_tag(response, "predictions")
    if predictions_block:
        traits = extract_all_traits(predictions_block)
    else:
        # Fallback: strip reasoning block so quoted tags don't pollute results
        clean = re.sub(
            r"<\s*reasoning\s*>.*?<\s*/\s*reasoning\s*>", "",
            response, flags=re.DOTALL | re.IGNORECASE,
        )
        traits = extract_all_traits(clean)

    return {"traits": traits, "reasoning": reasoning}

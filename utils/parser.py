import re
from typing import Dict, Optional, Tuple


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

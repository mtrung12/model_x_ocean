import re


def extract_direct(response: str) -> str | None:
    match = re.search(r"\b(high|low)\b", response, re.IGNORECASE)
    return match.group(1).lower() if match else None


def extract_cot(response: str) -> str | None:
    # Extract label from a chain-of-thought response.
    #
    # Primary strategy: look for an explicit 'Label: high' / 'Label: low'
    # delimiter that the CoT prompts instruct the model to emit.
    #
    # Fallback strategy: scan non-empty lines from the bottom of the response
    # and return the first line that contains exactly one 'high' or 'low'
    # token. This handles cases where the model omits the delimiter but still
    # ends with the answer word on its own line.

    # Primary: explicit delimiter
    match = re.search(r"Label\s*:\s*(high|low)", response, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Fallback: last non-empty line containing exactly one label word
    for line in reversed(response.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        tokens = re.findall(r"\b(high|low)\b", line, re.IGNORECASE)
        if len(tokens) == 1:
            return tokens[0].lower()

    return None

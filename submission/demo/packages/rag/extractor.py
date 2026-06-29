EXTRACTOR_MAX_TOKENS = 512

CATEGORIES = ["Emotion", "Cognition", "Sensory Perception", "Sociality"]

SYS_PROMPT_EXTRACTOR = (
    "You are a psycholinguistics analyst.\n\n"
    "Your task is to extract short, direct evidence quotes from a piece of text\n"
    "across four psychological categories.\n\n"
    "Categories:\n"
    "- Emotion: words, phrases, or sentences expressing feelings, moods, or\n"
    "  affective states.\n"
    "- Cognition: words or phrases reflecting thoughts, beliefs, doubts,\n"
    "  decisions, or mental processes.\n"
    "- Sensory Perception: references to physical sensations, sensory\n"
    "  experiences, or environment perception.\n"
    "- Sociality: references to people, relationships, social roles, or\n"
    "  interpersonal interactions.\n\n"
    "Rules:\n"
    "- For each category, list only direct quotes or close paraphrases from\n"
    "  the text -- do NOT invent evidence.\n"
    "- If no evidence exists for a category, leave it blank.\n"
    "- Output MUST follow the exact format below (one category per line).\n"
)

USR_PROMPT_EXTRACTOR = (
    "Text:\n"
    "{text}\n\n"
    "Extract evidence for each category. Use the exact format:\n\n"
    "Emotion: <evidence>\n"
    "Cognition: <evidence>\n"
    "Sensory Perception: <evidence>\n"
    "Sociality: <evidence>\n"
)


def build_extractor_prompts(text: str) -> tuple[str, str]:
    sys_prompt = SYS_PROMPT_EXTRACTOR
    user_prompt = USR_PROMPT_EXTRACTOR.format(text=text)
    return sys_prompt, user_prompt


def parse_extractor_output(raw: str) -> dict[str, str]:
    result = {cat: "" for cat in CATEGORIES}
    current_cat = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        matched = False
        for cat in CATEGORIES:
            if line.startswith(cat + ":"):
                current_cat = cat
                result[cat] = line[len(cat) + 1:].strip()
                matched = True
                break
        if not matched and current_cat is not None:
            result[current_cat] = (result[current_cat] + " " + line).strip()
    return result

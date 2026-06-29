TRAITS = {
    "Neuroticism": {
        "high": "High Neuroticism scorers are prone to anxiety, emotional instability, stress, and negative emotions. They may struggle with impulse control and coping under pressure.",
        "low": "Low Neuroticism scorers are emotionally stable, calm, resilient, relaxed, and able to handle stressful situations effectively.",
    },
    "Extraversion": {
        "high": "Extraverts are outgoing, energetic, talkative, assertive, enthusiastic, and gain energy from social interaction.",
        "low": "Introverts are reserved, quiet, independent, reflective, and generally prefer solitary or small-group settings.",
    },
    "Openness": {
        "high": "People high in Openness are imaginative, curious, creative, intellectually adventurous, and receptive to new ideas and experiences.",
        "low": "People low in Openness are practical, conventional, traditional, and prefer familiarity, routine, and concrete experiences.",
    },
    "Agreeableness": {
        "high": "Highly agreeable individuals are compassionate, cooperative, trusting, altruistic, sympathetic, and eager to help others.",
        "low": "Low Agreeableness is characterized by skepticism, competitiveness, bluntness, self-interest, and reduced concern for interpersonal harmony.",
    },
    "Conscientiousness": {
        "high": "Highly conscientious individuals are organized, disciplined, dependable, hardworking, responsible, and goal-oriented.",
        "low": "Low Conscientiousness is associated with spontaneity, disorganization, carelessness, impulsiveness, and reduced self-discipline.",
    },
}


SYS_PROMPT = """
You are an expert in personality psychology and psychometrics.

Your task is to infer a single Big Five personality trait from a user's text.

You will be given:
- The target personality trait.
- Definitions of High and Low levels.

Your job is to determine whether the user exhibits a HIGH or LOW level of that trait.

Rules:
- Use only evidence from the provided text.
- Do not infer unsupported characteristics.
- Output exactly one word: high or low.
- Do not provide explanations unless explicitly requested.
"""


ZEROSHOT_USR_PROMPT = """
Trait: {trait_name}

High:
{high_definition}

Low:
{low_definition}

Text:
<text>

Based on the text, determine whether the user's {trait_name} is high or low.

Answer with exactly one word:

high
or
low
"""


ONESHOT_USR_PROMPT = """
Example:

Trait: {trait_name}

Text:
{example_text}

Answer:
{example_label}

---

Trait: {trait_name}

High:
{high_definition}

Low:
{low_definition}

Text:
<text>

Based on the text, determine whether the user's {trait_name} is high or low.

Answer with exactly one word:

high
or
low
"""


COT_USR_PROMPT = """
Trait: {trait_name}

High:
{high_definition}

Low:
{low_definition}

Text:
<text>

Analyze the user's {trait_name}.

Follow these steps:
1. Identify relevant evidence from the text.
2. Explain what that evidence suggests.
3. Decide whether the trait is high or low.

Answer in exactly this format:

Evidence:
<quoted evidence>

Reasoning:
<your reasoning>

Label:
high or low
"""
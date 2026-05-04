
DISCOVERY_SYS_PROMPT = """
You are an expert in personality psychology, psycholinguistics, and
qualitative text analysis.

You will help discover the linguistic and behavioural patterns that
distinguish HIGH vs LOW levels of a single Big Five personality trait
through a short multi-turn dialogue.

Guidelines:
- Reason carefully and ground every claim in the provided texts.
- Focus on language style, recurring topics, values, concerns, emotional
  tone, pronoun usage, and behavioural indicators.
- Be concise but specific. Avoid generic statements that could apply to
  any group.
- Do not invent evidence that is not present in the texts.
"""


TURN1_LOW_PROMPT = """
Analyze the following LOW {trait_name} examples:

{examples}

What patterns characterize LOW {trait_name}?
Focus on: language style, values, concerns, behavioral indicators.

Provide a structured description of the patterns you observe (bullet
points are fine). Be specific and reference the kind of cues
(vocabulary, sentence structure, topics, emotional tone, pronouns, etc.)
that recur across these LOW {trait_name} writers.
"""


TURN2_HIGH_PROMPT = """
Now analyze the following HIGH {trait_name} examples:

{examples}

What patterns characterize HIGH {trait_name}?
Focus on: language style, values, concerns, behavioral indicators.

Provide a structured description of the patterns you observe (bullet
points are fine). Be specific and reference the kind of cues
(vocabulary, sentence structure, topics, emotional tone, pronouns, etc.)
that recur across these HIGH {trait_name} writers.
"""


TURN3_SYNTHESIS_PROMPT = """
Summary so far:

- LOW {trait_name} patterns:
{low_patterns}

- HIGH {trait_name} patterns:
{high_patterns}

Based on the contrast between LOW and HIGH {trait_name}, extract 5 to 7
clear, actionable principles that can be used to distinguish HIGH from
LOW {trait_name} when reading a new text.

Each principle must be a single line in EXACTLY this format:

[Principle name]: [HIGH description] vs [LOW description]

Requirements:
- Number the principles 1. 2. 3. ...
- Each principle should capture ONE concrete, observable contrast
  (lexical, stylistic, topical, or behavioural).
- Keep descriptions short (one clause each) and parallel in structure.
- Do not add commentary, headers, or explanations outside the numbered
  list.
"""


# ---------------------------------------------------------------------------
# PHASE 2 - DIRECT CLASSIFICATION USING DISCOVERED PRINCIPLES
# ---------------------------------------------------------------------------

SYS_PROMPT = """
You are an expert in personality psychology and psychometrics.

Your task is to infer a single Big Five personality trait from a user's text.

You will be given:
- The target personality trait.
- Principles to distinguish High and Low levels.

Your job is to determine whether the user exhibits a HIGH or LOW level of that trait.

Rules:
- Use only evidence from the provided text.
- Do not infer unsupported characteristics.
- Output exactly one word: high or low.
- Do not provide explanations unless explicitly requested.
"""


PRINCIPLE_PROMPT = """
Trait: {trait_name}

{trait_name} distinguishing principles:
{principles}

Text:
<text>

Based on the principles above and evidence from the text, determine
whether the user's {trait_name} is high or low.

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

{trait_name} distinguishing principles:
{principles}

Text:
<text>

Based on the principles above and evidence from the text, determine
whether the user's {trait_name} is high or low.

Answer with exactly one word:

high
or
low
"""




RAG_PRINCIPLE_PROMPT = """
Trait: {trait_name}

{trait_name} distinguishing principles:
{principles}

---

The following are the {top_k} most similar texts from the training set,
with their known labels and extracted psychological evidence.
Use them as additional reference to calibrate your judgment:

{similar_context}

---

Text to classify:
<text>

Based on the principles above and the similar examples, determine
whether the user's {trait_name} is high or low.

Answer with exactly one word:

high
or
low
"""


RAG_ONESHOT_USR_PROMPT = """
Trait: {trait_name}

{trait_name} distinguishing principles:
{principles}

---

The following are the {top_k} most similar texts from the training set,
with their known labels and extracted psychological evidence.
Use them as examples to calibrate your judgment:

{similar_context}

---

Text to classify:
<text>

Based on the principles above and the similar examples, determine
whether the user's {trait_name} is high or low.

Answer with exactly one word:

high
or
low
"""


# ---------------------------------------------------------------------------
# CHAIN-OF-THOUGHT VARIANTS
# These prompts instruct the model to reason step by step before committing
# to a label. Every CoT prompt ends with a mandatory Label: high / Label: low
# line so that extract_cot() can parse the answer reliably.
# ---------------------------------------------------------------------------

SYS_PROMPT_COT = """
You are an expert in personality psychology and psychometrics.

Your task is to infer a single Big Five personality trait from a user's text.

You will be given:
- The target personality trait.
- Principles to distinguish High and Low levels.

Your job is to determine whether the user exhibits a HIGH or LOW level of that trait.

Rules:
- Use only evidence from the provided text.
- Do not infer unsupported characteristics.
- Think step by step: identify relevant evidence, apply each principle,
  weigh the evidence, then state your conclusion.
- End your response with EXACTLY this line (no punctuation, no extra words):
  Label: high
  or
  Label: low
"""


COT_PRINCIPLE_PROMPT = """
Trait: {trait_name}

{trait_name} distinguishing principles:
{principles}

Text:
<text>

Think step by step:
1. Identify concrete evidence in the text relevant to {trait_name}.
2. Apply each principle to the evidence - note whether each points high or low.
3. Weigh the overall pattern of evidence.
4. State your conclusion.

End with exactly one line in this format:
Label: high
or
Label: low
"""


COT_ONESHOT_USR_PROMPT = """
Example:

Trait: {trait_name}

Text:
{example_text}

Reasoning:
[evidence noted, principles applied]

Label: {example_label}

---

Trait: {trait_name}

{trait_name} distinguishing principles:
{principles}

Text:
<text>

Think step by step:
1. Identify concrete evidence in the text relevant to {trait_name}.
2. Apply each principle to the evidence - note whether each points high or low.
3. Weigh the overall pattern of evidence.
4. State your conclusion.

End with exactly one line in this format:
Label: high
or
Label: low
"""


COT_RAG_PRINCIPLE_PROMPT = """
Trait: {trait_name}

{trait_name} distinguishing principles:
{principles}

---

The following are the {top_k} most similar texts from the training set,
with their known labels and extracted psychological evidence.
Use them as calibration reference - note how their evidence maps to the label:

{similar_context}

---

Text to classify:
<text>

Think step by step:
1. Identify concrete evidence in the text relevant to {trait_name}.
2. Apply each principle to the evidence - note whether each points high or low.
3. Compare the text's evidence pattern against the similar examples above.
4. Weigh the overall pattern and state your conclusion.

End with exactly one line in this format:
Label: high
or
Label: low
"""


COT_RAG_ONESHOT_USR_PROMPT = """
Trait: {trait_name}

{trait_name} distinguishing principles:
{principles}

---

The following are the {top_k} most similar texts from the training set,
with their known labels and extracted psychological evidence.
Use them as calibration reference - note how their evidence maps to the label:

{similar_context}

---

Text to classify:
<text>

Think step by step:
1. Identify concrete evidence in the text relevant to {trait_name}.
2. Apply each principle to the evidence - note whether each points high or low.
3. Compare the text's evidence pattern against the similar examples above.
4. Weigh the overall pattern and state your conclusion.

End with exactly one line in this format:
Label: high
or
Label: low
"""

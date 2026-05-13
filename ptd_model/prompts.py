
TRAITS = {
    "Neuroticism": {
        "high": "High Neuroticism scorers are prone to anxiety, emotional instability, stress, and negative emotions. They may struggle with impulse control and coping under pressure.",
        "low":  "Low Neuroticism scorers are emotionally stable, calm, resilient, relaxed, and able to handle stressful situations effectively.",
    },
    "Extraversion": {
        "high": "Extraverts are outgoing, energetic, talkative, assertive, enthusiastic, and gain energy from social interaction.",
        "low":  "Introverts are reserved, quiet, independent, reflective, and generally prefer solitary or small-group settings.",
    },
    "Openness": {
        "high": "People high in Openness are imaginative, curious, creative, intellectually adventurous, and receptive to new ideas and experiences.",
        "low":  "People low in Openness are practical, conventional, traditional, and prefer familiarity, routine, and concrete experiences.",
    },
    "Agreeableness": {
        "high": "Highly agreeable individuals are compassionate, cooperative, trusting, altruistic, sympathetic, and eager to help others.",
        "low":  "Low Agreeableness is characterized by skepticism, competitiveness, bluntness, self-interest, and reduced concern for interpersonal harmony.",
    },
    "Conscientiousness": {
        "high": "Highly conscientious individuals are organized, disciplined, dependable, hardworking, responsible, and goal-oriented.",
        "low":  "Low Conscientiousness is associated with spontaneity, disorganization, carelessness, impulsiveness, and reduced self-discipline.",
    },
}


SYS_PROMPT = """
You are an expert in personality psychology and psychometrics.

Your task is to infer a single Big Five personality trait from a user's text.

You will be given:
- The target personality trait.
- HIGH and LOW definitions of the trait.

Your job is to determine whether the user exhibits a HIGH or LOW level of that trait.

Rules:
- Use only evidence from the provided text.
- Do not infer unsupported characteristics.
- Output exactly one word: high or low.
- Do not provide explanations unless explicitly requested.
"""


SYS_PROMPT_REASONED = """
You are an expert in personality psychology and psychometrics.

Your task is to infer a single Big Five personality trait from a user's text
and to expose your reasoning in a strictly structured format so it can be
audited.

You will be given:
- The target personality trait (with HIGH and LOW definitions).
- A small set of similar texts retrieved from a labelled corpus, with
  their known labels and extracted psychological evidence.

Rules:
- Use only evidence from the provided text. Do NOT invent details.
- Quote or paraphrase concrete cues; abstract trait words alone are not
  evidence.
- Output MUST follow the XML tag structure below, in this exact order,
  with no extra text outside the tags.

Output format (replicate verbatim, fill in the contents):

<evidence>
- one concrete cue from the text
- another concrete cue
- ...
</evidence>
<facet_check>
- facet name -> high|low|mixed -> brief reason
- ...
</facet_check>
<example_alignment>
The test text most closely matches Similar Profile <i> (label: <label>) because <reason>.
The test text diverges from Similar Profile <j> on <axis>.
</example_alignment>
<verdict>
1-2 sentence synthesis of why the overall pattern points high or low.
</verdict>
<label>high</label>

The final <label> tag MUST contain exactly one word: high or low. Nothing else.
"""


DEF_ZEROSHOT_PROMPT = """
Trait: {trait_name}

HIGH {trait_name}: {definition_high}
LOW  {trait_name}: {definition_low}

---

The following are the {top_k} most similar texts from the training set
with their known labels for {trait_name}:

{similar_context}

---

Text to classify:
<text>

Based on the definitions and similar examples above, determine whether
the user's {trait_name} is high or low.

Answer with exactly one word:

high
or
low
"""


DEF_ONESHOT_PROMPT = """
Trait: {trait_name}

HIGH {trait_name}: {definition_high}
LOW  {trait_name}: {definition_low}

---

The following are the {top_k} most similar profiles from the training set,
with their known labels and extracted psychological evidence for {trait_name}.
Use them as calibration anchors:

{similar_context}

---

Text to classify:
<text>

Based on the definitions and similar examples above, determine whether
the user's {trait_name} is high or low.

Answer with exactly one word:

high
or
low
"""


REASONED_RAG_DEF_ONESHOT_PROMPT = """
Trait: {trait_name}

HIGH {trait_name}: {definition_high}
LOW  {trait_name}: {definition_low}

---

The following are the {top_k} most similar texts from the training set,
with their known labels and extracted psychological evidence. Use them
as calibration anchors - note how each one's cues map to its label.

{similar_context}

---

Text to classify:
<text>

Reason step by step and emit your output in the EXACT XML structure
specified in the system message:

<evidence> ... </evidence>
<facet_check> ... </facet_check>
<example_alignment> ... </example_alignment>
<verdict> ... </verdict>
<label>high</label>   (or <label>low</label>)

Do not output any text outside these tags.
"""

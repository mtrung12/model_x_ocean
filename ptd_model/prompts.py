
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


# ---------------------------------------------------------------------------
# Multi-LLM debate prompts (all-traits prediction + adversarial verification)
# ---------------------------------------------------------------------------

SYS_PROMPT_ALL_TRAITS = """\
You are an expert in personality psychology and psychometrics.
Predict ALL FIVE Big Five personality traits from the essay below.

CRITICAL OUTPUT RULE — do NOT write anything before or after these 5 lines:
<Openness>high</Openness>
<Conscientiousness>low</Conscientiousness>
<Extraversion>high</Extraversion>
<Agreeableness>low</Agreeableness>
<Neuroticism>high</Neuroticism>

Replace each value with your actual prediction for that trait.
The only legal values are the exact words:  high  or  low
Do not write "high or low", do not add punctuation, do not add prose.
"""

ALL_TRAITS_PROMPT = """\
BIG FIVE TRAIT DEFINITIONS:

Openness to Experience:
  HIGH: {openness_high}
  LOW:  {openness_low}

Conscientiousness:
  HIGH: {conscientiousness_high}
  LOW:  {conscientiousness_low}

Extraversion:
  HIGH: {extraversion_high}
  LOW:  {extraversion_low}

Agreeableness:
  HIGH: {agreeableness_high}
  LOW:  {agreeableness_low}

Neuroticism:
  HIGH: {neuroticism_high}
  LOW:  {neuroticism_low}

---
SIMILAR PROFILES ({top_k} from training set — calibration anchors):

{similar_context}

---
ESSAY TO CLASSIFY:
<text>

---
Output ONLY these 5 XML lines — replace each value with high or low:
<Openness>high</Openness>
<Conscientiousness>low</Conscientiousness>
<Extraversion>high</Extraversion>
<Agreeableness>low</Agreeableness>
<Neuroticism>high</Neuroticism>
"""

SYS_PROMPT_DEBATE = """\
You are an expert in personality psychology and psychometrics.

You previously made predictions about a text. Another model disagrees
with you on some traits. Reflect critically on your own reasoning.
Be specific: cite concrete evidence from the text, not abstract labels.

For every disputed trait answer exactly:
  1. Assumption:  the hidden premise behind your prediction
  2. Evidence:    a paraphrase (12-20 words) of a concrete cue in the text
  3. Falsifier:   what text evidence would force you to switch your answer

Use the section format shown below — one block per disputed trait.
Write nothing outside those blocks.
"""

DEBATE_PROBE_PROMPT = """\
Your predictions for this text:
{your_predictions}

The other model disagrees on these traits:
{disagreements}

For each disagreed trait, fill in the template below.
Do not write anything outside the === blocks.

{debate_template}

Text:
<text>
"""

# Built dynamically per call so each trait gets its own === block
_DEBATE_BLOCK_TEMPLATE = """\
=== {trait} ===
1. Assumption: [your hidden premise]
2. Evidence:   [12-20 word paraphrase from the text]
3. Falsifier:  [what would change your answer]"""


def _build_debate_template(disagreed_traits):
    """Return the === block template string for the given list of trait names."""
    return "\n\n".join(_DEBATE_BLOCK_TEMPLATE.format(trait=t) for t in disagreed_traits)


SYS_PROMPT_VERIFIER = """\
You are a senior personality psychology verification expert.

Two independent models predicted Big Five personality traits for a text
and disagreed on some traits. You receive:
  - The original text
  - Both models' full predictions
  - Each model's debate (assumption / evidence / falsifier per disputed trait)

Your job:
1. Critically evaluate each model's evidence and assumptions against the text.
2. Resolve every disputed trait with a clear reason.
3. Emit a final verdict for ALL FIVE traits.

Output format — follow this structure exactly:
<reasoning>
[Address every disputed trait: which model's evidence is stronger and why.]
</reasoning>
<predictions>
<Openness>high</Openness>
<Conscientiousness>low</Conscientiousness>
<Extraversion>high</Extraversion>
<Agreeableness>low</Agreeableness>
<Neuroticism>high</Neuroticism>
</predictions>

Rules:
- Replace each value in <predictions> with your actual verdict: high or low.
- Write nothing outside <reasoning>...</reasoning> and <predictions>...</predictions>.
- Do not repeat the models' XML tags inside <reasoning> — use plain text there.
"""

VERIFIER_PROMPT = """\
ORIGINAL TEXT:
\"\"\"
<text>
\"\"\"

---
MODEL A ({model_a_name}):
{model_a_predictions}

MODEL B ({model_b_name}):
{model_b_predictions}

---
AGREE:    {agreements}
DISAGREE: {disagreements}

---
MODEL A debate:
{model_a_debate}

---
MODEL B debate:
{model_b_debate}

---
Produce your output now.  Follow the format in the system message exactly:

<reasoning>
...
</reasoning>
<predictions>
<Openness>high</Openness>
<Conscientiousness>low</Conscientiousness>
<Extraversion>high</Extraversion>
<Agreeableness>low</Agreeableness>
<Neuroticism>high</Neuroticism>
</predictions>
"""

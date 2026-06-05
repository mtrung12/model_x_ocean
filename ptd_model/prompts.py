
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


TRAIT_NOTES = {
    "Neuroticism": """\
Scoring note — Neuroticism:
- Situational stress, fatigue, homesickness, exam pressure, or adjustment to
  new environments are universal reactions. Tag them [state]; they are NOT
  reliable evidence of high Neuroticism on their own.
- Tag as [trait] only recurring, cross-situational patterns of emotional
  instability that persist independent of an obvious external cause (e.g.
  habitual anxiety, chronic emotional dysregulation stated explicitly).
- Treat [state] cues as weak or neutral unless extreme or accompanied by
  explicit statements of habitual emotional difficulty.""",

    "Extraversion": """\
Scoring note — Extraversion:
- Expressive writing, enthusiastic tone, humor, strong opinions, excitement
  about activities, or vivid descriptions are stylistic features. Tag them
  [state]; they do NOT constitute evidence of Extraversion. A person can
  write expressively while being deeply introverted.
- Tag as [trait] only cues about social energy and orientation: actively
  seeking groups or parties, gaining energy from social interaction,
  discomfort with solitude, preferring large gatherings, or explicit
  statements about enjoying socialising.""",
}


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
 — distinguish transient states from stable traits:
    * Tag every evidence cue as [state] (temporary/situational) or
      [trait] (recurring/cross-situational). Weight [trait] cues heavily;
      treat [state] cues as weak or neutral.
    * Refer to the trait-specific scoring note in the user message for
      what counts as [state] vs [trait] for the current trait.
- HIGH requires positive trait-level evidence. If you have ZERO [trait] cues
  across all facets, your <label> MUST be low. Do not accumulate [state] cues
  to reach high — state-only evidence is insufficient for a high verdict.
- Output MUST follow the XML tag structure below, in this exact order,
  with no extra text outside the tags.

Output format (replicate verbatim, fill in the contents):

<evidence>
- [state|trait] one concrete cue from the text
- [state|trait] another concrete cue
- ...
</evidence>
<facet_check>
- facet name -> high|low|mixed -> brief reason (dominant evidence type: state|trait)
- ...
</facet_check>
<cue_tally>
[trait] cues: N  |  [state] cues: M
Presumption: HIGH (N > 0) or LOW (N = 0)
</cue_tally>
<example_alignment>
The test text most closely matches Similar Profile <i> (label: <label>) because <reason>.
The test text diverges from Similar Profile <j> on <axis>.
</example_alignment>
<verdict>
1-2 sentence synthesis. MUST be consistent with your <cue_tally>: if
[trait] count = 0, you MUST conclude low. Do NOT describe [state] cues
as "trait-level evidence" here.
</verdict>
<label>high</label>

The final <label> tag MUST contain exactly one word: high or low. Nothing else.
"""


REASONED_RAG_DEF_ONESHOT_PROMPT = """
Trait: {trait_name}

HIGH {trait_name}: {definition_high}
LOW  {trait_name}: {definition_low}

{trait_note}
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
<cue_tally> ... </cue_tally>
<example_alignment> ... </example_alignment>
<verdict> ... </verdict>
<label>high</label>   (or <label>low</label>)

Do not output any text outside these tags.
"""

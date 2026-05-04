"""Prompt construction and parsing for the 30-facet psychological profiler.

The profiler reads one essay and emits a structured psychological profile
(~1000 tokens) keyed to the 30 NEO-PI-R facets plus a short LIWC-style
linguistic fingerprint. The output is intended for:

  (a) embedding into a per-trait FAISS index for retrieval-time similarity,
  (b) sliced display as compact few-shot exemplars at inference time.

Bias-reduction techniques layered into the prompts (see README of the module
for full rationale):

  1. Label-aware but counterfactual framing. The profiler IS told the five
     trait labels, but the instructions explicitly require it to surface
     evidence on BOTH sides for every facet -- preventing the "label says
     high so write high" collapse seen in label-conditioned generation.
  2. Anti-genre-bias clause. Conscientiousness and Neuroticism are most
     commonly mis-profiled in informal student journals because the
     register itself reads as "low C / high N". The system prompt names
     this trap explicitly so the profiler discounts register-only signals.
  3. First-class "n/e" (not-evidenced) value. Facets like A1 Trust and A3
     Altruism are typically unobservable in solo introspective text
     (Pennebaker & King, 1999; Mairesse et al., 2007). Allowing "n/e" as
     a valid signal removes the pressure to fabricate.
  4. Mandatory paraphrase anchoring. Evidence cell must be a 12-20 word
     paraphrase referring to concrete behaviour or stated belief; abstract
     trait inferences ("she is anxious") are forbidden.
  5. No-justification-by-label rule. The profiler is told the label is a
     loose sanity anchor only; it must not write "because the label is X".
  6. Closing self-check. After emitting all 30 lines, the profiler is asked
     to demote any facet whose evidence is thin to "mod" or "n/e".

References (for prompt content choices):
  - Costa & McCrae 1992 (NEO-PI-R facet structure)
  - Pennebaker & King 1999 (LIWC-Big-Five correlations on this dataset)
  - Mairesse et al. 2007 (linguistic predictors of personality)
  - Schwartz et al. 2013 (open-vocabulary Big-Five predictors)
"""

from __future__ import annotations

from typing import Dict, List, Tuple


# -----------------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------------

# Token budget for the profiler output. ~25 tokens/facet x 30 + ~120 for the
# linguistic block + ~30 header ~= 900 tokens. We give 1024 as a soft ceiling
# so the model has room without dilution.
PROFILER_MAX_TOKENS = 1024

# Controlled vocabulary for the signal cell. "n/e" = not evidenced (facet
# is not observable in this kind of writing -- different from "none" which
# means observable but neutral / absent).
SIGNAL_VOCAB = ("high", "mod", "low", "none", "n/e")


# (code, name, trait, short_definition)
# `trait` is a dataset trait abbreviation (cOPN, cCON, cEXT, cAGR, cNEU)
# `short_definition` is a 6-12 word psychometric gloss used in the prompt
# to anchor the model's interpretation of each facet.
FACETS: List[Tuple[str, str, str, str]] = [
    # Neuroticism
    ("N1", "anxiety",          "cNEU", "free-floating worry, fear, tension"),
    ("N2", "hostility",        "cNEU", "anger, irritation, resentment toward others"),
    ("N3", "depression",       "cNEU", "sadness, hopelessness, low mood"),
    ("N4", "self-conscious",   "cNEU", "shame, embarrassment, social discomfort"),
    ("N5", "impulsiveness",    "cNEU", "difficulty resisting urges or delaying"),
    ("N6", "vulnerability",    "cNEU", "feeling overwhelmed, helpless under stress"),
    # Extraversion
    ("E1", "warmth",           "cEXT", "affectionate, friendly engagement with others"),
    ("E2", "gregariousness",   "cEXT", "preference for company, group settings"),
    ("E3", "assertiveness",    "cEXT", "leadership, dominance, decisive voice"),
    ("E4", "activity",         "cEXT", "energetic pace, busy schedule, motion"),
    ("E5", "excitement-seek",  "cEXT", "thrill-seeking, novelty appetite"),
    ("E6", "positive emotion", "cEXT", "joy, enthusiasm, optimism"),
    # Openness
    ("O1", "fantasy",          "cOPN", "vivid imagination, daydreaming"),
    ("O2", "aesthetics",       "cOPN", "appreciation of art, beauty, nature"),
    ("O3", "feelings",         "cOPN", "emotional depth, introspective awareness"),
    ("O4", "actions/variety",  "cOPN", "openness to new experiences, change"),
    ("O5", "ideas",            "cOPN", "intellectual curiosity, abstract thought"),
    ("O6", "values",           "cOPN", "willingness to re-examine values, norms"),
    # Agreeableness
    ("A1", "trust",            "cAGR", "belief in others' good intentions"),
    ("A2", "straightforward",  "cAGR", "candor, frankness, low manipulation"),
    ("A3", "altruism",         "cAGR", "active concern for others' welfare"),
    ("A4", "compliance",       "cAGR", "deference, low antagonism in conflict"),
    ("A5", "modesty",          "cAGR", "humility, low self-promotion"),
    ("A6", "tender-minded",    "cAGR", "sympathy, empathy with others' pain"),
    # Conscientiousness
    ("C1", "competence",       "cCON", "sense of capability, preparedness"),
    ("C2", "order",            "cCON", "tidiness, organization, structure"),
    ("C3", "dutifulness",      "cCON", "adherence to obligations, ethics"),
    ("C4", "achievement",      "cCON", "striving for goals, ambition"),
    ("C5", "self-discipline",  "cCON", "follow-through, resisting distraction"),
    ("C6", "deliberation",     "cCON", "careful planning, thinking before acting"),
]


# Lines in the LINGUISTIC fingerprint block. Each line is one fixed key with
# a free-text value (~15-20 tokens). These follow Pennebaker/Schwartz's
# strongest meta-analytic predictors.
LINGUISTIC_LINES: List[Tuple[str, str]] = [
    ("pronouns",  "I/we/you/they density and dominance pattern"),
    ("emotion",   "pos/neg balance + dominant affect"),
    ("tense",     "past/present/future emphasis"),
    ("cognitive", "insight/causal/tentative density (low/mod/high)"),
    ("social",    "social-process word density (low/mod/high)"),
    ("register",  "informal|mixed|formal + exploratory|focused"),
    ("length",    "approx word count, sentence style, lexical diversity"),
]


TRAIT_FULL_NAME = {
    "cOPN": "Openness",
    "cCON": "Conscientiousness",
    "cEXT": "Extraversion",
    "cAGR": "Agreeableness",
    "cNEU": "Neuroticism",
}


# -----------------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------------

SYS_PROMPT_PROFILER = """\
You are a personality-psychology coding analyst. Your job is to read a
short stream-of-consciousness essay and emit a STRUCTURED psychological
profile keyed to the 30 NEO-PI-R facets plus a short linguistic
fingerprint.

The profile will be stored in a vector database and used as retrieval
context for a separate downstream classifier. Quality of the profile
directly determines the quality of every downstream prediction.

You will be given the writer's five Big-Five trait labels (high/low) as
LOOSE ANCHORS. The labels are weak and noisy. You MUST NOT:
  - justify any evidence by referencing the label,
  - write evidence that is not present in the text,
  - make a facet "high" just because the matching trait label is "high",
  - make a facet "low" just because the matching trait label is "low".

For EVERY facet you must independently scan the text for evidence on
BOTH the high side AND the low side, then decide a signal in
{high, mod, low, none, n/e}:
  - high : clear, repeated, on-text evidence in the high direction
  - low  : clear, repeated, on-text evidence in the low direction
  - mod  : mixed evidence, or single weak signal in either direction
  - none : facet IS observable in this writing genre but the writer
           shows neutral / absent expression of it
  - n/e  : "not evidenced" -- facet is not realistically observable
           in solo introspective journal writing (e.g. A1 trust,
           A3 altruism, E1 warmth, E2 gregariousness in many essays).
           Use this whenever you would otherwise have to fabricate.

Critical anti-genre-bias rule:
  These essays are informal student stream-of-consciousness journals.
  Their casual register, run-on sentences, and informal vocabulary
  are GENRE features, not personality features. DO NOT push C2 order,
  C5 self-discipline, C6 deliberation, or N1 anxiety toward "low/high"
  on register evidence alone. Demand content-level evidence: explicit
  references to plans, deadlines, follow-through, schedule, or their
  absence; explicit statements of worry, calm, or affect. If the
  evidence is only stylistic, mark "mod" or "none".

Evidence cell rule:
  Every non-"n/e" facet line MUST cite concrete behaviour, stated
  belief, or affect from the text in 12-20 words of paraphrase
  (not verbatim quote). Examples:
    "C5 self-discipline | low  | admits sorority distractions; mind wanders;
       recognizes need for focus but does not enact"
    "O5 ideas           | high | weighs premed vs criminal psych; reads
       non-fiction to inform major choice"

Output format is RIGID. Do not add prose, headers, or commentary outside
the specified blocks.

After emitting all 30 facet lines and the linguistic block, perform a
silent self-check: re-read each "high" or "low" line and ASK YOURSELF
"would a skeptical reader see this evidence as decisive?" If not,
demote the line to "mod" or "n/e" before finalizing.
"""


def _format_facet_reference() -> str:
    lines = []
    for code, name, _trait, defn in FACETS:
        lines.append(f"  {code} {name:<18} = {defn}")
    return "\n".join(lines)


def _format_linguistic_reference() -> str:
    lines = []
    for key, defn in LINGUISTIC_LINES:
        lines.append(f"  {key:<10}: {defn}")
    return "\n".join(lines)


USR_PROMPT_PROFILER = """\
TRAIT LABELS (loose anchor only; do not justify by these):
{labels_block}

FACET REFERENCE (use these codes verbatim, in this exact order):
{facet_reference}

LINGUISTIC REFERENCE (fixed keys, in this order):
{linguistic_reference}

OUTPUT FORMAT (exactly):

[FACETS]
N1 anxiety            | <signal> | <12-20 word paraphrase or empty if n/e>
N2 hostility          | <signal> | <...>
... (all 30 facets, in the order shown above) ...
C6 deliberation       | <signal> | <...>

[LINGUISTIC]
pronouns:  <one line>
emotion:   <one line>
tense:     <one line>
cognitive: <one line>
social:    <one line>
register:  <one line>
length:    <one line>

Rules recap:
  - signal is one of: high, mod, low, none, n/e
  - evidence is paraphrase, NOT a quote
  - never justify with reference to the label
  - default to "n/e" or "mod" rather than fabricating
  - register/style alone is not evidence for C/N facets

ESSAY:
\"\"\"
{text}
\"\"\"

Now produce the profile. Output ONLY the [FACETS] and [LINGUISTIC] blocks.
"""


def _format_labels_block(labels: Dict[str, str]) -> str:
    if not labels:
        return "  (no labels available; profile blind)"
    out = []
    for trait_code in ("cOPN", "cCON", "cEXT", "cAGR", "cNEU"):
        if trait_code in labels:
            full = TRAIT_FULL_NAME[trait_code]
            out.append(f"  {full:<18} ({trait_code}) = {labels[trait_code]}")
    return "\n".join(out)


def build_profiler_prompts(
    text: str,
    trait_labels: Dict[str, str] | None = None,
) -> Tuple[str, str]:
    """Construct (system_prompt, user_prompt) for the profiler.

    Parameters
    ----------
    text : str
        The essay to profile.
    trait_labels : dict[str, str] | None
        Optional mapping of trait code (cOPN/cCON/cEXT/cAGR/cNEU) to
        "high" or "low". When provided, used as a loose anchor in the
        prompt. The prompt explicitly forbids label-justified reasoning,
        so providing labels is a calibration aid, not a leakage path.
    """
    sys_p = SYS_PROMPT_PROFILER
    usr_p = USR_PROMPT_PROFILER.format(
        labels_block=_format_labels_block(trait_labels or {}),
        facet_reference=_format_facet_reference(),
        linguistic_reference=_format_linguistic_reference(),
        text=text.strip(),
    )
    return sys_p, usr_p


# -----------------------------------------------------------------------------
# Output parsing
# -----------------------------------------------------------------------------

_FACET_CODES = [code for code, *_ in FACETS]
_FACET_NAMES = {code: name for code, name, *_ in FACETS}
_LING_KEYS = [key for key, _ in LINGUISTIC_LINES]


def _normalize_signal(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("not evidenced", "n/e").replace("not-evidenced", "n/e")
    if s in SIGNAL_VOCAB:
        return s
    # Tolerate common variants
    if s in ("h", "high+", "+high"):
        return "high"
    if s in ("l", "low-", "-low"):
        return "low"
    if s in ("m", "moderate", "mixed"):
        return "mod"
    if s in ("neutral", "absent", "no", "off"):
        return "none"
    if s in ("na", "n.a.", "ne", "n_e", "not_observed", "n/a"):
        return "n/e"
    return ""  # unparseable; treated as missing


def parse_profile_output(raw: str) -> Dict:
    """Parse the profiler's output into a structured dict.

    Returns
    -------
    dict with keys:
      "facets":     dict[code -> {"signal": str, "evidence": str}]
      "linguistic": dict[key  -> str]
      "raw":        the original string (kept for debugging / display)
      "valid":      bool, True iff all 30 facets parsed and no signal blank
    """
    facets: Dict[str, Dict[str, str]] = {
        code: {"signal": "", "evidence": ""} for code in _FACET_CODES
    }
    linguistic: Dict[str, str] = {key: "" for key in _LING_KEYS}

    section = None  # "facets" | "linguistic" | None
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        # Section markers
        if s.lower().startswith("[facets"):
            section = "facets"
            continue
        if s.lower().startswith("[linguistic"):
            section = "linguistic"
            continue
        if s.startswith("[") and s.endswith("]"):
            # unknown section header - ignore until known
            section = None
            continue

        if section == "facets":
            # Expect "<CODE> <name...> | <signal> | <evidence>"
            if "|" not in s:
                continue
            parts = [p.strip() for p in s.split("|")]
            if len(parts) < 2:
                continue
            head = parts[0]
            signal_raw = parts[1] if len(parts) >= 2 else ""
            evidence = parts[2] if len(parts) >= 3 else ""
            # Extract code as the first whitespace-delimited token
            head_tokens = head.split()
            if not head_tokens:
                continue
            code = head_tokens[0].upper().rstrip(".:,")
            if code not in facets:
                continue
            sig = _normalize_signal(signal_raw)
            facets[code] = {"signal": sig, "evidence": evidence}
            continue

        if section == "linguistic":
            # Expect "<key>: <value>"
            if ":" not in s:
                continue
            key, _, val = s.partition(":")
            key_norm = key.strip().lower()
            if key_norm in linguistic:
                linguistic[key_norm] = val.strip()

    # Validity: every facet has a recognised signal in SIGNAL_VOCAB
    valid = all(f["signal"] in SIGNAL_VOCAB for f in facets.values())

    return {
        "facets": facets,
        "linguistic": linguistic,
        "raw": raw,
        "valid": valid,
    }


# -----------------------------------------------------------------------------
# Slicing helpers (for inference-time view construction)
# -----------------------------------------------------------------------------

# Per-trait facet selection for the inference-time slice. Drops facets that
# are typically n/e or weakly-observable in journal text.
TRAIT_FACETS_FOR_INFERENCE: Dict[str, List[str]] = {
    "cOPN": ["O3", "O4", "O5", "O6", "O1", "O2"],
    "cCON": ["C1", "C2", "C3", "C4", "C5", "C6"],
    "cEXT": ["E3", "E4", "E5", "E6"],
    "cAGR": ["A2", "A4", "A5", "A6"],
    "cNEU": ["N1", "N3", "N5", "N6"],
}

# Per-trait linguistic-block keys to keep in the inference slice.
TRAIT_LING_FOR_INFERENCE: Dict[str, List[str]] = {
    "cOPN": ["cognitive", "register", "tense"],
    "cCON": ["tense", "register", "cognitive"],
    "cEXT": ["social", "emotion", "pronouns"],
    "cAGR": ["social", "emotion"],
    "cNEU": ["pronouns", "emotion", "tense"],
}


def slice_profile_for_trait(profile: Dict, trait_code: str) -> str:
    """Render a compact, trait-focused brief from a parsed profile.

    Output is plain text (~120-180 tokens) suitable for inclusion as a
    few-shot exemplar in a downstream classifier prompt.
    """
    facets = profile.get("facets", {})
    ling = profile.get("linguistic", {})
    out_lines = []
    for code in TRAIT_FACETS_FOR_INFERENCE.get(trait_code, []):
        f = facets.get(code)
        if not f or not f["signal"]:
            continue
        name = _FACET_NAMES.get(code, code)
        sig = f["signal"]
        ev = f["evidence"]
        if ev:
            out_lines.append(f"{code} {name:<18}| {sig:<4}| {ev}")
        else:
            out_lines.append(f"{code} {name:<18}| {sig:<4}|")
    ling_kept = []
    for key in TRAIT_LING_FOR_INFERENCE.get(trait_code, []):
        v = ling.get(key)
        if v:
            ling_kept.append(f"{key}: {v}")
    if ling_kept:
        out_lines.append("Linguistic: " + " | ".join(ling_kept))
    return "\n".join(out_lines)

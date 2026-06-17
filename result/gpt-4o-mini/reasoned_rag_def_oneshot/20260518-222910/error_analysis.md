# Error Analysis: `20260518-222910`

Sources reviewed:

- `log/gpt-4o-mini/reasoned_rag_def_oneshot/20260518-222910_log.txt`
- `result/gpt-4o-mini/reasoned_rag_def_oneshot/20260518-222910/predictions.csv`

## Quick take

This pipeline is not stable yet because it is strongly biased toward predicting `high`, especially for `EXT` and `NEU`.

Trait-level accuracy from `predictions.csv`:

| Trait | Accuracy | True High | True Low | Pred High | Pred Low |
|---|---:|---:|---:|---:|---:|
| OPN | 0.72 | 36 | 14 | 48 | 2 |
| CON | 0.70 | 27 | 23 | 28 | 22 |
| EXT | 0.28 | 12 | 38 | 44 | 6 |
| AGR | 0.56 | 29 | 21 | 41 | 9 |
| NEU | 0.54 | 24 | 26 | 47 | 3 |

`EXT` is the clearest failure:

- true `high`: 12
- true `low`: 38
- predicted `high`: 44
- predicted `low`: 6
- `high` recall: `0.8333`
- `low` recall: `0.1053`

So the model is not learning low Extraversion well. It mostly flips low `EXT` samples into `high`.

## What goes wrong on low `EXT`

### 1. The model confuses energy, humor, and expressive writing with Extraversion

Many false samples are lively, funny, or emotionally expressive, but that is not the same as being outgoing, socially dominant, or energized by interaction.

Examples from false low-`EXT` samples:

- `idx 5`: music, jokes, excitement, and slang-heavy updates. The text is expressive, but not clearly socially driven.
- `idx 14`: lots of opera/music enthusiasm and high verbal energy, but much of it is solitary reflection and niche interest talk.
- `idx 49`: sports opinions and excitement, but that is still opinion broadcasting, not evidence of gregariousness or assertive social behavior.

This matches the log behavior: the Extraversion reasoning repeatedly treats enthusiasm and positive emotion as enough evidence for `high`.

### 2. The model underweights introversion cues inside mixed-status bundles

The texts are long concatenations of short statuses separated by `||`. Many low-`EXT` people still show occasional excitement, events, travel, or social references. The pipeline seems to latch onto the loudest positive snippets and ignore the broader pattern.

Examples:

- `idx 0`: mixed life updates, travel, exams, hobbies, and frustration. There is no strong sustained signal of outgoingness, but the model still predicts `high`.
- `idx 6`: homework, flu, deadlines, and isolated self-talk dominate, yet the presence of a few upbeat lines appears to pull the label upward.
- `idx 16`: recovery from illness and basic social thanks are treated as stronger social evidence than they should be.

In other words, the current pipeline is doing salience matching, not trait balancing.

### 3. Retrieval likely brings in misleading anchors

The prompt asks the model to calibrate against 5 similar profiles. If retrieval is semantically similar on topic or tone rather than trait-discriminative evidence, the LLM gets nudged toward the wrong label.

From the log, the retrieved examples for Extraversion often contain:

- high positive emotion
- busy schedules
- mentions of friends or events
- excitement words

Those cues are correlated with `high EXT`, but they are not sufficient. For low-`EXT` authors, the important negative cues are things like:

- preference for solitary activity
- reflective rather than socially assertive tone
- low social initiative
- low evidence of group-seeking behavior

The current retrieved anchors do not seem to force that distinction strongly enough.

### 4. The prompt structure encourages over-interpretation

The model is required to fill:

- `facet_check`
- `example_alignment`
- `verdict`

for every sample, even when evidence is thin.

That creates a failure mode:

- weak cue appears
- model expands it into multiple `E` facets
- verdict becomes `high`

For `EXT`, a single cue like "excited", "great night", "fun", "birthday", or "friends" often becomes:

- `E2 gregariousness -> high`
- `E4 activity -> high`
- `E5 excitement-seek -> high`
- `E6 positive emotion -> high`

That is too much inference from too little evidence.

### 5. The model seems to default to `high` when uncertain

This is visible across multiple traits, not only `EXT`:

- `OPN`: predicted `high` in `48/50`
- `EXT`: predicted `high` in `44/50`
- `AGR`: predicted `high` in `41/50`
- `NEU`: predicted `high` in `47/50`

So the issue is not just one bad trait head. The broader pattern is:

- the reasoning prompt favors trait-present narratives
- the LLM is more comfortable justifying `high` than justifying `low`
- low labels require absence-based reasoning, which the current setup handles poorly

## Representative false low-`EXT` samples

### `idx 0`

Why prediction is likely wrong:

- mostly fragmented self-updates, obligations, fandom, frustration, and reflection
- little evidence of social dominance, warmth, or seeking interaction
- "excited" snippets are present, but they are episodic rather than a stable extraverted pattern

Likely failure:

- positive affect and event mentions were treated as Extraversion

### `idx 5`

Why prediction is likely wrong:

- strong aesthetic/mood language and playful style
- many solitary preferences: music, reflection, food, random thoughts, internal jokes
- expressive tone does not imply social orientation

Likely failure:

- verbosity and playfulness were mistaken for gregariousness

### `idx 8`

Why prediction is likely wrong:

- political argumentation, legal/policy commentary, and opinionated writing
- assertive stance is present, but social energy is not clearly present
- this looks more like ideology + intellect + conviction than Extraversion

Likely failure:

- forceful language was mistaken for extraverted social behavior

### `idx 14`

Why prediction is likely wrong:

- highly verbal and enthusiastic, but centered on music, performance, technical interest, and self-reflection
- lots of solo cognition and niche enthusiasm
- little direct evidence that the writer prefers social stimulation

Likely failure:

- enthusiasm + many activities were mapped to `high EXT` without checking whether the energy is social

### `idx 49`

Why prediction is likely wrong:

- sports chatter and strong opinions dominate
- emotional intensity is visible, but not necessarily sociability or interpersonal warmth
- could easily be a reserved person posting intense opinions online

Likely failure:

- the model equates loud tone with extraverted personality

## Why the pipeline is still not good overall

The core problem is not only raw accuracy. It is the failure pattern:

- predictions are systematically skewed toward `high`
- low labels are rarely recovered
- the model uses style/tone as a shortcut for personality
- retrieval calibration does not adequately protect against false-positive `high`
- concatenated multi-status inputs make the model overweight memorable snippets

That means the pipeline is not reliable for trait inference, especially when the true label depends on subtle absence-of-evidence reasoning.

## Most likely root causes

1. Retrieval similarity is probably based on semantic content, not trait-discriminative evidence.
2. The prompt lets positive affect stand in for Extraversion.
3. The prompt forces facet completion even when evidence is weak.
4. Concatenated status texts contain mixed signals, and the model overweights flashy segments.
5. Low-trait recognition is harder because it often depends on missing cues, not explicit cues.

## What to change next

1. Add a rule for `EXT`: do not infer `high` from positive emotion, humor, or activity alone; require evidence of sociability, assertiveness, or social-energy seeking.
2. Add a rule for `low` labels: when evidence for the high pole is absent, explicitly consider whether the sample is better explained by reserve, introspection, or solitary focus.
3. Retrieve examples by trait-discriminative cues, not just embedding similarity over the whole text.
4. Score per-status snippet first, then aggregate, instead of reasoning over one long concatenated block.
5. Make the model cite counter-evidence before finalizing `high`, especially for `EXT`.
6. Track per-trait calibration separately; `EXT` needs targeted prompt and retrieval tuning, not just global changes.

## Bottom line

The current run is overpredicting `high`, and `EXT` is the worst case. The pipeline is reading expressive, excited, or opinionated text as extraverted text, which collapses the distinction between "socially outgoing" and merely "energetic on the page."

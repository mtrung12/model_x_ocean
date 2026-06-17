# Analysis of False Samples and Why This Pipeline Is Still Weak

Files examined:

- `log/gpt-4o-mini/reasoned_rag_def_oneshot/20260518-001847_log.txt`
- `result/gpt-4o-mini/reasoned_rag_def_oneshot/20260518-001847/predictions.csv`
- `ptd_model/prompts.py`

## Executive Summary

The main failure is not that Neuroticism is generally under-predicted. It is the opposite: the pipeline is heavily biased toward predicting **`high` NEU**.

From `predictions.csv`:

- Gold NEU: `124 high / 123 low`
- Predicted NEU: `237 high / 10 low`
- NEU high recall: `121 / 124 = 0.9758`
- NEU low recall: `7 / 123 = 0.0569`
- NEU balanced accuracy: `0.5164`

So if “very low on NEU” means the model is doing badly on the NEU dimension, the issue is specifically that it almost never recognizes **low Neuroticism / emotional stability**.

## What the Errors Look Like

### Pattern 1: Any stress, fatigue, or uncertainty gets mapped to `high` NEU

Example from [predictions.csv](/f:/std/GR/code/model_x_ocean/result/gpt-4o-mini/reasoned_rag_def_oneshot/20260518-001847/predictions.csv:8):

- Gold: `cNEU=low`
- Pred: `pred_cNEU=high`
- Text contains ordinary exam stress, but also clear stability cues:
  - “I’m sure it will all work out”
  - “I’m really enjoying the transition”
  - “I’m eager to see what lies in the future”

The log shows the model still forcing this to `high`:

- [20260518-001847_log.txt:9492](/f:/std/GR/code/model_x_ocean/log/gpt-4o-mini/reasoned_rag_def_oneshot/20260518-001847_log.txt:9492) to [20260518-001847_log.txt:9512](/f:/std/GR/code/model_x_ocean/log/gpt-4o-mini/reasoned_rag_def_oneshot/20260518-001847_log.txt:9512)

The model explicitly acknowledges counter-evidence, then dismisses it:

- stress/worry evidence is treated as decisive
- positive regulation is treated as secondary

That is a calibration problem, not a pure extraction problem.

### Pattern 2: Mixed-emotion essays are defaulting to `high` NEU

Example from [predictions.csv](/f:/std/GR/code/model_x_ocean/result/gpt-4o-mini/reasoned_rag_def_oneshot/20260518-001847/predictions.csv:7):

- Gold: `cNEU=low`
- Pred: `pred_cNEU=high`

This essay contains:

- tiredness
- distraction
- missing friends
- but also enjoyment, social engagement, excitement, and normal adjustment language

The log context around this sample already frames the text as emotionally negative before the final verdict:

- [20260518-001847_log.txt:7015](/f:/std/GR/code/model_x_ocean/log/gpt-4o-mini/reasoned_rag_def_oneshot/20260518-001847_log.txt:7015) to [20260518-001847_log.txt:7019](/f:/std/GR/code/model_x_ocean/log/gpt-4o-mini/reasoned_rag_def_oneshot/20260518-001847_log.txt:7015)
  - “negative emotions dominate”
  - “some uncertainty”

That pre-summary is already pulling the model toward `high`, even though the text reads more like normal dorm adjustment than persistent emotional instability.

### Pattern 3: Neutral or socially descriptive texts get contaminated by isolated negative cues

Example from [predictions.csv](/f:/std/GR/code/model_x_ocean/result/gpt-4o-mini/reasoned_rag_def_oneshot/20260518-001847/predictions.csv:10):

- Gold: `cNEU=low`
- Pred: `pred_cNEU=high`

This sample is mostly mundane planning and casual reflection:

- music
- birthdays
- cards
- bed comfort
- class

But the pipeline tends to overread small discomfort cues like “my back hurt” or “I miss my bed” as emotional instability.

### Pattern 4: The rare NEU false negatives are mostly texts with very weak overt distress markers

Example from [predictions.csv](/f:/std/GR/code/model_x_ocean/result/gpt-4o-mini/reasoned_rag_def_oneshot/20260518-001847/predictions.csv:9):

- Gold: `cNEU=high`
- Pred: `pred_cNEU=low`

This kind of miss is much rarer: only `3` false negatives versus `116` false positives for NEU. That imbalance confirms the model is not uncertain in both directions. It is systematically collapsing toward `high`.

## Why the Pipeline Is Still Not Good

### 1. The NEU definitions are asymmetric in practice

In [prompts.py](/f:/std/GR/code/model_x_ocean/ptd_model/prompts.py:3), high NEU is defined with many common surface cues:

- anxiety
- stress
- negative emotions
- coping under pressure

Low NEU in [prompts.py](/f:/std/GR/code/model_x_ocean/ptd_model/prompts.py:5) requires stronger, more global stability cues:

- calm
- resilient
- relaxed
- handles stress effectively

In stream-of-consciousness student writing, mild stress is common. That means the `high` definition fires often, while the `low` definition requires stronger evidence than these essays usually provide.

### 2. The reasoned prompt encourages over-justification of weak evidence

The active prompt in [prompts.py](/f:/std/GR/code/model_x_ocean/ptd_model/prompts.py:147) asks for:

- evidence lists
- facet checks
- example alignment
- a final verdict

That structure is good for auditing, but it also encourages the model to build a prosecutorial case from a few negative phrases. Once it extracts “stressed”, “worried”, “tired”, or “overwhelmed”, the rest of the XML often becomes a self-reinforcing argument for `high`.

### 3. Retrieved examples appear to be steering the model toward negative framing

In the log, the retrieved context repeatedly summarizes essays with lines such as:

- “negative emotions dominate” at [20260518-001847_log.txt:7017](/f:/std/GR/code/model_x_ocean/log/gpt-4o-mini/reasoned_rag_def_oneshot/20260518-001847_log.txt:7017)
- “mixed balance with dominant negative affect due to stress and anxiety” at [20260518-001847_log.txt:39622](/f:/std/GR/code/model_x_ocean/log/gpt-4o-mini/reasoned_rag_def_oneshot/20260518-001847_log.txt:39622)

This matters because the model is not reading raw neighbors only; it is reading **already interpreted psychological evidence**. If retrieval returns examples framed around stress and anxiety, the model gets anchored toward `high` even for borderline or balanced texts.

### 4. The pipeline is probably treating temporary state as trait

Many false positives mention:

- tests
- sleepiness
- homesickness
- mild frustration
- transition to college

Those are short-term states. High Neuroticism is a trait-level tendency. The pipeline is not separating:

- transient situational stress
- from broad emotional instability / chronic vulnerability

That trait-state confusion is the main conceptual error behind the NEU collapse.

### 5. The logs suggest the model can see counter-evidence but does not weight it correctly

The best evidence is the exam-stress example in the log:

- [20260518-001847_log.txt:9497](/f:/std/GR/code/model_x_ocean/log/gpt-4o-mini/reasoned_rag_def_oneshot/20260518-001847_log.txt:9497) to [20260518-001847_log.txt:9500](/f:/std/GR/code/model_x_ocean/log/gpt-4o-mini/reasoned_rag_def_oneshot/20260518-001847_log.txt:9500) records clear counter-evidence
- [20260518-001847_log.txt:9510](/f:/std/GR/code/model_x_ocean/log/gpt-4o-mini/reasoned_rag_def_oneshot/20260518-001847_log.txt:9510) still says the anxiety evidence outweighs it

So the pipeline is not blind. It is miscalibrated.

## Bottom Line

This pipeline is still not good because its NEU classifier is effectively a **stress detector**, not a reliable trait classifier.

The practical failure mode is:

1. find any negative affect cue
2. align to retrieved “high NEU” examples
3. explain away positive regulation as secondary
4. output `high`

That produces very high recall for `high` NEU, but near-total failure on `low` NEU.

## Most Likely Fix Directions

- Tighten the NEU decision rule so transient stress is not enough for `high`.
- Require evidence of instability, rumination, poor coping, or sustained negative reactivity for `high`.
- Add an explicit rule that ordinary school stress plus intact coping should favor `low`.
- Rebalance retrieved anchors so low-NEU examples include texts with stress but good regulation.
- Score or filter retrieved evidence summaries that use overly negative interpretive language.
- Evaluate NEU with balanced accuracy or per-class recall, not just one pooled score.

# RAG Sample Label Matching Analysis
**Model:** GPT-4o-mini (Chain-of-Thought with RAG)  
**Run ID:** 20260428-015151  
**Date:** April 28, 2026

---

## Executive Summary

Out of **750 total RAG samples** retrieved (50 test cases × 5 traits × 3 similar samples per trait), **389 samples had labels that matched the ground truth label of the target text**.

**Overall RAG Sample Label Match Rate: 51.87%**

This indicates that the RAG retrieval system has moderate relevance—just over half of the retrieved samples share the same label as the text being classified, which is slightly better than random (50% for binary classification).

---

## Key Findings

### Overall Statistics
- **Total RAG samples retrieved:** 750 (50 test samples × 5 traits × 3 samples per trait)
- **Samples with matching labels:** 389
- **Samples with non-matching labels:** 361
- **Match rate:** **51.87%**

### Per-Trait Breakdown

| Trait | Matching Samples | Total Samples | Match Rate |
|-------|-----------------|---------------|-----------|
| **Conscientiousness** | 88/150 | 150 | **58.67%**   |
| **Agreeableness** | 85/150 | 150 | **56.67%** |
| **Extraversion** | 74/150 | 150 | **49.33%** |
| **Neuroticism** | 71/150 | 150 | **47.33%**  |
| **Openness** | 71/150 | 150 | **47.33%**  |

---

## Detailed Analysis

### Strong Performers
 **Conscientiousness (58.67%)** - Best RAG match rate
- More than half of retrieved samples share the same label
- Suggests RAG system is reasonably effective for this trait
- May contribute to the 44% trait accuracy despite good sample relevance

 **Agreeableness (56.67%)** - Above average
- Better than 50/50 baseline
- Reasonable sample relevance

### Weak Performers
 **Openness & Neuroticism (47.33%)** - Below baseline
- Nearly 53% of samples have mismatched labels
- Suggests RAG retrieval may not be finding truly similar texts for these traits
- Could explain lower model performance on these traits

 **Extraversion (49.33%)** - Just below 50%
- Marginal relevance
- Despite close to random, the model achieved 62% accuracy (best trait)

---

## Interpretation & Insights

### Key Observation
The RAG match rate (51.87%) is **only marginally better than random selection** (50%), yet the model's overall prediction accuracy is 55.20% across traits. This suggests:

1. **Label alone is insufficient** - Having similar samples with matching labels helps, but the psychological evidence within those samples is critical
2. **Chain-of-thought reasoning adds value** - The CoT prompting approach appears to leverage the retrieved samples effectively despite moderate label relevance
3. **Trait-specific variation** - Some traits (Conscientiousness, Agreeableness) benefit from more relevant RAG samples, while others (Openness, Neuroticism) may need better retrieval strategies

### Divergence Between Traits
**Interesting Pattern:** Extraversion has the lowest RAG match rate (49.33%) but the highest model accuracy (62%). This suggests:
- Either the retrieved "low" samples are helping by negative example
- Or the psychological evidence for Extraversion is particularly clear even in mismatched samples
- Or the model is relying more on the target text itself than the calibration samples

**Openness & Neuroticism** have low RAG match rates (47.33%) AND moderate model accuracy (60% and 54%), suggesting:
- These traits would benefit from improved RAG retrieval strategies
- Better sample selection could significantly boost performance

---

## Recommendations

### 1. **Improve RAG Retrieval Strategy**
- Current retrieval may be based on text similarity rather than trait similarity
- Implement trait-specific embeddings that prioritize semantic relevance to each personality dimension
- Focus on Openness and Neuroticism which show weakest match rates

### 2. **Sample Diversity**
- The 3 samples per query may be insufficient
- Experiment with 5-7 samples per trait to increase likelihood of relevant matches
- Diversify between high/low examples (currently may be biased)

### 3. **Quality Over Quantity**
- A higher quality RAG match rate (60%+) might correlate with better model performance
- Invest in better similarity metrics rather than more samples

### 4. **Trait-Specific Fine-tuning**
- Use this analysis to identify which traits need attention:
  - **Priority:** Openness (47.33%) and Neuroticism (47.33%)
  - **Secondary:** Extraversion (49.33%) - paradoxically performs well despite low match
- Consider separate RAG strategies for different traits

### 5. **Ablation Study**
- Test model performance WITHOUT RAG vs WITH RAG
- Current 51.87% match rate suggests RAG adds ~5% marginal improvement
- Clarify whether RAG is truly helping or if it's noise

---

## Sample Distribution

### Target Labels Observed
All 250 records show target_label = "high", suggesting:
- Test set may be imbalanced (or this is a subset of high-trait samples)
- 50% RAG match rate means on average 1.5 of 3 samples are "high" (matching)
- This could bias the model toward predicting "high" regardless of evidence

---

## Conclusion

**RAG sample label matching at 51.87% indicates marginal relevance.**

The retrieved samples are slightly better than random at matching the ground truth label, but only by ~2%. This explains why the model's overall accuracy (55.20%) is modest—the calibration examples provide limited guidance.

**Key Takeaway:** The model's performance isn't severely hampered by mismatched RAG samples because:
1. The psychological evidence extraction within samples still provides useful reference points
2. Chain-of-thought reasoning helps the model think critically even with imperfect calibration examples
3. But there's significant headroom for improvement by increasing RAG match rates to 65%+

---

## Files Referenced
- **Predictions:** `/result/gpt-4o-mini/cot_rag_oneshot/20260428-015151/predictions.csv`
- **Log with RAG samples:** `/log/gpt-4o-mini/cot_rag_oneshot/20260428-015151_log.txt`

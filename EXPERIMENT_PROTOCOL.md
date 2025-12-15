# Recursive Depth Probe: Main Experiment Protocol

## Version
1.0 — December 2025

## Overview
Multi-model test of the 3c recursive composition hypothesis from the Abstraction Primitive Hypothesis (APH) framework.

## Hypotheses

### H1: Threshold Collapse
All models show accuracy drop >30 percentage points at some critical depth d*, with post-collapse accuracy statistically indistinguishable from chance.

**Operationalization:** 
- "Collapse" = accuracy decrease >30pp between consecutive depths
- "Chance" = 1/k where k = number of nouns in sentence (~17% for depth 3+)

### H2: Cross-Model Consistency  
Critical depth d* is similar (±1 level) across architecturally distinct models.

**Rationale:** If collapse reflects fundamental limitation of pattern-matching approaches, different implementations should fail at similar depths.

### H3: Systematic Error Structure
Post-collapse errors favor recency (sentence-final object noun) above chance.

**Operationalization:** Among errors, proportion selecting object noun > 1/k.

## Design

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Depths | 1-6 | Spans pre-collapse to clear failure |
| Trials/cell | 30 | 95% power for 30pp effect |
| Models | 3 | Cross-architecture comparison |
| Total N | 540 | 3 × 6 × 30 |

## Models

1. **claude-3-5-haiku-20241022** (Anthropic) — Transformer, ~20B params estimated
2. **gpt-4o-mini** (OpenAI) — Transformer, size undisclosed  
3. **gemini-2.0-flash** (Google) — Transformer/MoE, size undisclosed

## Stimuli

Center-embedded relative clauses following Gibson (1998):
- Depth 1: "The X V1 the Y."
- Depth 2: "The X [that the H1 V2] V1 the Y."
- Depth 3: "The X [that the H1 [that the H2 V3] V2] V1 the Y."
- etc.

### Lexical Items
- Animals (subjects/objects): dog, cat, bird, horse, rabbit, fox, wolf, bear, deer, mouse
- Humans (embedders): man, woman, boy, girl, teacher, doctor, chef, artist, farmer, writer
- Main verbs: chased, watched, followed, approached, startled
- Embedding verbs: saw, knew, met, helped, called

### Controls
- No lexical repetition within sentence
- Balanced across lexical items
- Same 180 stimuli presented to all models

## Procedure

1. Generate 180 unique stimuli (30 per depth)
2. Randomize presentation order (fixed across models for comparability)
3. Present each stimulus with prompt: "In the following sentence, who performed the main action? Sentence: '[X]' Answer with just the single word (the noun) that performed the main action. Do not explain."
4. Record: response, latency, token counts
5. Score: correct/incorrect, error type if incorrect

## Analysis Plan

### Primary Analysis
Mixed-effects logistic regression:
```
accuracy ~ depth * model + (1|stimulus)
```
- Fixed effects: depth (continuous or categorical), model, interaction
- Random effect: stimulus (accounts for item-level variance)

### Collapse Detection
Segmented regression to identify breakpoint:
```
accuracy ~ depth (segment 1) + depth (segment 2)
```
Compare AIC/BIC to linear model.

### Error Analysis
Among incorrect responses:
- Proportion selecting object noun (recency)
- Proportion selecting embedded human
- Proportion selecting other

Chi-square test: observed vs. uniform distribution.

### Multiple Comparisons
Bonferroni correction for 3 model comparisons: α = 0.05/3 = 0.017

## Expected Results

### If APH prediction correct:
- All models: ~90%+ accuracy depths 1-2, <30% depths 4+
- Collapse at depth 2-3 or 3-4
- Recency bias in errors

### If null hypothesis:
- Linear decline (no threshold)
- OR sustained high accuracy through depth 6
- Random error distribution

## References

Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Routledge.

Gibson, E. (1998). Linguistic complexity: Locality of syntactic dependencies. Cognition, 68(1), 1-76.

Lake, B., & Baroni, M. (2018). Generalization without systematicity. ICML.
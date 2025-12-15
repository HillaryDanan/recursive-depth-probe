# Recursive Depth Probe

**Testing the Abstraction Primitive Hypothesis (APH) through recursive linguistic structure**

## Status: Concluded — Returning to Drawing Board

This repository documents a series of experiments testing whether LLMs show "threshold collapse" on recursive linguistic structures, as predicted by the APH framework. While we obtained clean, interpretable data, the results reveal fundamental limitations in using linguistic tests to distinguish "pattern matching" from "structural construction."

---

## Background

### The Abstraction Primitive Hypothesis (APH)

The APH framework (Danan, 2025) proposes that intelligence emerges from recursive interaction between symbol formation and compositional structure. It distinguishes:

| Composition Type | Structure | Mechanism |
|------------------|-----------|-----------|
| **3a-3b** | Concatenative, Role-filler | Pattern matching (bounded space) |
| **3c-3d** | Recursive, Analogical | Construction (unbounded space) |

**Core prediction:** Systems without "embeddedness" (persistent self, stakes, feedback closure) should show **threshold collapse** on 3c tasks — sudden failure at some depth, not graceful degradation.

### Our Goal

Test whether LLMs show threshold collapse on center-embedded sentences, the canonical test of recursive linguistic processing (Gibson, 1998).

---

## Experiments

### Experiment 1: Pilot Study

**Design:** Center-embedded sentences, depths 1-6, n=10 per depth, Haiku only

**Stimuli:**
- D1: "The dog chased the cat."
- D3: "The dog that the man that the woman knew saw chased the cat."

**Question:** "Who performed the main action?"

**Results:**

| Depth | Haiku |
|-------|-------|
| 1 | 100% |
| 2 | 80% |
| 3 | 20% |
| 4 | 10% |
| 5 | 20% |
| 6 | 10% |

**Finding:** Clear threshold collapse at depth 3. Consistent with APH prediction.

---

### Experiment 2: Multi-Model Comparison

**Design:** Depths 1-6, n=30 per depth, Haiku + GPT-4o-mini

**Results:**

| Depth | Haiku | GPT-4o-mini |
|-------|-------|-------------|
| 1 | 100% | 100% |
| 2 | 73% | 100% |
| 3 | 20% | 100% |
| 4 | 10% | 100% |
| 5 | 7% | 97% |
| 6 | 23% | 97% |

**Finding:** Dramatic model dissociation. Haiku collapses; GPT-4o-mini doesn't.

**Complication:** This challenges the "all LLMs collapse" prediction. GPT's success could be:
1. Better training coverage
2. Architectural differences
3. Actual structural parsing capacity

---

### Experiment 3: Extended Depth (1-12)

**Design:** Depths 1-12, n=30 per depth, Haiku + Sonnet + GPT-4o-mini

**Results:**

| Depth | Haiku | GPT-4o-mini | Sonnet |
|-------|-------|-------------|--------|
| 1 | 100% | 100% | 100% |
| 2 | 67% | 100% | 100% |
| 3 | 23% | 100% | 100% |
| 4 | 7% | 100% | 100% |
| 5 | 20% | 93% | 97% |
| 6 | 20% | 90% | 100% |
| 7 | 17% | 80% | 100% |
| 8 | 17% | 73% | 97% |
| 9 | 10% | 57% | 97% |
| 10 | 13% | 60% | 100% |
| 11 | 20% | 47% | 100% |
| 12 | 57%* | 47% | 93% |

*Anomalous — likely noise

**Key findings:**

1. **Curve fitting:** All models fit sigmoid (threshold collapse) better than linear, but at different thresholds
   - Haiku: midpoint ~2
   - GPT-4o-mini: midpoint ~10
   - Sonnet: midpoint >12 (not reached)

2. **Cross-model agreement:** Near-zero Cohen's Kappa — models fail on different items, suggesting idiosyncratic coverage gaps rather than shared structural difficulty

3. **Response time:** Flat across depths for all models — no evidence of depth-sensitive processing

---

### Experiment 4: Structural Transfer

**Hypothesis:** If models have genuine recursive capacity (not domain-specific coverage), performance should transfer across structurally isomorphic domains.

**Design:** Three domains, depths 1-6, n=20 per cell
- **A:** Relative clauses ("The dog that the man saw chased the cat")
- **B:** Possessive chains ("The teacher's doctor's lawyer's car is expensive")  
- **C:** Prepositional chains ("The letter from the lawyer from the doctor arrived")

**Results (Depth 3+ accuracy):**

| Model | Domain A | Domain B | Domain C | Transfer? |
|-------|----------|----------|----------|-----------|
| Haiku | 36% | 96% | 64% | NO |
| GPT-4o-mini | 98% | 100% | 100% | YES |
| Sonnet | 100% | 99% | 100% | YES |

**Finding:** Haiku shows domain-specific failure (center-embedding specifically hard). GPT and Sonnet show transfer — but this could still be broad coverage rather than construction.

---

### Experiment 5: Binding Recovery

**Hypothesis:** If models parse structure, they should recover ALL bindings in a sentence, not just the main clause.

**Design:** Same sentences, multiple questions probing different bindings

**Results:** Unexpectedly flat — models showed similar accuracy on main vs. embedded bindings. But deeper analysis revealed a confound: **center-embedding creates locality at deep levels** (innermost embeddings have adjacent subject-verb), making deep bindings potentially EASIER than shallow ones.

**Conclusion:** Test design was flawed.

---

### Experiment 6: Subject vs. Object Relatives

**Hypothesis:** Object relatives break first-noun heuristics; subject relatives don't.

**Design:**
- Subject RC: "The cat that chased the dog ran away" (first noun = agent)
- Object RC: "The cat that the dog chased ran away" (first noun ≠ agent)

**Results:** All models performed near-ceiling on both conditions.

**Conclusion:** Either models parse structure, or training coverage includes sufficient object relative examples. The test cannot distinguish these.

---

## Deep Analysis: Curve Fitting

Formal model comparison using BIC:

| Model | Better Fit | Midpoint | ΔBIC |
|-------|------------|----------|------|
| Haiku | Sigmoid | 1.7 | 122.5 |
| GPT-4o-mini | Sigmoid | 10.3 | 298.7 |
| Sonnet | Linear (flat) | N/A | 1039.3 |

**Interpretation:** All models that fail show sigmoid (threshold collapse) pattern — they just collapse at different depths. Sonnet doesn't collapse through depth 12.

---

## Error Analysis

### Haiku Error Patterns

| Depth | Object Errors | Human Errors |
|-------|---------------|--------------|
| D2-3 | 90-100% | 0-10% |
| D4 | 68% | 32% |
| D5+ | 12-38% | 62-88% |

**Two distinct failure modes:**
1. **Shallow (D2-3):** Recency heuristic — picks noun nearest to end
2. **Deep (D4+):** Embedding confusion — picks wrong agent from structure

---

## What We Learned

### Confirmed

1. **Models differ dramatically** in recursive depth capacity
2. **Threshold collapse** (sigmoid pattern) fits failing models better than gradual decline
3. **Failures are model-specific**, not item-specific (near-zero cross-model agreement)
4. **Error patterns are systematic**, suggesting heuristic use rather than random failure

### Not Resolved

1. **Construction vs. coverage:** Ceiling performance cannot distinguish genuine structural parsing from massive training coverage
2. **What explains model differences:** Training data? Architecture? Scale? Unknown
3. **Whether any model "constructs":** Every test we designed could potentially be passed by pattern matching with sufficient coverage

---

## The Fundamental Limitation

Every linguistic test we designed had:
- Clear right/wrong answers
- Structured, predictable formats
- Patterns that exist in training corpora

A sufficiently trained pattern matcher could ace all of them.

**The problem:** We cannot design a linguistic test that definitively distinguishes "learned the pattern" from "understands the structure" — because any structure we test is a pattern that could have been learned.

**Subjective observation:** Despite passing increasingly sophisticated tests, the experience of interacting with these models still "feels like" pattern matching — quick confident answers, missing deeper intent, generic solutions. This subjective signal may be detecting something the formal tests cannot capture.

---

## Returning to Drawing Board

This line of investigation has reached diminishing returns. Future work might explore:

1. **Non-linguistic tests** of recursive/compositional capacity
2. **Process measures** (not just accuracy) — what does internal computation look like?
3. **Adversarial tests** designed to break specific heuristics
4. **Novel structures** with no training distribution coverage
5. **ARC-AGI style tasks** that require genuine abstraction

The APH framework's predictions about recursive composition remain plausible, but linguistic center-embedding may not be the right operationalization for testing them with modern LLMs.

---

## Repository Structure
```
recursive-depth-probe/
├── README.md
├── EXPERIMENT_PROTOCOL.md
├── requirements.txt
├── config.py
├── .env.example
├── .gitignore
│
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── anthropic_client.py
│   │   └── openai_client.py
│   │
│   ├── generate_stimuli.py
│   ├── generate_extended_stimuli.py
│   ├── generate_transfer_stimuli.py
│   ├── generate_binding_stimuli.py
│   ├── generate_relative_stimuli.py
│   │
│   ├── run_experiment.py
│   ├── run_main_experiment.py
│   ├── run_extended_experiment.py
│   ├── run_transfer_experiment.py
│   ├── run_binding_experiment.py
│   ├── run_relative_experiment.py
│   │
│   ├── analyze_results.py
│   ├── analyze_main_results.py
│   ├── analyze_extended_results.py
│   ├── analyze_extended_deep.py
│   ├── analyze_transfer_results.py
│   ├── analyze_binding_results.py
│   └── analyze_relative_results.py
│
├── data/
│   ├── stimuli.json
│   ├── main_stimuli.json
│   ├── extended_stimuli.json
│   ├── transfer/
│   ├── binding/
│   └── relatives/
│
└── results/
    ├── pilot/
    ├── main/
    ├── extended/
    ├── transfer/
    ├── binding/
    └── relatives/
```

---

## Usage
```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add API keys

# Run any experiment
python3 src/generate_[experiment]_stimuli.py
python3 src/run_[experiment]_experiment.py
python3 src/analyze_[experiment]_results.py
```

---

## References

Gibson, E. (1998). Linguistic complexity: Locality of syntactic dependencies. *Cognition*, 68(1), 1-76.

Grodner, D., & Gibson, E. (2005). Consequences of the serial nature of linguistic input for sentential complexity. *Cognitive Science*, 29(2), 261-290.

Lake, B., & Baroni, M. (2018). Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. *ICML*.

Fodor, J. A., & Pylyshyn, Z. W. (1988). Connectionism and cognitive architecture: A critical analysis. *Cognition*, 28(1-2), 3-71.

---

## Author

**Hillary Danan, PhD** — Cognitive Neuroscience

Part of the [Abstraction-Intelligence](https://github.com/HillaryDanan/abstraction-intelligence) research program.

---

## License

MIT

---

*"The map is not the territory. The test is not the capacity."*

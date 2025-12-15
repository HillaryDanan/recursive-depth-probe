"""
Generate Subject vs Object Relative Clause stimuli.

This is the cleanest test of structural parsing vs pattern matching.

Subject relative: "The cat that chased the dog ran away."
  - Agent of "chased" = cat (first noun)
  - First-noun heuristic WORKS

Object relative: "The cat that the dog chased ran away."
  - Agent of "chased" = dog (NOT first noun)
  - First-noun heuristic FAILS
  - Recency heuristic FAILS
  - Only structural parsing works

References:
- Gibson, E. (1998). Linguistic complexity: Locality of syntactic dependencies.
- Grodner, D., & Gibson, E. (2005). Consequences of the serial nature of linguistic input.
"""

import json
import random
from pathlib import Path
from config import DATA_DIR


# =============================================================================
# PARAMETERS
# =============================================================================

TRIALS_PER_CONDITION = 30  # 30 subject, 30 object at each depth
DEPTHS = [1, 2, 3]  # 1 = single RC, 2 = nested, 3 = double nested


# =============================================================================
# LEXICAL ITEMS
# =============================================================================

ANIMATE_NOUNS = [
    "cat", "dog", "bird", "horse", "rabbit", "fox", "wolf", "bear", "deer", "mouse",
    "lion", "tiger", "eagle", "owl", "snake", "monkey", "elephant", "giraffe", "zebra", "dolphin"
]

TRANSITIVE_VERBS = [
    "chased", "followed", "watched", "pushed", "pulled", "carried", "lifted",
    "touched", "grabbed", "bumped", "startled", "frightened", "calmed", "amused"
]

INTRANSITIVE_VERBS = [
    "ran away", "fell down", "jumped up", "stood still", "walked away",
    "turned around", "sat down", "woke up", "calmed down", "backed away"
]


# =============================================================================
# STIMULUS GENERATION
# =============================================================================

def generate_depth1_subject(seed=None):
    """
    Generate depth-1 subject relative.
    
    Structure: "The X that verbed the Y did Z."
    Example: "The cat that chased the dog ran away."
    
    Agent of embedded verb = X (first noun)
    """
    if seed is not None:
        random.seed(seed)
    
    nouns = random.sample(ANIMATE_NOUNS, 2)
    head = nouns[0]
    embedded_obj = nouns[1]
    
    embedded_verb = random.choice(TRANSITIVE_VERBS)
    main_verb = random.choice(INTRANSITIVE_VERBS)
    
    sentence = f"The {head} that {embedded_verb} the {embedded_obj} {main_verb}."
    
    return {
        "sentence": sentence,
        "structure": "subject_relative",
        "depth": 1,
        "embedded_verb": embedded_verb,
        "question": f"Who {embedded_verb} the {embedded_obj}?",
        "correct_answer": head,
        "first_noun": head,
        "second_noun": embedded_obj,
        "heuristic_answer": head,  # First-noun heuristic gives this
        "heuristic_correct": True
    }


def generate_depth1_object(seed=None):
    """
    Generate depth-1 object relative.
    
    Structure: "The X that the Y verbed did Z."
    Example: "The cat that the dog chased ran away."
    
    Agent of embedded verb = Y (NOT first noun)
    """
    if seed is not None:
        random.seed(seed)
    
    nouns = random.sample(ANIMATE_NOUNS, 2)
    head = nouns[0]  # Patient of embedded verb
    embedded_subj = nouns[1]  # Agent of embedded verb
    
    embedded_verb = random.choice(TRANSITIVE_VERBS)
    main_verb = random.choice(INTRANSITIVE_VERBS)
    
    sentence = f"The {head} that the {embedded_subj} {embedded_verb} {main_verb}."
    
    return {
        "sentence": sentence,
        "structure": "object_relative",
        "depth": 1,
        "embedded_verb": embedded_verb,
        "question": f"Who {embedded_verb} the {head}?",
        "correct_answer": embedded_subj,
        "first_noun": head,
        "second_noun": embedded_subj,
        "heuristic_answer": head,  # First-noun heuristic gives this (WRONG)
        "heuristic_correct": False
    }


def generate_depth2_subject(seed=None):
    """
    Generate depth-2 subject relative (nested).
    
    Structure: "The X that verbed1 the Y that verbed2 the Z did W."
    Example: "The cat that chased the dog that bit the bird ran away."
    
    Questions about both verbs - both have first-noun-of-clause as agent.
    """
    if seed is not None:
        random.seed(seed)
    
    nouns = random.sample(ANIMATE_NOUNS, 3)
    n1, n2, n3 = nouns[0], nouns[1], nouns[2]
    
    v1 = random.choice(TRANSITIVE_VERBS)
    v2 = random.choice([v for v in TRANSITIVE_VERBS if v != v1])
    main_verb = random.choice(INTRANSITIVE_VERBS)
    
    # "The cat that chased the dog that bit the bird ran away"
    # cat chased dog, dog bit bird, cat ran away
    sentence = f"The {n1} that {v1} the {n2} that {v2} the {n3} {main_verb}."
    
    return {
        "sentence": sentence,
        "structure": "subject_relative",
        "depth": 2,
        "embedded_verb": v1,  # Ask about outer embedding
        "question": f"Who {v1} the {n2}?",
        "correct_answer": n1,
        "first_noun": n1,
        "second_noun": n2,
        "third_noun": n3,
        "heuristic_answer": n1,
        "heuristic_correct": True
    }


def generate_depth2_object(seed=None):
    """
    Generate depth-2 object relative (nested).
    
    Structure: "The X that the Y that the Z verbed2 verbed1 did W."
    Example: "The cat that the dog that the bird saw chased ran away."
    
    Parse: The cat [that the dog [that the bird saw] chased] ran away
    - bird saw dog
    - dog chased cat
    - cat ran away
    """
    if seed is not None:
        random.seed(seed)
    
    nouns = random.sample(ANIMATE_NOUNS, 3)
    n1, n2, n3 = nouns[0], nouns[1], nouns[2]  # cat, dog, bird
    
    v1 = random.choice(TRANSITIVE_VERBS)  # chased (outer)
    v2 = random.choice([v for v in TRANSITIVE_VERBS if v != v1])  # saw (inner)
    main_verb = random.choice(INTRANSITIVE_VERBS)
    
    # "The cat that the dog that the bird saw chased ran away"
    # bird saw dog, dog chased cat
    sentence = f"The {n1} that the {n2} that the {n3} {v2} {v1} {main_verb}."
    
    return {
        "sentence": sentence,
        "structure": "object_relative",
        "depth": 2,
        "embedded_verb": v1,  # Ask about outer embedding: who chased?
        "question": f"Who {v1} the {n1}?",
        "correct_answer": n2,  # dog chased cat
        "first_noun": n1,
        "second_noun": n2,
        "third_noun": n3,
        "heuristic_answer": n1,  # First noun = wrong
        "heuristic_correct": False
    }


def generate_depth3_subject(seed=None):
    """Depth-3 subject relative."""
    if seed is not None:
        random.seed(seed)
    
    nouns = random.sample(ANIMATE_NOUNS, 4)
    n1, n2, n3, n4 = nouns
    
    verbs = random.sample(TRANSITIVE_VERBS, 3)
    v1, v2, v3 = verbs
    main_verb = random.choice(INTRANSITIVE_VERBS)
    
    # "The cat that chased the dog that bit the bird that pecked the mouse ran away"
    sentence = f"The {n1} that {v1} the {n2} that {v2} the {n3} that {v3} the {n4} {main_verb}."
    
    return {
        "sentence": sentence,
        "structure": "subject_relative",
        "depth": 3,
        "embedded_verb": v1,
        "question": f"Who {v1} the {n2}?",
        "correct_answer": n1,
        "first_noun": n1,
        "heuristic_answer": n1,
        "heuristic_correct": True
    }


def generate_depth3_object(seed=None):
    """Depth-3 object relative."""
    if seed is not None:
        random.seed(seed)
    
    nouns = random.sample(ANIMATE_NOUNS, 4)
    n1, n2, n3, n4 = nouns
    
    verbs = random.sample(TRANSITIVE_VERBS, 3)
    v1, v2, v3 = verbs
    main_verb = random.choice(INTRANSITIVE_VERBS)
    
    # "The cat that the dog that the bird that the mouse saw chased bit ran away"
    # mouse saw bird, bird chased dog, dog bit cat
    sentence = f"The {n1} that the {n2} that the {n3} that the {n4} {v3} {v2} {v1} {main_verb}."
    
    return {
        "sentence": sentence,
        "structure": "object_relative",
        "depth": 3,
        "embedded_verb": v1,  # outermost embedded verb
        "question": f"Who {v1} the {n1}?",
        "correct_answer": n2,
        "first_noun": n1,
        "heuristic_answer": n1,
        "heuristic_correct": False
    }


def generate_all_stimuli(trials_per_condition, seed_offset=0):
    """Generate all stimuli."""
    stimuli = []
    
    generators = {
        (1, "subject"): generate_depth1_subject,
        (1, "object"): generate_depth1_object,
        (2, "subject"): generate_depth2_subject,
        (2, "object"): generate_depth2_object,
        (3, "subject"): generate_depth3_subject,
        (3, "object"): generate_depth3_object,
    }
    
    for (depth, rel_type), generator in generators.items():
        for trial in range(trials_per_condition):
            seed = seed_offset + depth * 10000 + (0 if rel_type == "subject" else 5000) + trial
            stimulus = generator(seed)
            stimulus["trial_id"] = f"d{depth}_{rel_type[:3]}_{trial:02d}"
            stimulus["seed"] = seed
            stimuli.append(stimulus)
    
    return stimuli


def main():
    """Generate and save stimuli."""
    output_dir = Path(DATA_DIR) / "relatives"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING SUBJECT VS OBJECT RELATIVE CLAUSE STIMULI")
    print("=" * 70)
    print(f"Trials per condition: {TRIALS_PER_CONDITION}")
    print(f"Depths: {DEPTHS}")
    print(f"Total: {TRIALS_PER_CONDITION * len(DEPTHS) * 2} stimuli")
    print()
    
    stimuli = generate_all_stimuli(TRIALS_PER_CONDITION, seed_offset=300000)
    
    # Save
    output_path = output_dir / "relative_stimuli.json"
    with open(output_path, "w") as f:
        json.dump(stimuli, f, indent=2)
    
    print(f"Saved to: {output_path}")
    
    # Examples
    print("\n" + "=" * 70)
    print("EXAMPLES")
    print("=" * 70)
    
    for depth in DEPTHS:
        print(f"\n--- Depth {depth} ---")
        
        subj = [s for s in stimuli if s["depth"] == depth and s["structure"] == "subject_relative"][0]
        obj = [s for s in stimuli if s["depth"] == depth and s["structure"] == "object_relative"][0]
        
        print(f"\nSUBJECT RELATIVE (heuristic works):")
        print(f"  Sentence: {subj['sentence']}")
        print(f"  Question: {subj['question']}")
        print(f"  Answer: {subj['correct_answer']}")
        print(f"  First-noun heuristic gives: {subj['heuristic_answer']} ({'✓' if subj['heuristic_correct'] else '✗'})")
        
        print(f"\nOBJECT RELATIVE (heuristic fails):")
        print(f"  Sentence: {obj['sentence']}")
        print(f"  Question: {obj['question']}")
        print(f"  Answer: {obj['correct_answer']}")
        print(f"  First-noun heuristic gives: {obj['heuristic_answer']} ({'✓' if obj['heuristic_correct'] else '✗'})")
    
    # Summary
    print("\n" + "=" * 70)
    print("KEY PREDICTION")
    print("=" * 70)
    print("""
    Pattern Matching:
      Subject relatives: HIGH (heuristics work)
      Object relatives: LOW (heuristics fail)
      
    Structural Parsing:
      Subject relatives: HIGH
      Object relatives: HIGH (slightly lower due to complexity, but above chance)
      
    Chance level: ~50% (2 nouns to choose from)
    """)


if __name__ == "__main__":
    main()
"""
Generate binding recovery stimuli.

For each center-embedded sentence, we generate multiple questions:
- One for the main clause binding
- One for each embedded clause binding

This tests whether models can recover ALL bindings, not just the main one.

Key insight: Heuristics succeed on main clause but fail on embedded bindings.
Structural parsing succeeds on all.
"""

import json
import random
from pathlib import Path
from config import DATA_DIR


# =============================================================================
# PARAMETERS
# =============================================================================

DEPTHS = [2, 3, 4, 5, 6]  # Skip depth 1 (no embeddings to test)
SENTENCES_PER_DEPTH = 15  # Fewer sentences, but multiple questions each


# =============================================================================
# LEXICAL ITEMS
# =============================================================================

SUBJECTS = ["dog", "cat", "bird", "horse", "rabbit", "fox", "wolf", "bear", "deer", "mouse"]
OBJECTS = ["ball", "toy", "bone", "stick", "rock", "leaf", "rope", "box", "bag", "cup"]
HUMANS = ["man", "woman", "boy", "girl", "teacher", "doctor", "chef", "artist", "farmer", "writer",
          "nurse", "lawyer", "pilot", "sailor", "clerk"]
MAIN_VERBS = ["grabbed", "found", "touched", "moved", "carried"]
EMBED_VERBS = ["saw", "noticed", "recognized", "watched", "observed", 
               "met", "visited", "called", "helped", "followed"]


# =============================================================================
# SENTENCE AND QUESTION GENERATION
# =============================================================================

def generate_sentence_with_bindings(depth, seed=None):
    """
    Generate a sentence and all its binding questions.
    
    For depth D, there are D bindings:
    - 1 main clause binding
    - D-1 embedded clause bindings
    
    Returns sentence + list of (question, answer, binding_type) tuples.
    """
    if seed is not None:
        random.seed(seed)
    
    # Select lexical items
    main_subject = random.choice(SUBJECTS)
    main_object = random.choice(OBJECTS)
    main_verb = random.choice(MAIN_VERBS)
    
    n_embeddings = depth - 1
    humans = random.sample(HUMANS, n_embeddings)
    embed_verbs = random.sample(EMBED_VERBS, n_embeddings)
    
    # Build sentence (same as before)
    embedded_part = ""
    for i in range(n_embeddings - 1, -1, -1):
        human = humans[i]
        verb = embed_verbs[i]
        if i == n_embeddings - 1:
            embedded_part = f"that the {human} {verb}"
        else:
            embedded_part = f"that the {human} {embedded_part} {verb}"
    
    sentence = f"The {main_subject} {embedded_part} {main_verb} the {main_object}."
    
    # Generate binding questions
    bindings = []
    
    # Main clause binding: Who [main_verb]ed the [main_object]? → main_subject
    main_q = f"Who {main_verb} the {main_object}?"
    bindings.append({
        "question": main_q,
        "answer": main_subject,
        "binding_type": "main",
        "binding_depth": 0,
        "verb": main_verb,
        "object_asked": main_object
    })
    
    # Embedded bindings
    # embed_verbs[0] + humans[0]: humans[0] [verb] main_subject
    # embed_verbs[i] + humans[i]: humans[i] [verb] humans[i-1] for i > 0
    
    for i in range(n_embeddings):
        human = humans[i]
        verb = embed_verbs[i]
        
        if i == 0:
            # First embedding: agent saw the main_subject
            obj = main_subject
        else:
            # Deeper embeddings: agent [verb]ed the previous human
            obj = humans[i - 1]
        
        q = f"Who {verb} the {obj}?"
        bindings.append({
            "question": q,
            "answer": human,
            "binding_type": "embedded",
            "binding_depth": i + 1,
            "verb": verb,
            "object_asked": obj
        })
    
    return {
        "sentence": sentence,
        "depth": depth,
        "main_subject": main_subject,
        "main_object": main_object,
        "main_verb": main_verb,
        "humans": humans,
        "embed_verbs": embed_verbs,
        "bindings": bindings,
        "n_bindings": len(bindings)
    }


def generate_all_stimuli(depths, sentences_per_depth, seed_offset=0):
    """Generate all stimuli with binding questions."""
    stimuli = []
    
    for depth in depths:
        for sent_idx in range(sentences_per_depth):
            seed = seed_offset + depth * 1000 + sent_idx
            item = generate_sentence_with_bindings(depth, seed)
            
            # Create one stimulus per binding question
            for bind_idx, binding in enumerate(item["bindings"]):
                stimulus = {
                    "trial_id": f"d{depth}_s{sent_idx:02d}_b{bind_idx}",
                    "sentence_id": f"d{depth}_s{sent_idx:02d}",
                    "sentence": item["sentence"],
                    "depth": depth,
                    "question": binding["question"],
                    "correct_answer": binding["answer"],
                    "binding_type": binding["binding_type"],
                    "binding_depth": binding["binding_depth"],
                    "verb_asked": binding["verb"],
                    "object_asked": binding["object_asked"],
                    "all_humans": item["humans"],
                    "main_subject": item["main_subject"],
                    "main_object": item["main_object"],
                    "seed": seed
                }
                stimuli.append(stimulus)
    
    return stimuli


def main():
    """Generate and save binding stimuli."""
    output_dir = Path(DATA_DIR) / "binding"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING BINDING RECOVERY STIMULI")
    print("=" * 70)
    print(f"Depths: {DEPTHS}")
    print(f"Sentences per depth: {SENTENCES_PER_DEPTH}")
    
    stimuli = generate_all_stimuli(DEPTHS, SENTENCES_PER_DEPTH, seed_offset=200000)
    
    # Count
    n_main = sum(1 for s in stimuli if s["binding_type"] == "main")
    n_embedded = sum(1 for s in stimuli if s["binding_type"] == "embedded")
    
    print(f"\nGenerated {len(stimuli)} total questions:")
    print(f"  Main clause bindings: {n_main}")
    print(f"  Embedded bindings: {n_embedded}")
    
    # Save
    output_path = output_dir / "binding_stimuli.json"
    with open(output_path, "w") as f:
        json.dump(stimuli, f, indent=2)
    print(f"\nSaved to: {output_path}")
    
    # Examples
    print("\n" + "=" * 70)
    print("EXAMPLE: Full binding recovery for one sentence")
    print("=" * 70)
    
    # Find a depth-4 sentence
    d4_stimuli = [s for s in stimuli if s["depth"] == 4]
    sentence_id = d4_stimuli[0]["sentence_id"]
    sentence_items = [s for s in d4_stimuli if s["sentence_id"] == sentence_id]
    
    print(f"\nSentence (depth 4):")
    print(f"  \"{sentence_items[0]['sentence']}\"")
    print(f"\nBinding questions:")
    for item in sentence_items:
        print(f"  [{item['binding_type']:8s} d={item['binding_depth']}] {item['question']}")
        print(f"       → Answer: {item['correct_answer']}")
    
    # Heuristic predictions
    print("\n" + "=" * 70)
    print("HEURISTIC ANALYSIS")
    print("=" * 70)
    print("""
    For embedded bindings, simple heuristics FAIL:
    
    Sentence: "The dog that the man that the woman knew saw chased the cat."
    
    Question: "Who saw the dog?"
    - First noun heuristic: "dog" → WRONG
    - Recency heuristic: "woman" (most recent noun before "saw") → WRONG  
    - Structural parse: "man" → CORRECT
    
    Question: "Who knew the man?"
    - First noun heuristic: "dog" → WRONG
    - Recency heuristic: "woman" → CORRECT (by luck at this depth)
    - Structural parse: "woman" → CORRECT
    
    KEY PREDICTION:
    - Main clause: All strategies succeed
    - Embedded clauses: Only structural parsing succeeds reliably
    """)
    
    # Summary by depth
    print("\n" + "=" * 70)
    print("QUESTIONS PER DEPTH")
    print("=" * 70)
    for depth in DEPTHS:
        depth_stim = [s for s in stimuli if s["depth"] == depth]
        n_main = sum(1 for s in depth_stim if s["binding_type"] == "main")
        n_embed = sum(1 for s in depth_stim if s["binding_type"] == "embedded")
        print(f"  Depth {depth}: {len(depth_stim)} questions ({n_main} main, {n_embed} embedded)")


if __name__ == "__main__":
    main()
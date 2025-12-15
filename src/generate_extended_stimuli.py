"""
Generate extended center-embedded sentence stimuli (depths 1-12).

For deeper embeddings, we need more lexical items to avoid repetition.

Reference: Gibson, E. (1998). Linguistic complexity: Locality of syntactic dependencies.
"""

import json
import random
from pathlib import Path
from config import EXTENDED_DEPTHS, EXTENDED_TRIALS_PER_DEPTH, DATA_DIR


# EXTENDED LEXICAL ITEMS (need more for depth 12)
AGENTS = [
    "dog", "cat", "bird", "horse", "rabbit", 
    "fox", "wolf", "bear", "deer", "mouse",
    "lion", "tiger", "eagle", "owl", "snake"
]

HUMANS = [
    "man", "woman", "boy", "girl", "teacher",
    "doctor", "chef", "artist", "farmer", "writer",
    "nurse", "lawyer", "pilot", "sailor", "clerk",
    "mayor", "judge", "coach", "actor", "singer"
]

MAIN_VERBS = ["chased", "watched", "followed", "approached", "startled"]
EMBED_VERBS = ["saw", "knew", "met", "helped", "called", "noticed", "recognized", "remembered", "liked", "trusted", "visited", "admired"]


def generate_sentence(depth, seed=None):
    """
    Generate a center-embedded sentence at specified depth.
    
    Depth 1: "The dog chased the cat."
    Depth 2: "The dog that the man saw chased the cat."
    ...
    Depth 12: Very long nested sentence
    
    Returns dict with sentence, correct_answer, and metadata.
    """
    if seed is not None:
        random.seed(seed)
    
    # Select lexical items (no repeats)
    main_subject = random.choice(AGENTS)
    remaining_agents = [a for a in AGENTS if a != main_subject]
    main_object = random.choice(remaining_agents)
    
    main_verb = random.choice(MAIN_VERBS)
    
    # For embeddings: need (depth - 1) humans and embed verbs
    n_embeddings = depth - 1
    
    if n_embeddings > len(HUMANS):
        raise ValueError(f"Depth {depth} requires {n_embeddings} humans, only have {len(HUMANS)}")
    
    humans = random.sample(HUMANS, n_embeddings) if n_embeddings > 0 else []
    embed_verbs = random.sample(EMBED_VERBS, n_embeddings) if n_embeddings > 0 else []
    
    # Build sentence
    if depth == 1:
        sentence = f"The {main_subject} {main_verb} the {main_object}."
    else:
        # Build center-embedded structure from inside out
        embedded_part = ""
        for i in range(n_embeddings - 1, -1, -1):
            human = humans[i]
            verb = embed_verbs[i]
            if i == n_embeddings - 1:  # Innermost
                embedded_part = f"that the {human} {verb}"
            else:
                embedded_part = f"that the {human} {embedded_part} {verb}"
        
        sentence = f"The {main_subject} {embedded_part} {main_verb} the {main_object}."
    
    # Count nouns for chance calculation
    n_nouns = 2 + n_embeddings  # subject + object + embedded humans
    
    return {
        "sentence": sentence,
        "correct_answer": main_subject,
        "main_object": main_object,
        "embedded_humans": humans,
        "depth": depth,
        "n_nouns": n_nouns,
        "main_verb": main_verb,
        "structure": "center-embedded"
    }


def generate_stimulus_set(depths, trials_per_depth, seed_offset=0):
    """Generate full stimulus set."""
    stimuli = []
    
    for depth in depths:
        for trial in range(trials_per_depth):
            seed = seed_offset + depth * 1000 + trial
            stimulus = generate_sentence(depth, seed)
            stimulus["trial_id"] = f"d{depth:02d}_t{trial:02d}"
            stimulus["seed"] = seed
            stimuli.append(stimulus)
    
    return stimuli


def main():
    """Generate extended experiment stimuli."""
    Path(DATA_DIR).mkdir(exist_ok=True)
    
    print("=" * 70)
    print("GENERATING EXTENDED EXPERIMENT STIMULI (Depths 1-12)")
    print("=" * 70)
    print(f"Depths: {EXTENDED_DEPTHS}")
    print(f"Trials per depth: {EXTENDED_TRIALS_PER_DEPTH}")
    print(f"Total stimuli: {len(EXTENDED_DEPTHS) * EXTENDED_TRIALS_PER_DEPTH}")
    print()
    
    stimuli = generate_stimulus_set(
        EXTENDED_DEPTHS, 
        EXTENDED_TRIALS_PER_DEPTH, 
        seed_offset=50000  # Different from previous experiments
    )
    
    # Save
    output_path = Path(DATA_DIR) / "extended_stimuli.json"
    with open(output_path, "w") as f:
        json.dump(stimuli, f, indent=2)
    
    print(f"Generated {len(stimuli)} stimuli")
    print(f"Saved to: {output_path}")
    
    # Print examples
    print("\n" + "-" * 70)
    print("EXAMPLE STIMULI")
    print("-" * 70)
    
    for depth in [1, 3, 6, 9, 12]:
        example = [s for s in stimuli if s["depth"] == depth][0]
        print(f"\nDepth {depth} (n_nouns={example['n_nouns']}):")
        sent = example['sentence']
        if len(sent) > 100:
            print(f"  {sent[:100]}...")
            print(f"  ...{sent[-50:]}")
        else:
            print(f"  {sent}")
        print(f"  Answer: {example['correct_answer']}")
    
    # Stats
    print("\n" + "-" * 70)
    print("SENTENCE LENGTH STATS")
    print("-" * 70)
    for depth in EXTENDED_DEPTHS:
        depths_stim = [s for s in stimuli if s["depth"] == depth]
        avg_len = sum(len(s["sentence"]) for s in depths_stim) / len(depths_stim)
        print(f"  Depth {depth:2d}: avg {avg_len:.0f} chars")


if __name__ == "__main__":
    main()
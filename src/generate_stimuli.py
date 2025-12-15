"""
Generate center-embedded sentence stimuli at varying depths.

Theoretical basis: Center-embedding requires stack-based processing.
Each embedding adds a dependency that must be tracked across intervening material.

Reference: Gibson, E. (1998). Linguistic complexity: Locality of syntactic dependencies.
"""

import json
import random
from pathlib import Path
from config import MAIN_DEPTHS, MAIN_TRIALS_PER_DEPTH, DATA_DIR


# =============================================================================
# LEXICAL ITEMS
# =============================================================================

AGENTS = ["dog", "cat", "bird", "horse", "rabbit", "fox", "wolf", "bear", "deer", "mouse"]
HUMANS = ["man", "woman", "boy", "girl", "teacher", "doctor", "chef", "artist", "farmer", "writer"]
MAIN_VERBS = ["chased", "watched", "followed", "approached", "startled"]
EMBED_VERBS = ["saw", "knew", "met", "helped", "called"]


# =============================================================================
# SENTENCE GENERATION
# =============================================================================

def generate_sentence(depth: int, seed: int = None) -> dict:
    """
    Generate a center-embedded sentence at specified depth.
    
    Structure:
        Depth 1: "The dog chased the cat."
        Depth 2: "The dog that the man saw chased the cat."
        Depth 3: "The dog that the man that the woman knew saw chased the cat."
    
    The correct answer is always the outermost subject (first noun).
    
    Args:
        depth: Number of embedding levels (1 = no embedding)
        seed: Random seed for reproducibility
    
    Returns:
        dict with sentence, correct_answer, and metadata
    """
    if seed is not None:
        random.seed(seed)
    
    # Select lexical items (no repeats within sentence)
    main_subject = random.choice(AGENTS)
    remaining_agents = [a for a in AGENTS if a != main_subject]
    main_object = random.choice(remaining_agents)
    
    main_verb = random.choice(MAIN_VERBS)
    
    # For embeddings, select humans and verbs
    humans = random.sample(HUMANS, depth - 1) if depth > 1 else []
    embed_verbs = random.sample(EMBED_VERBS, depth - 1) if depth > 1 else []
    
    # Build sentence
    if depth == 1:
        sentence = f"The {main_subject} {main_verb} the {main_object}."
    else:
        # Build center-embedded structure from inside out
        # Innermost: "that the [human_n-1] [verb_n-1]"
        # Each layer wraps: "that the [human_i] [inner] [verb_i]"
        
        embedded_part = ""
        for i in range(depth - 2, -1, -1):
            human = humans[i]
            verb = embed_verbs[i]
            if i == depth - 2:  # Innermost
                embedded_part = f"that the {human} {verb}"
            else:
                embedded_part = f"that the {human} {embedded_part} {verb}"
        
        sentence = f"The {main_subject} {embedded_part} {main_verb} the {main_object}."
    
    # Count nouns for chance calculation
    n_nouns = 2 + (depth - 1)  # subject + object + embedded humans
    
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


def generate_stimulus_set(depths: list, trials_per_depth: int, seed_offset: int = 0) -> list:
    """
    Generate full stimulus set.
    
    Args:
        depths: List of depth levels to generate
        trials_per_depth: Number of unique sentences per depth
        seed_offset: Offset for random seeds (for generating different sets)
    
    Returns:
        List of stimulus dicts
    """
    stimuli = []
    
    for depth in depths:
        for trial in range(trials_per_depth):
            seed = seed_offset + depth * 1000 + trial
            stimulus = generate_sentence(depth, seed)
            stimulus["trial_id"] = f"d{depth}_t{trial:02d}"
            stimulus["seed"] = seed
            stimuli.append(stimulus)
    
    return stimuli


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate and save main experiment stimuli."""
    Path(DATA_DIR).mkdir(exist_ok=True)
    
    print("=" * 60)
    print("GENERATING MAIN EXPERIMENT STIMULI")
    print("=" * 60)
    print(f"Depths: {MAIN_DEPTHS}")
    print(f"Trials per depth: {MAIN_TRIALS_PER_DEPTH}")
    print(f"Total stimuli: {len(MAIN_DEPTHS) * MAIN_TRIALS_PER_DEPTH}")
    print()
    
    # Generate main experiment stimuli
    stimuli = generate_stimulus_set(
        MAIN_DEPTHS, 
        MAIN_TRIALS_PER_DEPTH, 
        seed_offset=10000  # Different from pilot
    )
    
    # Save
    output_path = Path(DATA_DIR) / "main_stimuli.json"
    with open(output_path, "w") as f:
        json.dump(stimuli, f, indent=2)
    
    print(f"Generated {len(stimuli)} stimuli")
    print(f"Saved to: {output_path}")
    
    # Print examples at each depth
    print("\n" + "-" * 60)
    print("EXAMPLE STIMULI")
    print("-" * 60)
    
    for depth in MAIN_DEPTHS:
        example = [s for s in stimuli if s["depth"] == depth][0]
        print(f"\nDepth {depth} (n_nouns={example['n_nouns']}):")
        print(f"  {example['sentence']}")
        print(f"  Answer: {example['correct_answer']}")
    
    # Verify no duplicate sentences
    sentences = [s["sentence"] for s in stimuli]
    n_unique = len(set(sentences))
    print(f"\n✓ Unique sentences: {n_unique}/{len(stimuli)}")
    
    if n_unique < len(stimuli):
        print("  WARNING: Some duplicate sentences detected!")


if __name__ == "__main__":
    main()
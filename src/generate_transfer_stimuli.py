"""
Generate structurally isomorphic stimuli across three domains.

Domain A: Relative Clauses (center-embedded)
Domain B: Possessive Chains  
Domain C: Prepositional Modification

All three test: "Can you identify the head of the subject NP 
through intervening material?"

Theoretical basis: If recursive capacity is structural (not domain-specific),
performance should be similar across domains at the same depth.

Reference: Fodor & Pylyshyn (1988) on systematicity as a test of compositional structure.
"""

import json
import random
from pathlib import Path
from config import DATA_DIR


# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================

DEPTHS = [1, 2, 3, 4, 5, 6]
TRIALS_PER_DEPTH = 20  # Per domain, so 20 × 3 = 60 per depth total


# =============================================================================
# DOMAIN A: RELATIVE CLAUSES (from original experiment)
# Structure: Subject [that Embedder1 [that Embedder2 V2] V1] MainVerb Object
# Question: "Who performed the main action?"
# Answer: Subject (first noun)
# =============================================================================

DOMAIN_A_SUBJECTS = ["dog", "cat", "bird", "horse", "rabbit", "fox", "wolf", "bear", "deer", "mouse"]
DOMAIN_A_OBJECTS = ["ball", "toy", "bone", "stick", "rock", "leaf", "rope", "box", "bag", "cup"]
DOMAIN_A_HUMANS = ["man", "woman", "boy", "girl", "teacher", "doctor", "chef", "artist", "farmer", "writer",
                   "nurse", "lawyer", "pilot", "sailor", "clerk", "mayor", "judge", "coach", "actor", "singer"]
DOMAIN_A_MAIN_VERBS = ["grabbed", "found", "touched", "moved", "carried"]
DOMAIN_A_EMBED_VERBS = ["saw", "knew", "met", "helped", "called", "noticed", "recognized", "watched", "heard", "followed"]


def generate_domain_a(depth, seed=None):
    """Generate relative clause sentence."""
    if seed is not None:
        random.seed(seed)
    
    subject = random.choice(DOMAIN_A_SUBJECTS)
    obj = random.choice(DOMAIN_A_OBJECTS)
    main_verb = random.choice(DOMAIN_A_MAIN_VERBS)
    
    n_embeddings = depth - 1
    humans = random.sample(DOMAIN_A_HUMANS, n_embeddings) if n_embeddings > 0 else []
    embed_verbs = random.sample(DOMAIN_A_EMBED_VERBS, n_embeddings) if n_embeddings > 0 else []
    
    if depth == 1:
        sentence = f"The {subject} {main_verb} the {obj}."
    else:
        embedded_part = ""
        for i in range(n_embeddings - 1, -1, -1):
            human = humans[i]
            verb = embed_verbs[i]
            if i == n_embeddings - 1:
                embedded_part = f"that the {human} {verb}"
            else:
                embedded_part = f"that the {human} {embedded_part} {verb}"
        sentence = f"The {subject} {embedded_part} {main_verb} the {obj}."
    
    return {
        "domain": "A_relative_clause",
        "sentence": sentence,
        "question": "In the following sentence, who performed the main action?",
        "correct_answer": subject,
        "depth": depth,
        "n_nouns": 2 + n_embeddings,
        "distractor_nouns": humans + [obj],
        "answer_position": "first"  # Answer is first noun
    }


# =============================================================================
# DOMAIN B: POSSESSIVE CHAINS
# Structure: Possessor1's Possessor2's... Head is ADJECTIVE
# Question: "What is ADJECTIVE?"
# Answer: Head (LAST noun before "is")
# =============================================================================

DOMAIN_B_POSSESSORS = ["teacher", "doctor", "lawyer", "artist", "farmer", "writer", "chef", "nurse", "pilot", "engineer",
                       "manager", "director", "captain", "professor", "student", "neighbor", "friend", "cousin", "uncle", "aunt"]
DOMAIN_B_HEADS = ["car", "house", "book", "phone", "laptop", "garden", "office", "boat", "watch", "painting",
                  "desk", "chair", "bicycle", "camera", "guitar", "piano", "jacket", "briefcase", "umbrella", "telescope"]
DOMAIN_B_ADJECTIVES = ["expensive", "broken", "beautiful", "ancient", "valuable", "damaged", "impressive", "unusual"]


def generate_domain_b(depth, seed=None):
    """Generate possessive chain sentence."""
    if seed is not None:
        random.seed(seed)
    
    head = random.choice(DOMAIN_B_HEADS)
    adjective = random.choice(DOMAIN_B_ADJECTIVES)
    
    n_possessors = depth - 1
    possessors = random.sample(DOMAIN_B_POSSESSORS, n_possessors) if n_possessors > 0 else []
    
    if depth == 1:
        sentence = f"The {head} is {adjective}."
    else:
        possessive_chain = "'s ".join(possessors) + "'s"
        sentence = f"The {possessive_chain} {head} is {adjective}."
    
    return {
        "domain": "B_possessive_chain",
        "sentence": sentence,
        "question": f"In the following sentence, what is {adjective}?",
        "correct_answer": head,
        "depth": depth,
        "n_nouns": 1 + n_possessors,
        "distractor_nouns": possessors,
        "answer_position": "last_before_verb"  # Answer is last noun before "is"
    }


# =============================================================================
# DOMAIN C: PREPOSITIONAL MODIFICATION
# Structure: Head from/in NP1 from/in NP2... VERB
# Question: "What VERB?"
# Answer: Head (FIRST noun)
# =============================================================================

DOMAIN_C_HEADS = ["letter", "package", "message", "report", "document", "gift", "invitation", "notice", "memo", "file",
                  "receipt", "contract", "proposal", "invoice", "certificate", "permit", "ticket", "voucher", "order", "bill"]
DOMAIN_C_PP_NOUNS = ["lawyer", "doctor", "company", "agency", "office", "department", "firm", "bureau", "bank", "hospital",
                     "university", "foundation", "institute", "corporation", "ministry", "embassy", "consulate", "headquarters", "branch", "division"]
DOMAIN_C_VERBS = ["arrived yesterday", "was delivered", "got lost", "was found", "disappeared", "was opened", "was signed", "was approved"]


def generate_domain_c(depth, seed=None):
    """Generate prepositional chain sentence."""
    if seed is not None:
        random.seed(seed)
    
    head = random.choice(DOMAIN_C_HEADS)
    verb_phrase = random.choice(DOMAIN_C_VERBS)
    
    n_pps = depth - 1
    pp_nouns = random.sample(DOMAIN_C_PP_NOUNS, n_pps) if n_pps > 0 else []
    
    if depth == 1:
        sentence = f"The {head} {verb_phrase}."
    else:
        pp_chain = " from the ".join(pp_nouns)
        sentence = f"The {head} from the {pp_chain} {verb_phrase}."
    
    # Extract the main verb for the question
    main_verb = verb_phrase.split()[0]  # "arrived", "was", "got", etc.
    if main_verb == "was" or main_verb == "got":
        main_verb = verb_phrase.split()[0] + " " + verb_phrase.split()[1]  # "was delivered", "got lost"
    
    return {
        "domain": "C_prepositional_chain",
        "sentence": sentence,
        "question": f"In the following sentence, what {verb_phrase.split()[0]}?",  # "what arrived?", "what was?", "what got?"
        "correct_answer": head,
        "depth": depth,
        "n_nouns": 1 + n_pps,
        "distractor_nouns": pp_nouns,
        "answer_position": "first"  # Answer is first noun
    }


# =============================================================================
# STIMULUS GENERATION
# =============================================================================

def generate_all_stimuli(depths, trials_per_depth, seed_offset=0):
    """Generate stimuli for all three domains."""
    stimuli = []
    
    generators = {
        "A": generate_domain_a,
        "B": generate_domain_b,
        "C": generate_domain_c
    }
    
    for domain_key, generator in generators.items():
        for depth in depths:
            for trial in range(trials_per_depth):
                seed = seed_offset + ord(domain_key) * 10000 + depth * 100 + trial
                stimulus = generator(depth, seed)
                stimulus["trial_id"] = f"{domain_key}_d{depth}_t{trial:02d}"
                stimulus["seed"] = seed
                stimuli.append(stimulus)
    
    return stimuli


def main():
    """Generate and save transfer experiment stimuli."""
    output_dir = Path(DATA_DIR) / "transfer"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING STRUCTURAL TRANSFER STIMULI")
    print("=" * 70)
    print(f"Domains: A (relative clauses), B (possessives), C (prepositions)")
    print(f"Depths: {DEPTHS}")
    print(f"Trials per depth per domain: {TRIALS_PER_DEPTH}")
    print(f"Total stimuli: {len(DEPTHS) * TRIALS_PER_DEPTH * 3}")
    print()
    
    stimuli = generate_all_stimuli(DEPTHS, TRIALS_PER_DEPTH, seed_offset=100000)
    
    # Save
    output_path = output_dir / "transfer_stimuli.json"
    with open(output_path, "w") as f:
        json.dump(stimuli, f, indent=2)
    
    print(f"Generated {len(stimuli)} stimuli")
    print(f"Saved to: {output_path}")
    
    # Print examples
    print("\n" + "=" * 70)
    print("EXAMPLE STIMULI BY DOMAIN")
    print("=" * 70)
    
    for domain in ["A_relative_clause", "B_possessive_chain", "C_prepositional_chain"]:
        print(f"\n{'='*70}")
        print(f"DOMAIN: {domain}")
        print(f"{'='*70}")
        
        domain_stim = [s for s in stimuli if s["domain"] == domain]
        
        for depth in [1, 3, 5]:
            example = [s for s in domain_stim if s["depth"] == depth][0]
            print(f"\nDepth {depth}:")
            print(f"  Sentence: {example['sentence']}")
            print(f"  Question: {example['question']}")
            print(f"  Answer: {example['correct_answer']}")
            print(f"  Distractors: {example['distractor_nouns']}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("STRUCTURAL COMPARISON")
    print("=" * 70)
    print("""
    Domain A (Relative Clauses):
      - Answer position: FIRST noun (subject before embedding)
      - Intervening material: Embedded clauses with own subjects
      - Challenge: Track subject through center-embedded clauses
    
    Domain B (Possessives):
      - Answer position: LAST noun before verb (head of NP)
      - Intervening material: Possessor chain
      - Challenge: Identify head vs possessors
    
    Domain C (Prepositions):
      - Answer position: FIRST noun (head with PP modifiers)
      - Intervening material: Prepositional phrase chain
      - Challenge: Track head through PP modifiers
    
    KEY TEST:
      - "First noun" heuristic → Succeeds on A & C, FAILS on B
      - "Last noun" heuristic → Succeeds on B, FAILS on A & C
      - Structural parsing → Succeeds on ALL THREE
    """)


if __name__ == "__main__":
    main()
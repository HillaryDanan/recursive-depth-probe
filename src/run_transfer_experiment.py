"""
Run structural transfer experiment across three isomorphic domains.

Tests whether recursive processing capacity transfers across domains
(evidence for construction) or is domain-specific (evidence for coverage).
"""

import json
import time
import random
from pathlib import Path
from datetime import datetime

from config import (
    ANTHROPIC_API_KEY, OPENAI_API_KEY,
    MODELS, DATA_DIR
)
from models.anthropic_client import AnthropicClient
from models.openai_client import OpenAIClient


TRANSFER_RESULTS_DIR = "results/transfer"


def create_prompt(stimulus):
    """Create prompt from stimulus (includes domain-specific question)."""
    return f"""{stimulus['question']}

Sentence: "{stimulus['sentence']}"

Answer with just the single word (the noun). Do not explain."""


def get_client(model_key):
    """Initialize client for model."""
    model_config = MODELS[model_key]
    provider = model_config["provider"]
    model_id = model_config["model_id"]
    
    if provider == "anthropic":
        return AnthropicClient(ANTHROPIC_API_KEY, model_id)
    elif provider == "openai":
        return OpenAIClient(OPENAI_API_KEY, model_id)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def score_response(response_text, stimulus):
    """Score response and identify error type."""
    cleaned = response_text.lower().strip().rstrip(".").rstrip(",")
    correct = stimulus["correct_answer"].lower()
    distractors = [d.lower() for d in stimulus.get("distractor_nouns", [])]
    
    is_correct = correct in cleaned
    
    if is_correct:
        error_type = None
        selected_noun = correct
    elif any(d in cleaned for d in distractors):
        error_type = "distractor_error"
        selected_noun = next((d for d in distractors if d in cleaned), None)
    else:
        error_type = "other_error"
        selected_noun = cleaned
    
    return {
        "is_correct": is_correct,
        "error_type": error_type,
        "selected_noun": selected_noun,
        "cleaned_response": cleaned
    }


def run_experiment(model_keys=None, dry_run=False):
    """Run transfer experiment."""
    Path(TRANSFER_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load stimuli
    stimuli_path = Path(DATA_DIR) / "transfer" / "transfer_stimuli.json"
    if not stimuli_path.exists():
        print("ERROR: No stimuli found. Run generate_transfer_stimuli.py first.")
        return
    
    with open(stimuli_path) as f:
        stimuli = json.load(f)
    
    if dry_run:
        print("=" * 70)
        print("DRY RUN MODE (2 trials per depth per domain)")
        print("=" * 70)
        stimuli = [s for s in stimuli if int(s["trial_id"].split("_t")[1]) < 2]
    
    if model_keys is None:
        model_keys = list(MODELS.keys())
    
    print("=" * 70)
    print("STRUCTURAL TRANSFER EXPERIMENT")
    print("=" * 70)
    print(f"Models: {model_keys}")
    print(f"Domains: A (relative clauses), B (possessives), C (prepositions)")
    print(f"Stimuli per model: {len(stimuli)}")
    print(f"Total trials: {len(stimuli) * len(model_keys)}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Randomize (same order for all models)
    random.seed(42)
    random.shuffle(stimuli)
    
    all_results = []
    
    for model_key in model_keys:
        print(f"\n{'='*70}")
        print(f"MODEL: {MODELS[model_key]['display_name']}")
        print(f"{'='*70}")
        
        try:
            client = get_client(model_key)
        except Exception as e:
            print(f"ERROR initializing {model_key}: {e}")
            continue
        
        model_results = []
        
        # Track progress by domain
        domain_progress = {"A": 0, "B": 0, "C": 0}
        
        for i, stimulus in enumerate(stimuli):
            domain = stimulus["domain"][0]  # First char: A, B, or C
            depth = stimulus["depth"]
            
            print(f"  [{i+1}/{len(stimuli)}] {domain} d={depth}", end=" ")
            
            try:
                prompt = create_prompt(stimulus)
                response = client.query(prompt)
                score = score_response(response["response_text"], stimulus)
                
                result = {
                    "model_key": model_key,
                    "model_id": MODELS[model_key]["model_id"],
                    **stimulus,
                    **response,
                    **score,
                    "timestamp": datetime.now().isoformat()
                }
                
                model_results.append(result)
                all_results.append(result)
                
                status = "✓" if score["is_correct"] else "✗"
                print(f"-> {status} ({response['response_text'][:10]}...)")
                
            except Exception as e:
                print(f"-> ERROR: {str(e)[:40]}")
                result = {
                    "model_key": model_key,
                    **stimulus,
                    "error": str(e),
                    "is_correct": None,
                    "timestamp": datetime.now().isoformat()
                }
                all_results.append(result)
            
            time.sleep(0.15)
        
        # Save per-model
        model_output = Path(TRANSFER_RESULTS_DIR) / f"results_{model_key}.json"
        with open(model_output, "w") as f:
            json.dump(model_results, f, indent=2)
        print(f"\nSaved: {model_output}")
        
        # Quick summary by domain
        valid = [r for r in model_results if r.get("is_correct") is not None]
        if valid:
            print(f"\n--- {model_key} Summary by Domain ---")
            for domain in ["A_relative_clause", "B_possessive_chain", "C_prepositional_chain"]:
                domain_short = domain[0]
                print(f"\n  Domain {domain_short}:")
                domain_results = [r for r in valid if r["domain"] == domain]
                for depth in sorted(set(r["depth"] for r in domain_results)):
                    depth_results = [r for r in domain_results if r["depth"] == depth]
                    acc = sum(r["is_correct"] for r in depth_results) / len(depth_results)
                    print(f"    D{depth}: {acc:5.0%} ({sum(r['is_correct'] for r in depth_results)}/{len(depth_results)})")
    
    # Save combined
    combined_output = Path(TRANSFER_RESULTS_DIR) / "results_all_models.json"
    with open(combined_output, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"COMPLETE — {len(all_results)} total trials")
    print(f"Combined results: {combined_output}")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--dry":
            run_experiment(dry_run=True)
        elif arg in MODELS:
            run_experiment(model_keys=[arg])
        else:
            print(f"Unknown arg: {arg}")
            print(f"Options: --dry, {', '.join(MODELS.keys())}")
    else:
        run_experiment()
"""
Run the main multi-model recursive depth probe experiment.

Tests 3 models across 6 depths with 30 trials each (540 total).
"""

import json
import time
import random
from pathlib import Path
from datetime import datetime

from config import (
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY,
    MODELS, DATA_DIR, MAIN_RESULTS_DIR
)

from models.anthropic_client import AnthropicClient
from models.openai_client import OpenAIClient
from models.google_client import GoogleClient


# =============================================================================
# EXPERIMENT SETUP
# =============================================================================

def create_prompt(sentence: str) -> str:
    """Create the probe question."""
    return f"""In the following sentence, who performed the main action?

Sentence: "{sentence}"

Answer with just the single word (the noun) that performed the main action. Do not explain."""


def get_client(model_key: str):
    """Initialize appropriate client for model."""
    model_config = MODELS[model_key]
    provider = model_config["provider"]
    model_id = model_config["model_id"]
    
    if provider == "anthropic":
        return AnthropicClient(ANTHROPIC_API_KEY, model_id)
    elif provider == "openai":
        return OpenAIClient(OPENAI_API_KEY, model_id)
    elif provider == "google":
        return GoogleClient(GOOGLE_API_KEY, model_id)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def score_response(response_text: str, stimulus: dict) -> dict:
    """
    Score the response and categorize error type.
    
    Error types:
    - correct: Response contains correct answer
    - object_error: Selected main object (recency bias)
    - human_error: Selected an embedded human
    - other_error: Some other response
    """
    cleaned = response_text.lower().strip().rstrip(".")
    correct_answer = stimulus["correct_answer"].lower()
    main_object = stimulus["main_object"].lower()
    embedded_humans = [h.lower() for h in stimulus.get("embedded_humans", [])]
    
    # Check correctness
    is_correct = correct_answer in cleaned
    
    if is_correct:
        error_type = None
    elif main_object in cleaned:
        error_type = "object_error"  # Recency bias
    elif any(h in cleaned for h in embedded_humans):
        error_type = "human_error"  # Selected embedded noun
    else:
        error_type = "other_error"
    
    return {
        "is_correct": is_correct,
        "error_type": error_type,
        "cleaned_response": cleaned
    }


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_experiment(model_keys: list = None, dry_run: bool = False):
    """
    Run full experiment across specified models.
    
    Args:
        model_keys: List of model keys to test (default: all)
        dry_run: If True, only run 2 trials per depth per model
    """
    Path(MAIN_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load stimuli
    stimuli_path = Path(DATA_DIR) / "main_stimuli.json"
    if not stimuli_path.exists():
        print("ERROR: No stimuli found. Run generate_stimuli.py first.")
        return
    
    with open(stimuli_path) as f:
        stimuli = json.load(f)
    
    # Dry run mode
    if dry_run:
        print("=" * 60)
        print("DRY RUN MODE (2 trials per depth)")
        print("=" * 60)
        # Take first 2 trials of each depth
        stimuli = [s for s in stimuli if int(s["trial_id"].split("_t")[1]) < 2]
    
    # Default to all models
    if model_keys is None:
        model_keys = list(MODELS.keys())
    
    print("=" * 60)
    print("RECURSIVE DEPTH PROBE — MAIN EXPERIMENT")
    print("=" * 60)
    print(f"Models: {model_keys}")
    print(f"Stimuli: {len(stimuli)}")
    print(f"Total trials: {len(stimuli) * len(model_keys)}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Randomize stimulus order (same order for all models)
    random.seed(42)
    random.shuffle(stimuli)
    
    # Run each model
    all_results = []
    
    for model_key in model_keys:
        print(f"\n{'='*60}")
        print(f"MODEL: {MODELS[model_key]['display_name']}")
        print(f"{'='*60}")
        
        try:
            client = get_client(model_key)
        except Exception as e:
            print(f"ERROR initializing {model_key}: {e}")
            continue
        
        model_results = []
        
        for i, stimulus in enumerate(stimuli):
            depth = stimulus["depth"]
            print(f"  [{i+1}/{len(stimuli)}] depth={depth}", end=" ")
            
            try:
                # Query model
                prompt = create_prompt(stimulus["sentence"])
                response = client.query(prompt)
                
                # Score
                score = score_response(response["response_text"], stimulus)
                
                # Combine result
                result = {
                    "model_key": model_key,
                    "model_id": MODELS[model_key]["model_id"],
                    "display_name": MODELS[model_key]["display_name"],
                    **stimulus,
                    **response,
                    **score,
                    "timestamp": datetime.now().isoformat()
                }
                
                model_results.append(result)
                all_results.append(result)
                
                status = "✓" if score["is_correct"] else "✗"
                print(f"-> {status} ({response['response_text'][:15]}...)")
                
            except Exception as e:
                print(f"-> ERROR: {e}")
                # Log error but continue
                result = {
                    "model_key": model_key,
                    "model_id": MODELS[model_key]["model_id"],
                    **stimulus,
                    "error": str(e),
                    "is_correct": None,
                    "timestamp": datetime.now().isoformat()
                }
                all_results.append(result)
            
            # Rate limiting
            time.sleep(0.2)
        
        # Save per-model results
        model_output = Path(MAIN_RESULTS_DIR) / f"results_{model_key}.json"
        with open(model_output, "w") as f:
            json.dump(model_results, f, indent=2)
        print(f"\nSaved: {model_output}")
        
        # Quick model summary
        valid = [r for r in model_results if r.get("is_correct") is not None]
        if valid:
            print(f"\n--- {model_key} Quick Summary ---")
            for depth in sorted(set(r["depth"] for r in valid)):
                depth_results = [r for r in valid if r["depth"] == depth]
                acc = sum(r["is_correct"] for r in depth_results) / len(depth_results)
                print(f"  Depth {depth}: {acc:.0%} ({sum(r['is_correct'] for r in depth_results)}/{len(depth_results)})")
    
    # Save combined results
    combined_output = Path(MAIN_RESULTS_DIR) / "results_all_models.json"
    with open(combined_output, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"COMPLETE — {len(all_results)} total trials")
    print(f"Combined results: {combined_output}")
    print("=" * 60)


# =============================================================================
# ENTRY POINTS
# =============================================================================

def run_dry():
    """Quick test with 2 trials per depth."""
    run_experiment(dry_run=True)


def run_single_model(model_key: str):
    """Run just one model."""
    run_experiment(model_keys=[model_key])


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--dry":
            run_dry()
        elif arg in MODELS:
            run_single_model(arg)
        else:
            print(f"Unknown argument: {arg}")
            print(f"Options: --dry, {', '.join(MODELS.keys())}")
    else:
        # Full experiment
        run_experiment()
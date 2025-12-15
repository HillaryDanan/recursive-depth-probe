"""
Run extended multi-model recursive depth probe (depths 1-12).

Tests 3 models across 12 depths with 30 trials each (1,080 total).
"""

import json
import time
import random
from pathlib import Path
from datetime import datetime

from config import (
    ANTHROPIC_API_KEY, OPENAI_API_KEY,
    MODELS, DATA_DIR, EXTENDED_RESULTS_DIR
)
from models.anthropic_client import AnthropicClient
from models.openai_client import OpenAIClient


def create_prompt(sentence):
    """Create the probe question."""
    return f"""In the following sentence, who performed the main action?

Sentence: "{sentence}"

Answer with just the single word (the noun) that performed the main action. Do not explain."""


def get_client(model_key):
    """Initialize appropriate client for model."""
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
    """Score response and categorize error type."""
    cleaned = response_text.lower().strip().rstrip(".")
    correct_answer = stimulus["correct_answer"].lower()
    main_object = stimulus["main_object"].lower()
    embedded_humans = [h.lower() for h in stimulus.get("embedded_humans", [])]
    
    is_correct = correct_answer in cleaned
    
    if is_correct:
        error_type = None
    elif main_object in cleaned:
        error_type = "object_error"
    elif any(h in cleaned for h in embedded_humans):
        error_type = "human_error"
    else:
        error_type = "other_error"
    
    return {
        "is_correct": is_correct,
        "error_type": error_type,
        "cleaned_response": cleaned
    }


def run_experiment(model_keys=None, dry_run=False):
    """Run extended experiment."""
    Path(EXTENDED_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load stimuli
    stimuli_path = Path(DATA_DIR) / "extended_stimuli.json"
    if not stimuli_path.exists():
        print("ERROR: No stimuli found. Run generate_extended_stimuli.py first.")
        return
    
    with open(stimuli_path) as f:
        stimuli = json.load(f)
    
    if dry_run:
        print("=" * 70)
        print("DRY RUN MODE (2 trials per depth)")
        print("=" * 70)
        stimuli = [s for s in stimuli if int(s["trial_id"].split("_t")[1]) < 2]
    
    if model_keys is None:
        model_keys = list(MODELS.keys())
    
    print("=" * 70)
    print("RECURSIVE DEPTH PROBE — EXTENDED EXPERIMENT (Depths 1-12)")
    print("=" * 70)
    print(f"Models: {model_keys}")
    print(f"Stimuli per model: {len(stimuli)}")
    print(f"Total trials: {len(stimuli) * len(model_keys)}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Randomize order (same for all models)
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
        
        for i, stimulus in enumerate(stimuli):
            depth = stimulus["depth"]
            print(f"  [{i+1}/{len(stimuli)}] depth={depth:2d}", end=" ")
            
            try:
                prompt = create_prompt(stimulus["sentence"])
                response = client.query(prompt)
                score = score_response(response["response_text"], stimulus)
                
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
                print(f"-> {status} ({response['response_text'][:12]}...)")
                
            except Exception as e:
                print(f"-> ERROR: {str(e)[:50]}")
                result = {
                    "model_key": model_key,
                    "model_id": MODELS[model_key]["model_id"],
                    **stimulus,
                    "error": str(e),
                    "is_correct": None,
                    "timestamp": datetime.now().isoformat()
                }
                all_results.append(result)
            
            time.sleep(0.15)  # Rate limiting
        
        # Save per-model results
        model_output = Path(EXTENDED_RESULTS_DIR) / f"results_{model_key}.json"
        with open(model_output, "w") as f:
            json.dump(model_results, f, indent=2)
        print(f"\nSaved: {model_output}")
        
        # Quick summary
        valid = [r for r in model_results if r.get("is_correct") is not None]
        if valid:
            print(f"\n--- {model_key} Quick Summary ---")
            for depth in sorted(set(r["depth"] for r in valid)):
                depth_results = [r for r in valid if r["depth"] == depth]
                acc = sum(r["is_correct"] for r in depth_results) / len(depth_results)
                n_correct = sum(r["is_correct"] for r in depth_results)
                print(f"  Depth {depth:2d}: {acc:5.0%} ({n_correct}/{len(depth_results)})")
    
    # Save combined
    combined_output = Path(EXTENDED_RESULTS_DIR) / "results_all_models.json"
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
            print(f"Unknown argument: {arg}")
            print(f"Options: --dry, {', '.join(MODELS.keys())}")
    else:
        run_experiment()
"""
Run subject vs object relative clause experiment.

The cleanest test of structural parsing vs pattern matching.
"""

import json
import time
import random
from pathlib import Path
from datetime import datetime

from config import ANTHROPIC_API_KEY, OPENAI_API_KEY, MODELS, DATA_DIR
from models.anthropic_client import AnthropicClient
from models.openai_client import OpenAIClient


RELATIVE_RESULTS_DIR = "results/relatives"


def create_prompt(stimulus):
    """Create prompt."""
    return f"""Read the sentence carefully, then answer the question.

Sentence: "{stimulus['sentence']}"

Question: {stimulus['question']}

Answer with just the single word (the noun). Do not explain."""


def get_client(model_key):
    """Initialize client."""
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
    """Score response."""
    cleaned = response_text.lower().strip().rstrip(".,!?")
    correct = stimulus["correct_answer"].lower()
    
    is_correct = correct in cleaned
    
    # Did model use heuristic answer?
    heuristic = stimulus["heuristic_answer"].lower()
    used_heuristic = heuristic in cleaned
    
    # Error type
    if is_correct:
        error_type = None
    elif used_heuristic:
        error_type = "heuristic_error"  # Used first-noun heuristic
    elif stimulus["second_noun"].lower() in cleaned:
        error_type = "second_noun_error"
    else:
        error_type = "other_error"
    
    return {
        "is_correct": is_correct,
        "used_heuristic_answer": used_heuristic,
        "error_type": error_type,
        "cleaned_response": cleaned
    }


def run_experiment(model_keys=None, dry_run=False):
    """Run experiment."""
    Path(RELATIVE_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load stimuli
    stimuli_path = Path(DATA_DIR) / "relatives" / "relative_stimuli.json"
    if not stimuli_path.exists():
        print("ERROR: No stimuli. Run generate_relative_stimuli.py first.")
        return
    
    with open(stimuli_path) as f:
        stimuli = json.load(f)
    
    if dry_run:
        print("=" * 70)
        print("DRY RUN MODE (3 per condition)")
        print("=" * 70)
        # Take first 3 of each condition
        keep = []
        for depth in [1, 2, 3]:
            for struct in ["subject_relative", "object_relative"]:
                cond = [s for s in stimuli if s["depth"] == depth and s["structure"] == struct][:3]
                keep.extend(cond)
        stimuli = keep
    
    if model_keys is None:
        model_keys = list(MODELS.keys())
    
    print("=" * 70)
    print("SUBJECT VS OBJECT RELATIVE CLAUSE EXPERIMENT")
    print("=" * 70)
    print(f"Models: {model_keys}")
    print(f"Stimuli: {len(stimuli)}")
    print(f"Total trials: {len(stimuli) * len(model_keys)}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)
    
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
            print(f"ERROR: {e}")
            continue
        
        model_results = []
        
        for i, stimulus in enumerate(stimuli):
            struct = "SRC" if stimulus["structure"] == "subject_relative" else "ORC"
            depth = stimulus["depth"]
            
            print(f"  [{i+1}/{len(stimuli)}] D{depth} {struct}", end=" ")
            
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
                heur = "(H)" if score["used_heuristic_answer"] and not score["is_correct"] else ""
                print(f"-> {status}{heur} ({response['response_text'][:10]}...)")
                
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
        model_output = Path(RELATIVE_RESULTS_DIR) / f"results_{model_key}.json"
        with open(model_output, "w") as f:
            json.dump(model_results, f, indent=2)
        print(f"\nSaved: {model_output}")
        
        # Summary
        valid = [r for r in model_results if r.get("is_correct") is not None]
        if valid:
            print(f"\n--- {model_key} Summary ---")
            
            for struct in ["subject_relative", "object_relative"]:
                struct_results = [r for r in valid if r["structure"] == struct]
                acc = sum(r["is_correct"] for r in struct_results) / len(struct_results)
                label = "Subject RC" if struct == "subject_relative" else "Object RC"
                print(f"  {label}: {acc:.0%} ({sum(r['is_correct'] for r in struct_results)}/{len(struct_results)})")
            
            # Heuristic usage on object relatives (where it's wrong)
            obj_wrong = [r for r in valid if r["structure"] == "object_relative" and not r["is_correct"]]
            if obj_wrong:
                heur_errors = sum(1 for r in obj_wrong if r.get("used_heuristic_answer"))
                print(f"\n  Object RC errors using first-noun heuristic: {heur_errors}/{len(obj_wrong)} ({heur_errors/len(obj_wrong):.0%})")
    
    # Save combined
    combined_output = Path(RELATIVE_RESULTS_DIR) / "results_all_models.json"
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
            print(f"Unknown: {arg}")
    else:
        run_experiment()
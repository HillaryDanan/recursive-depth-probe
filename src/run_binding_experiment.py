"""
Run binding recovery experiment.

Tests whether models can recover ALL bindings in center-embedded sentences,
not just the main clause binding.
"""

import json
import time
import random
from pathlib import Path
from datetime import datetime

from config import ANTHROPIC_API_KEY, OPENAI_API_KEY, MODELS, DATA_DIR
from models.anthropic_client import AnthropicClient
from models.openai_client import OpenAIClient


BINDING_RESULTS_DIR = "results/binding"


def create_prompt(stimulus):
    """Create prompt for binding question."""
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
    
    # Identify what was selected
    all_nouns = [stimulus["main_subject"].lower(), stimulus["main_object"].lower()] + \
                [h.lower() for h in stimulus["all_humans"]]
    
    selected = None
    for noun in all_nouns:
        if noun in cleaned:
            selected = noun
            break
    
    # Error categorization
    if is_correct:
        error_type = None
    elif selected == stimulus["main_subject"].lower():
        error_type = "main_subject_error"
    elif selected == stimulus["main_object"].lower():
        error_type = "main_object_error"
    elif selected in [h.lower() for h in stimulus["all_humans"]]:
        error_type = "wrong_human_error"
    else:
        error_type = "other_error"
    
    return {
        "is_correct": is_correct,
        "error_type": error_type,
        "selected_noun": selected,
        "cleaned_response": cleaned
    }


def run_experiment(model_keys=None, dry_run=False):
    """Run binding experiment."""
    Path(BINDING_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load stimuli
    stimuli_path = Path(DATA_DIR) / "binding" / "binding_stimuli.json"
    if not stimuli_path.exists():
        print("ERROR: No stimuli. Run generate_binding_stimuli.py first.")
        return
    
    with open(stimuli_path) as f:
        stimuli = json.load(f)
    
    if dry_run:
        print("=" * 70)
        print("DRY RUN MODE (subset of stimuli)")
        print("=" * 70)
        # Take first 2 sentences per depth (all their bindings)
        keep_sentences = set()
        for depth in [2, 3, 4, 5, 6]:
            depth_sentences = sorted(set(s["sentence_id"] for s in stimuli if s["depth"] == depth))[:2]
            keep_sentences.update(depth_sentences)
        stimuli = [s for s in stimuli if s["sentence_id"] in keep_sentences]
    
    if model_keys is None:
        model_keys = list(MODELS.keys())
    
    print("=" * 70)
    print("BINDING RECOVERY EXPERIMENT")
    print("=" * 70)
    print(f"Models: {model_keys}")
    print(f"Total questions: {len(stimuli)}")
    print(f"Total trials: {len(stimuli) * len(model_keys)}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Randomize (same for all models)
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
            bind_type = stimulus["binding_type"]
            bind_depth = stimulus["binding_depth"]
            
            print(f"  [{i+1}/{len(stimuli)}] D{depth} {bind_type}[{bind_depth}]", end=" ")
            
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
        model_output = Path(BINDING_RESULTS_DIR) / f"results_{model_key}.json"
        with open(model_output, "w") as f:
            json.dump(model_results, f, indent=2)
        print(f"\nSaved: {model_output}")
        
        # Quick summary
        valid = [r for r in model_results if r.get("is_correct") is not None]
        if valid:
            print(f"\n--- {model_key} Summary ---")
            
            # By binding type
            main_results = [r for r in valid if r["binding_type"] == "main"]
            embed_results = [r for r in valid if r["binding_type"] == "embedded"]
            
            main_acc = sum(r["is_correct"] for r in main_results) / len(main_results) if main_results else 0
            embed_acc = sum(r["is_correct"] for r in embed_results) / len(embed_results) if embed_results else 0
            
            print(f"  Main clause bindings:     {main_acc:.0%} ({sum(r['is_correct'] for r in main_results)}/{len(main_results)})")
            print(f"  Embedded clause bindings: {embed_acc:.0%} ({sum(r['is_correct'] for r in embed_results)}/{len(embed_results)})")
            
            # By binding depth
            print(f"\n  By binding depth:")
            for bd in sorted(set(r["binding_depth"] for r in valid)):
                bd_results = [r for r in valid if r["binding_depth"] == bd]
                bd_acc = sum(r["is_correct"] for r in bd_results) / len(bd_results)
                label = "main" if bd == 0 else f"embed-{bd}"
                print(f"    {label}: {bd_acc:.0%}")
    
    # Save combined
    combined_output = Path(BINDING_RESULTS_DIR) / "results_all_models.json"
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
            print(f"Options: --dry, {', '.join(MODELS.keys())}")
    else:
        run_experiment()
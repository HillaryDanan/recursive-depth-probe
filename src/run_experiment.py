"""
Run the recursive depth probe experiment.

Calls Claude API with each stimulus and records responses.
"""

import json
import time
from pathlib import Path
from datetime import datetime

import anthropic

from config import ANTHROPIC_API_KEY, MODEL, DATA_DIR, RESULTS_DIR


def create_prompt(sentence: str) -> str:
    """Create the probe question."""
    return f"""In the following sentence, who performed the main action?

Sentence: "{sentence}"

Answer with just the single word (the noun) that performed the main action. Do not explain."""


def query_model(client: anthropic.Anthropic, sentence: str) -> dict:
    """Query the model and return response with metadata."""
    prompt = create_prompt(sentence)
    
    start_time = time.time()
    
    response = client.messages.create(
        model=MODEL,
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}]
    )
    
    elapsed = time.time() - start_time
    
    return {
        "response_text": response.content[0].text.strip().lower(),
        "response_time": elapsed,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "stop_reason": response.stop_reason
    }


def score_response(response_text: str, correct_answer: str) -> dict:
    """Score the response."""
    # Clean response
    cleaned = response_text.lower().strip().rstrip(".")
    
    # Check if correct
    is_correct = correct_answer.lower() in cleaned
    
    # Categorize error type if wrong
    error_type = None
    if not is_correct:
        if len(cleaned.split()) == 1:
            error_type = "wrong_noun"  # Gave a single noun, just wrong one
        elif len(cleaned.split()) > 3:
            error_type = "verbose"  # Didn't follow instructions
        else:
            error_type = "other"
    
    return {
        "is_correct": is_correct,
        "error_type": error_type,
        "cleaned_response": cleaned
    }


def run_experiment():
    """Run full experiment."""
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    
    # Load stimuli
    stimuli_path = Path(DATA_DIR) / "stimuli.json"
    if not stimuli_path.exists():
        print("ERROR: No stimuli found. Run generate_stimuli.py first.")
        return
    
    with open(stimuli_path) as f:
        stimuli = json.load(f)
    
    print(f"Loaded {len(stimuli)} stimuli")
    print(f"Model: {MODEL}")
    print("-" * 50)
    
    # Initialize client
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Run trials
    results = []
    for i, stimulus in enumerate(stimuli):
        print(f"Trial {i+1}/{len(stimuli)}: depth={stimulus['depth']}", end=" ")
        
        # Query model
        response = query_model(client, stimulus["sentence"])
        
        # Score
        score = score_response(response["response_text"], stimulus["correct_answer"])
        
        # Combine
        result = {
            **stimulus,
            **response,
            **score,
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)
        
        print(f"-> {'✓' if score['is_correct'] else '✗'} ({response['response_text'][:20]})")
        
        # Small delay to be nice to API
        time.sleep(0.1)
    
    # Save results
    output_path = Path(RESULTS_DIR) / "raw_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("-" * 50)
    print(f"Saved results to: {output_path}")
    
    # Quick summary
    print("\n--- Quick Summary ---")
    for depth in sorted(set(r["depth"] for r in results)):
        depth_results = [r for r in results if r["depth"] == depth]
        accuracy = sum(r["is_correct"] for r in depth_results) / len(depth_results)
        print(f"Depth {depth}: {accuracy:.0%} ({sum(r['is_correct'] for r in depth_results)}/{len(depth_results)})")


if __name__ == "__main__":
    run_experiment()
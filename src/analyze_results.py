"""
Analyze recursive depth probe results.

Key analyses:
1. Accuracy by depth (degradation curve)
2. Error structure analysis
3. Test for threshold collapse vs. graceful degradation
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import RESULTS_DIR


def load_results() -> pd.DataFrame:
    """Load results as DataFrame."""
    results_path = Path(RESULTS_DIR) / "raw_results.json"
    
    if not results_path.exists():
        raise FileNotFoundError("No results found. Run run_experiment.py first.")
    
    with open(results_path) as f:
        results = json.load(f)
    
    return pd.DataFrame(results)


def analyze_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy by depth."""
    accuracy = df.groupby("depth").agg(
        n_trials=("is_correct", "count"),
        n_correct=("is_correct", "sum"),
        accuracy=("is_correct", "mean")
    ).reset_index()
    
    return accuracy


def analyze_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze error types by depth."""
    errors = df[~df["is_correct"]].copy()
    
    if len(errors) == 0:
        return pd.DataFrame()
    
    error_analysis = errors.groupby(["depth", "error_type"]).size().unstack(fill_value=0)
    return error_analysis


def plot_degradation_curve(accuracy_df: pd.DataFrame, output_path: Path):
    """Plot accuracy by depth."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(accuracy_df["depth"], accuracy_df["accuracy"], 
            marker="o", linewidth=2, markersize=8, color="#2E86AB")
    
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    
    ax.set_xlabel("Embedding Depth", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Recursive Depth Probe: Degradation Curve", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(accuracy_df["depth"])
    
    # Add annotation
    ax.annotate("Threshold collapse\npredicted here", 
                xy=(4, 0.5), fontsize=9, ha="center", color="gray")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot: {output_path}")
    plt.close()


def compute_collapse_metrics(accuracy_df: pd.DataFrame) -> dict:
    """
    Compute metrics to distinguish threshold collapse from graceful degradation.
    
    Threshold collapse: Sharp drop at some depth
    Graceful degradation: Smooth decline across depths
    """
    accuracies = accuracy_df["accuracy"].values
    depths = accuracy_df["depth"].values
    
    # Compute consecutive drops
    drops = [accuracies[i] - accuracies[i+1] for i in range(len(accuracies)-1)]
    
    # Max single drop
    max_drop = max(drops) if drops else 0
    max_drop_depth = depths[drops.index(max_drop) + 1] if drops else None
    
    # Variance of drops (high = uneven = more collapse-like)
    drop_variance = pd.Series(drops).var() if len(drops) > 1 else 0
    
    # Simple heuristic: collapse if any single drop > 0.3 (30%)
    shows_collapse = max_drop > 0.3
    
    return {
        "max_single_drop": max_drop,
        "max_drop_at_depth": max_drop_depth,
        "drop_variance": drop_variance,
        "shows_threshold_collapse": shows_collapse,
        "interpretation": "threshold collapse" if shows_collapse else "gradual degradation"
    }


def main():
    """Run all analyses."""
    print("Loading results...")
    df = load_results()
    print(f"Loaded {len(df)} trials")
    
    # Accuracy analysis
    print("\n" + "="*50)
    print("ACCURACY BY DEPTH")
    print("="*50)
    accuracy_df = analyze_accuracy(df)
    print(accuracy_df.to_string(index=False))
    
    # Save accuracy table
    accuracy_df.to_csv(Path(RESULTS_DIR) / "accuracy_by_depth.csv", index=False)
    
    # Plot
    plot_degradation_curve(accuracy_df, Path(RESULTS_DIR) / "degradation_curve.png")
    
    # Collapse metrics
    print("\n" + "="*50)
    print("COLLAPSE ANALYSIS")
    print("="*50)
    collapse = compute_collapse_metrics(accuracy_df)
    for k, v in collapse.items():
        print(f"  {k}: {v}")
    
    # Error analysis
    print("\n" + "="*50)
    print("ERROR ANALYSIS")
    print("="*50)
    error_df = analyze_errors(df)
    if len(error_df) > 0:
        print(error_df)
    else:
        print("No errors to analyze (perfect performance)")
    
    # Sample errors
    errors = df[~df["is_correct"]]
    if len(errors) > 0:
        print("\n--- Sample Errors ---")
        for _, row in errors.head(5).iterrows():
            print(f"\nDepth {row['depth']}:")
            print(f"  Sentence: {row['sentence']}")
            print(f"  Correct: {row['correct_answer']}")
            print(f"  Response: {row['cleaned_response']}")
    
    # Final summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Model: claude-3-5-haiku-20241022")
    print(f"Total trials: {len(df)}")
    print(f"Overall accuracy: {df['is_correct'].mean():.1%}")
    print(f"Pattern: {collapse['interpretation']}")
    
    if collapse["shows_threshold_collapse"]:
        print(f"\n⚠️  FINDING: Threshold collapse detected at depth {collapse['max_drop_at_depth']}")
        print("   This is consistent with APH prediction for non-embedded systems.")
    else:
        print(f"\n📊 FINDING: Gradual degradation pattern")
        print("   This would challenge APH prediction if performance remains high at depth 4+")


if __name__ == "__main__":
    main()
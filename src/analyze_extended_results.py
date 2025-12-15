"""
Analyze extended experiment results (depths 1-12, 3 models).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from config import EXTENDED_RESULTS_DIR, ALPHA, COLLAPSE_THRESHOLD


def load_results():
    """Load combined results as DataFrame."""
    results_path = Path(EXTENDED_RESULTS_DIR) / "results_all_models.json"
    
    if not results_path.exists():
        raise FileNotFoundError("No results found. Run run_extended_experiment.py first.")
    
    with open(results_path) as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    df = df[df["is_correct"].notna()].copy()
    df["is_correct"] = df["is_correct"].astype(bool)
    
    return df


def wilson_ci(n_success, n_total, alpha=0.05):
    """Wilson score confidence interval."""
    if n_total == 0:
        return 0, 0, 0
    
    p = n_success / n_total
    z = stats.norm.ppf(1 - alpha/2)
    
    denominator = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denominator
    spread = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denominator
    
    return p, max(0, center - spread), min(1, center + spread)


def compute_accuracy_table(df):
    """Compute accuracy with CIs by depth and model."""
    results = []
    
    for model in df["model_key"].unique():
        for depth in sorted(df["depth"].unique()):
            subset = df[(df["model_key"] == model) & (df["depth"] == depth)]
            n_total = len(subset)
            n_correct = subset["is_correct"].sum()
            
            acc, ci_low, ci_high = wilson_ci(n_correct, n_total)
            
            results.append({
                "model": model,
                "depth": depth,
                "n": n_total,
                "n_correct": n_correct,
                "accuracy": acc,
                "ci_low": ci_low,
                "ci_high": ci_high
            })
    
    return pd.DataFrame(results)


def detect_collapse(accuracy_df, model, threshold=0.30):
    """Detect threshold collapse point."""
    model_data = accuracy_df[accuracy_df["model"] == model].sort_values("depth")
    accuracies = model_data["accuracy"].values
    depths = model_data["depth"].values
    
    # Find first big drop
    for i in range(len(accuracies) - 1):
        drop = accuracies[i] - accuracies[i + 1]
        if drop > threshold:
            # Calculate post-collapse mean
            post = model_data[model_data["depth"] > depths[i]]
            post_mean = post["accuracy"].mean() if len(post) > 0 else None
            
            return {
                "model": model,
                "collapse_detected": True,
                "collapse_from": int(depths[i]),
                "collapse_to": int(depths[i + 1]),
                "drop_magnitude": drop,
                "post_collapse_accuracy": post_mean
            }
    
    return {
        "model": model,
        "collapse_detected": False,
        "collapse_from": None,
        "collapse_to": None,
        "drop_magnitude": None,
        "post_collapse_accuracy": None
    }


def plot_extended_results(accuracy_df, output_dir):
    """Generate publication-quality plot for extended experiment."""
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    
    colors = {
        "haiku": "#E07A5F",      # Coral red
        "sonnet": "#8B5CF6",     # Purple  
        "gpt4o-mini": "#3D405B"  # Dark blue
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for model in accuracy_df["model"].unique():
        model_data = accuracy_df[accuracy_df["model"] == model].sort_values("depth")
        
        ax.plot(model_data["depth"], model_data["accuracy"], 
                marker="o", linewidth=2.5, markersize=8,
                color=colors.get(model, "gray"),
                label=model)
        
        ax.fill_between(model_data["depth"], 
                        model_data["ci_low"], model_data["ci_high"],
                        alpha=0.15, color=colors.get(model, "gray"))
    
    # Chance line (varies by depth, use average)
    ax.axhline(y=0.10, color="gray", linestyle="--", alpha=0.7, label="Chance (~10%)")
    
    ax.set_xlabel("Embedding Depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Recursive Depth Probe: Extended Test (Depths 1-12)")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(range(1, 13))
    ax.legend(loc="lower left", fontsize=10)
    
    # Add annotation for collapse zone
    ax.axvspan(2.5, 4.5, alpha=0.1, color="red", label="_nolegend_")
    ax.text(3.5, 0.98, "Haiku\ncollapse\nzone", ha="center", va="top", fontsize=9, color="#E07A5F")
    
    plt.tight_layout()
    output_path = output_dir / "extended_accuracy_depths_1-12.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def analyze_error_patterns(df):
    """Analyze error patterns by model and depth."""
    errors = df[~df["is_correct"]].copy()
    
    if len(errors) == 0:
        return None
    
    # Aggregate
    error_summary = errors.groupby(["model_key", "depth", "error_type"]).size().reset_index(name="count")
    
    return error_summary


def main():
    """Run all analyses."""
    output_dir = Path(EXTENDED_RESULTS_DIR)
    
    print("=" * 70)
    print("RECURSIVE DEPTH PROBE — EXTENDED ANALYSIS (Depths 1-12)")
    print("=" * 70)
    
    # Load
    print("\nLoading results...")
    df = load_results()
    print(f"Loaded {len(df)} valid trials")
    print(f"Models: {df['model_key'].unique().tolist()}")
    print(f"Depths: {sorted(df['depth'].unique())}")
    
    # Accuracy Table
    print("\n" + "=" * 70)
    print("ACCURACY BY DEPTH × MODEL")
    print("=" * 70)
    
    accuracy_df = compute_accuracy_table(df)
    
    # Print as formatted table
    for model in sorted(accuracy_df["model"].unique()):
        print(f"\n{model}:")
        model_data = accuracy_df[accuracy_df["model"] == model].sort_values("depth")
        for _, row in model_data.iterrows():
            bar = "█" * int(row["accuracy"] * 20) + "░" * (20 - int(row["accuracy"] * 20))
            print(f"  D{row['depth']:2d}: {row['accuracy']:5.0%} {bar} [{row['ci_low']:.0%}-{row['ci_high']:.0%}]")
    
    accuracy_df.to_csv(output_dir / "extended_accuracy_table.csv", index=False)
    
    # Collapse Detection
    print("\n" + "=" * 70)
    print("COLLAPSE ANALYSIS")
    print("=" * 70)
    
    collapse_results = []
    for model in sorted(accuracy_df["model"].unique()):
        result = detect_collapse(accuracy_df, model)
        collapse_results.append(result)
        
        print(f"\n{model}:")
        if result["collapse_detected"]:
            print(f"  ⚠️  COLLAPSE at depth {result['collapse_from']} → {result['collapse_to']}")
            print(f"  Drop magnitude: {result['drop_magnitude']:.1%}")
            print(f"  Post-collapse accuracy: {result['post_collapse_accuracy']:.1%}")
        else:
            print(f"  ✓ No collapse detected (threshold: {COLLAPSE_THRESHOLD:.0%})")
    
    collapse_df = pd.DataFrame(collapse_results)
    collapse_df.to_csv(output_dir / "extended_collapse_analysis.csv", index=False)
    
    # Error Analysis
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 70)
    
    error_df = analyze_error_patterns(df)
    if error_df is not None:
        # Summarize by model
        for model in sorted(df["model_key"].unique()):
            model_errors = error_df[error_df["model_key"] == model]
            total = model_errors["count"].sum()
            print(f"\n{model}: {total} total errors")
            
            if total > 0:
                by_type = model_errors.groupby("error_type")["count"].sum()
                for err_type, count in by_type.items():
                    print(f"  {err_type}: {count} ({count/total:.0%})")
        
        error_df.to_csv(output_dir / "extended_error_analysis.csv", index=False)
    
    # Plot
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plot_extended_results(accuracy_df, output_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nCollapse Status:")
    for result in collapse_results:
        status = f"COLLAPSE at depth {result['collapse_to']}" if result["collapse_detected"] else "NO COLLAPSE"
        print(f"  {result['model']}: {status}")
    
    # Final depths comparison
    print("\nAccuracy at Depth 12:")
    for model in sorted(accuracy_df["model"].unique()):
        d12 = accuracy_df[(accuracy_df["model"] == model) & (accuracy_df["depth"] == 12)]
        if len(d12) > 0:
            row = d12.iloc[0]
            print(f"  {model}: {row['accuracy']:.0%} [{row['ci_low']:.0%}-{row['ci_high']:.0%}]")
    
    print(f"\nResults saved to: {output_dir}")
    
    # Scientific interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    n_collapse = sum(1 for r in collapse_results if r["collapse_detected"])
    n_models = len(collapse_results)
    
    if n_collapse == n_models:
        print("""
All models show threshold collapse, supporting the APH prediction that
LLMs lack genuine recursive construction capacity. Differences in 
collapse depth may reflect training data coverage rather than 
architectural capability.
        """)
    elif n_collapse == 0:
        print("""
No models show threshold collapse through depth 12. This challenges
the APH prediction, suggesting either:
(a) The task doesn't adequately test recursive construction
(b) Models have sufficient coverage for this structure
(c) Some form of construction-like processing is occurring
        """)
    else:
        print(f"""
Mixed results: {n_collapse}/{n_models} models show collapse.

This dissociation is scientifically interesting. Possible explanations:
1. Training data differences (more recursive structures in some corpora)
2. Architectural differences (unknown)
3. Model scale effects
4. Collapse occurs at different depths for different models

Further investigation needed to determine if non-collapsing models
are genuinely constructing or have better coverage.
        """)


if __name__ == "__main__":
    main()
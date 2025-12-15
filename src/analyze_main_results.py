"""
Analyze main experiment results with proper statistical tests.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from config import MAIN_RESULTS_DIR, ALPHA, COLLAPSE_THRESHOLD


def load_results():
    """Load combined results as DataFrame."""
    results_path = Path(MAIN_RESULTS_DIR) / "results_all_models.json"
    
    if not results_path.exists():
        raise FileNotFoundError("No results found. Run run_main_experiment.py first.")
    
    with open(results_path) as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    df = df[df["is_correct"].notna()].copy()
    df["is_correct"] = df["is_correct"].astype(bool)
    
    return df


def wilson_ci(n_success, n_total, alpha=0.05):
    """Wilson score confidence interval for binomial proportion."""
    if n_total == 0:
        return 0, 0, 0
    
    p = n_success / n_total
    z = stats.norm.ppf(1 - alpha/2)
    
    denominator = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denominator
    spread = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denominator
    
    return p, max(0, center - spread), min(1, center + spread)


def compute_accuracy_table(df):
    """Compute accuracy with 95% Wilson CIs by depth and model."""
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


def detect_collapse(accuracy_df, model):
    """Detect threshold collapse for a single model."""
    model_data = accuracy_df[accuracy_df["model"] == model].sort_values("depth")
    accuracies = model_data["accuracy"].values
    depths = model_data["depth"].values
    
    drops = []
    for i in range(len(accuracies) - 1):
        drop = accuracies[i] - accuracies[i + 1]
        drops.append({
            "from_depth": depths[i],
            "to_depth": depths[i + 1],
            "drop": drop
        })
    
    if drops:
        max_drop_info = max(drops, key=lambda x: x["drop"])
    else:
        max_drop_info = {"from_depth": None, "to_depth": None, "drop": 0}
    
    shows_collapse = max_drop_info["drop"] > COLLAPSE_THRESHOLD
    
    if shows_collapse and max_drop_info["to_depth"]:
        post_collapse = model_data[model_data["depth"] >= max_drop_info["to_depth"]]
        post_collapse_mean = post_collapse["accuracy"].mean()
    else:
        post_collapse_mean = None
    
    return {
        "model": model,
        "max_drop": max_drop_info["drop"],
        "collapse_from": max_drop_info["from_depth"],
        "collapse_to": max_drop_info["to_depth"],
        "shows_collapse": shows_collapse,
        "post_collapse_accuracy": post_collapse_mean
    }


def analyze_errors(df):
    """Analyze error types by model and depth."""
    errors = df[~df["is_correct"]].copy()
    
    if len(errors) == 0:
        return pd.DataFrame()
    
    error_counts = errors.groupby(["model_key", "depth", "error_type"]).size().reset_index(name="count")
    
    return error_counts


def test_recency_bias(df):
    """Test if object_error (recency) occurs more than chance."""
    errors = df[~df["is_correct"]].copy()
    
    if len(errors) == 0:
        return {"test": "recency_bias", "result": "no_errors"}
    
    n_total_errors = len(errors)
    n_object_errors = (errors["error_type"] == "object_error").sum()
    
    mean_n_nouns = errors["n_nouns"].mean()
    expected_prop = 1 / mean_n_nouns
    
    observed_prop = n_object_errors / n_total_errors
    
    result = stats.binomtest(n_object_errors, n_total_errors, expected_prop, alternative="greater")
    
    return {
        "test": "recency_bias",
        "n_errors": n_total_errors,
        "n_object_errors": n_object_errors,
        "observed_proportion": observed_prop,
        "expected_proportion": expected_prop,
        "p_value": result.pvalue,
        "significant": result.pvalue < ALPHA
    }


def plot_results(accuracy_df, output_dir):
    """Generate publication-quality plots."""
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 11
    
    colors = {"haiku": "#E07A5F", "gpt4o-mini": "#3D405B"}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in accuracy_df["model"].unique():
        model_data = accuracy_df[accuracy_df["model"] == model].sort_values("depth")
        
        ax.plot(model_data["depth"], model_data["accuracy"], 
                marker="o", linewidth=2, markersize=8,
                color=colors.get(model, "gray"),
                label=model)
        
        ax.fill_between(model_data["depth"], 
                        model_data["ci_low"], model_data["ci_high"],
                        alpha=0.2, color=colors.get(model, "gray"))
    
    ax.axhline(y=0.17, color="gray", linestyle="--", alpha=0.7, label="Chance (~17%)")
    
    ax.set_xlabel("Embedding Depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Recursive Depth Probe: Haiku vs GPT-4o-mini")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(sorted(accuracy_df["depth"].unique()))
    ax.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_by_depth_all_models.png", dpi=300)
    print(f"Saved: {output_dir / 'accuracy_by_depth_all_models.png'}")
    plt.close()


def compare_models_at_depth(df, depth):
    """Chi-square test comparing models at specific depth."""
    subset = df[df["depth"] == depth]
    
    contingency = pd.crosstab(subset["model_key"], subset["is_correct"])
    
    if contingency.shape == (2, 2):
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        return {"depth": depth, "chi2": chi2, "p_value": p, "dof": dof}
    else:
        return {"depth": depth, "chi2": None, "p_value": None, "note": "insufficient data"}


def main():
    """Run all analyses."""
    output_dir = Path(MAIN_RESULTS_DIR)
    
    print("=" * 70)
    print("RECURSIVE DEPTH PROBE — MAIN ANALYSIS")
    print("=" * 70)
    
    # Load data
    print("\nLoading results...")
    df = load_results()
    print(f"Loaded {len(df)} valid trials")
    print(f"Models: {df['model_key'].unique().tolist()}")
    
    # Accuracy Table
    print("\n" + "=" * 70)
    print("ACCURACY BY DEPTH × MODEL")
    print("=" * 70)
    
    accuracy_df = compute_accuracy_table(df)
    
    for model in accuracy_df["model"].unique():
        print(f"\n{model}:")
        model_data = accuracy_df[accuracy_df["model"] == model]
        for _, row in model_data.iterrows():
            ci_str = f"[{row['ci_low']:.0%}, {row['ci_high']:.0%}]"
            print(f"  Depth {row['depth']}: {row['accuracy']:.0%} {ci_str} (n={row['n']})")
    
    accuracy_df.to_csv(output_dir / "accuracy_table.csv", index=False)
    
    # Collapse Detection
    print("\n" + "=" * 70)
    print("COLLAPSE ANALYSIS")
    print("=" * 70)
    
    collapse_results = []
    for model in accuracy_df["model"].unique():
        result = detect_collapse(accuracy_df, model)
        collapse_results.append(result)
        
        print(f"\n{model}:")
        print(f"  Max drop: {result['max_drop']:.1%} (depth {result['collapse_from']} → {result['collapse_to']})")
        print(f"  Shows collapse (>{COLLAPSE_THRESHOLD:.0%}): {result['shows_collapse']}")
        if result["post_collapse_accuracy"] is not None:
            print(f"  Post-collapse accuracy: {result['post_collapse_accuracy']:.1%}")
    
    collapse_df = pd.DataFrame(collapse_results)
    collapse_df.to_csv(output_dir / "collapse_analysis.csv", index=False)
    
    # Model Comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (Chi-square tests)")
    print("=" * 70)
    
    for depth in sorted(df["depth"].unique()):
        result = compare_models_at_depth(df, depth)
        if result["p_value"] is not None:
            sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else ""
            print(f"  Depth {depth}: χ²={result['chi2']:.2f}, p={result['p_value']:.4f} {sig}")
    
    # Error Analysis
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)
    
    error_df = analyze_errors(df)
    if len(error_df) > 0:
        print("\nError counts by model × depth × type:")
        print(error_df.to_string(index=False))
        error_df.to_csv(output_dir / "error_analysis.csv", index=False)
    
    # Recency Bias
    print("\n--- Recency Bias Test ---")
    for model in df["model_key"].unique():
        model_df = df[df["model_key"] == model]
        result = test_recency_bias(model_df)
        
        if "n_errors" in result:
            sig = "SIGNIFICANT" if result["significant"] else "not significant"
            print(f"\n{model}:")
            print(f"  Object errors: {result['n_object_errors']}/{result['n_errors']} ({result['observed_proportion']:.1%})")
            print(f"  Expected if random: {result['expected_proportion']:.1%}")
            print(f"  p-value: {result['p_value']:.4f} ({sig})")
    
    # Generate Plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plot_results(accuracy_df, output_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    haiku_collapse = [r for r in collapse_results if r["model"] == "haiku"][0]
    gpt_collapse = [r for r in collapse_results if r["model"] == "gpt4o-mini"][0]
    
    print(f"\nHaiku: {'COLLAPSE' if haiku_collapse['shows_collapse'] else 'NO COLLAPSE'} at depth {haiku_collapse['collapse_to']}")
    print(f"GPT-4o-mini: {'COLLAPSE' if gpt_collapse['shows_collapse'] else 'NO COLLAPSE'}")
    
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    
    if haiku_collapse["shows_collapse"] and not gpt_collapse["shows_collapse"]:
        print("""
FINDING: Model dissociation detected.

- Haiku shows threshold collapse (consistent with APH prediction)
- GPT-4o-mini shows NO collapse through depth 6

Possible interpretations:
1. GPT-4o-mini has better training coverage of recursive structures
2. Architectural differences (unknown)
3. GPT-4o-mini may collapse at higher depths (untested)

This does NOT falsify APH — it suggests model differences in where/whether
collapse occurs. Further testing with deeper embeddings and additional
models would clarify whether this is a principled vs. practical limit.
        """)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
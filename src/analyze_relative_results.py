"""
Analyze subject vs object relative clause results.

Key question: Do models fail specifically on object relatives 
(where heuristics don't work)?
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

RELATIVE_RESULTS_DIR = Path("results/relatives")


def load_results():
    """Load results."""
    results_path = RELATIVE_RESULTS_DIR / "results_all_models.json"
    
    if not results_path.exists():
        raise FileNotFoundError("No results.")
    
    with open(results_path) as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    df = df[df["is_correct"].notna()].copy()
    df["is_correct"] = df["is_correct"].astype(bool)
    
    return df


def wilson_ci(n_success, n_total, alpha=0.05):
    """Wilson CI."""
    if n_total == 0:
        return 0, 0, 0
    p = n_success / n_total
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    spread = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return p, max(0, center - spread), min(1, center + spread)


def compute_accuracy_by_structure(df):
    """Compute accuracy for subject vs object relatives."""
    results = []
    
    for model in df["model_key"].unique():
        for struct in ["subject_relative", "object_relative"]:
            for depth in sorted(df["depth"].unique()):
                subset = df[(df["model_key"] == model) & 
                           (df["structure"] == struct) & 
                           (df["depth"] == depth)]
                n = len(subset)
                n_correct = subset["is_correct"].sum()
                acc, ci_lo, ci_hi = wilson_ci(n_correct, n)
                
                results.append({
                    "model": model,
                    "structure": struct,
                    "depth": depth,
                    "n": n,
                    "n_correct": n_correct,
                    "accuracy": acc,
                    "ci_low": ci_lo,
                    "ci_high": ci_hi
                })
    
    return pd.DataFrame(results)


def test_src_vs_orc(df):
    """Test difference between subject and object relatives."""
    results = []
    
    for model in df["model_key"].unique():
        model_df = df[df["model_key"] == model]
        
        src = model_df[model_df["structure"] == "subject_relative"]["is_correct"]
        orc = model_df[model_df["structure"] == "object_relative"]["is_correct"]
        
        src_acc = src.mean()
        orc_acc = orc.mean()
        diff = src_acc - orc_acc
        
        # Chi-square
        contingency = pd.crosstab(model_df["structure"], model_df["is_correct"])
        if contingency.shape == (2, 2):
            chi2, p, _, _ = stats.chi2_contingency(contingency)
        else:
            chi2, p = None, None
        
        results.append({
            "model": model,
            "subject_relative_acc": src_acc,
            "object_relative_acc": orc_acc,
            "difference": diff,
            "chi2": chi2,
            "p_value": p,
            "significant": p < 0.05 if p else None
        })
    
    return pd.DataFrame(results)


def analyze_heuristic_errors(df):
    """Analyze how often errors on object relatives use the first-noun heuristic."""
    results = []
    
    for model in df["model_key"].unique():
        # Object relative errors only
        orc_errors = df[(df["model_key"] == model) & 
                        (df["structure"] == "object_relative") & 
                        (~df["is_correct"])]
        
        if len(orc_errors) == 0:
            results.append({
                "model": model,
                "orc_errors": 0,
                "heuristic_errors": 0,
                "heuristic_pct": None
            })
            continue
        
        heuristic_errors = orc_errors["used_heuristic_answer"].sum()
        
        results.append({
            "model": model,
            "orc_errors": len(orc_errors),
            "heuristic_errors": heuristic_errors,
            "heuristic_pct": heuristic_errors / len(orc_errors)
        })
    
    return pd.DataFrame(results)


def plot_results(accuracy_df, output_dir):
    """Generate plots."""
    sns.set_style("whitegrid")
    colors = {"haiku": "#E07A5F", "sonnet": "#8B5CF6", "gpt4o-mini": "#3D405B"}
    
    # Main plot: SRC vs ORC by model
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = sorted(accuracy_df["model"].unique())
    x = np.arange(len(models))
    width = 0.35
    
    # Aggregate across depths
    src_accs = []
    orc_accs = []
    src_errs = []
    orc_errs = []
    
    for model in models:
        src_data = accuracy_df[(accuracy_df["model"] == model) & 
                               (accuracy_df["structure"] == "subject_relative")]
        orc_data = accuracy_df[(accuracy_df["model"] == model) & 
                               (accuracy_df["structure"] == "object_relative")]
        
        src_acc = src_data["n_correct"].sum() / src_data["n"].sum()
        orc_acc = orc_data["n_correct"].sum() / orc_data["n"].sum()
        
        src_accs.append(src_acc)
        orc_accs.append(orc_acc)
    
    bars1 = ax.bar(x - width/2, src_accs, width, label="Subject Relatives",
                   color=[colors.get(m, "gray") for m in models], alpha=0.9)
    bars2 = ax.bar(x + width/2, orc_accs, width, label="Object Relatives",
                   color=[colors.get(m, "gray") for m in models], alpha=0.5)
    
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    
    ax.set_ylabel("Accuracy")
    ax.set_title("Subject vs Object Relative Clauses\n(Heuristics work for SRC, fail for ORC)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Add difference annotations
    for i, model in enumerate(models):
        diff = src_accs[i] - orc_accs[i]
        if diff > 0.05:
            ax.annotate(f"Δ={diff:.0%}", xy=(i, max(src_accs[i], orc_accs[i]) + 0.05),
                       ha="center", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_dir / "src_vs_orc.png", dpi=300)
    print(f"Saved: {output_dir / 'src_vs_orc.png'}")
    plt.close()
    
    # Plot by depth
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    
    for idx, depth in enumerate([1, 2, 3]):
        ax = axes[idx]
        depth_data = accuracy_df[accuracy_df["depth"] == depth]
        
        for model in models:
            model_data = depth_data[depth_data["model"] == model]
            
            src = model_data[model_data["structure"] == "subject_relative"]["accuracy"].values
            orc = model_data[model_data["structure"] == "object_relative"]["accuracy"].values
            
            if len(src) > 0 and len(orc) > 0:
                ax.scatter(["SRC", "ORC"], [src[0], orc[0]], 
                          s=100, color=colors.get(model, "gray"), label=model if idx == 0 else "")
                ax.plot(["SRC", "ORC"], [src[0], orc[0]], 
                       color=colors.get(model, "gray"), alpha=0.5)
        
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"Depth {depth}")
        ax.set_ylim(0, 1.1)
        
        if idx == 0:
            ax.legend()
            ax.set_ylabel("Accuracy")
    
    plt.suptitle("Subject vs Object Relatives by Depth", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "src_vs_orc_by_depth.png", dpi=300)
    print(f"Saved: {output_dir / 'src_vs_orc_by_depth.png'}")
    plt.close()


def main():
    """Run analysis."""
    print("=" * 70)
    print("SUBJECT VS OBJECT RELATIVE CLAUSE ANALYSIS")
    print("=" * 70)
    
    df = load_results()
    print(f"Loaded {len(df)} trials")
    print(f"Models: {df['model_key'].unique().tolist()}")
    
    # Accuracy by structure
    print("\n" + "=" * 70)
    print("ACCURACY BY STRUCTURE")
    print("=" * 70)
    
    accuracy_df = compute_accuracy_by_structure(df)
    
    for model in sorted(accuracy_df["model"].unique()):
        print(f"\n{model}:")
        model_data = accuracy_df[accuracy_df["model"] == model]
        
        for struct in ["subject_relative", "object_relative"]:
            struct_data = model_data[model_data["structure"] == struct]
            total_n = struct_data["n"].sum()
            total_correct = struct_data["n_correct"].sum()
            overall_acc = total_correct / total_n
            
            label = "Subject RC" if struct == "subject_relative" else "Object RC"
            heur = "(heuristic works)" if struct == "subject_relative" else "(heuristic FAILS)"
            print(f"  {label}: {overall_acc:.0%} ({total_correct}/{total_n}) {heur}")
            
            for _, row in struct_data.iterrows():
                print(f"    D{row['depth']}: {row['accuracy']:.0%}")
    
    accuracy_df.to_csv(RELATIVE_RESULTS_DIR / "accuracy_by_structure.csv", index=False)
    
    # Statistical test
    print("\n" + "=" * 70)
    print("STATISTICAL TEST: Subject vs Object Relatives")
    print("=" * 70)
    
    test_df = test_src_vs_orc(df)
    
    print("\n  Model       | SRC    | ORC    | Diff   | χ²     | p-value")
    print("  " + "-" * 60)
    
    for _, row in test_df.iterrows():
        sig = "***" if row["p_value"] and row["p_value"] < 0.001 else \
              "**" if row["p_value"] and row["p_value"] < 0.01 else \
              "*" if row["p_value"] and row["p_value"] < 0.05 else ""
        p_str = f"{row['p_value']:.4f}" if row["p_value"] else "N/A"
        chi_str = f"{row['chi2']:.1f}" if row["chi2"] else "N/A"
        print(f"  {row['model']:12s} | {row['subject_relative_acc']:.0%}   | {row['object_relative_acc']:.0%}   | {row['difference']:+.0%}   | {chi_str:6s} | {p_str} {sig}")
    
    test_df.to_csv(RELATIVE_RESULTS_DIR / "src_orc_test.csv", index=False)
    
    # Heuristic error analysis
    print("\n" + "=" * 70)
    print("HEURISTIC ERROR ANALYSIS")
    print("=" * 70)
    print("\nWhen models fail on Object Relatives, do they use the first-noun heuristic?")
    
    heuristic_df = analyze_heuristic_errors(df)
    
    for _, row in heuristic_df.iterrows():
        if row["orc_errors"] > 0:
            print(f"\n{row['model']}:")
            print(f"  Object RC errors: {row['orc_errors']}")
            print(f"  Used first-noun heuristic: {row['heuristic_errors']} ({row['heuristic_pct']:.0%})")
    
    heuristic_df.to_csv(RELATIVE_RESULTS_DIR / "heuristic_errors.csv", index=False)
    
    # Plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plot_results(accuracy_df, RELATIVE_RESULTS_DIR)
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    for _, row in test_df.iterrows():
        model = row["model"]
        diff = row["difference"]
        sig = row["significant"]
        orc_acc = row["object_relative_acc"]
        
        print(f"\n{model}:")
        
        if diff > 0.2 and sig:
            print(f"  SRC >> ORC ({diff:+.0%}, p<0.05)")
            print(f"  → PATTERN MATCHING (fails when heuristic fails)")
        elif diff > 0.1:
            print(f"  SRC > ORC ({diff:+.0%})")
            print(f"  → Tendency toward heuristic use")
        elif orc_acc > 0.8:
            print(f"  SRC ≈ ORC (both high)")
            print(f"  → STRUCTURAL PARSING (succeeds without heuristic)")
        elif orc_acc < 0.6:
            print(f"  Both low (ORC={orc_acc:.0%})")
            print(f"  → General failure on relative clauses")
        else:
            print(f"  Difference = {diff:+.0%}")
            print(f"  → Inconclusive")
    
    print(f"\nResults saved to: {RELATIVE_RESULTS_DIR}")


if __name__ == "__main__":
    main()
"""
Analyze binding recovery results.

Key question: Do models recover ALL bindings, or just main clause?

Prediction:
- Pattern matching: Main clause HIGH, embedded LOW
- Structural parsing: Both HIGH
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

BINDING_RESULTS_DIR = Path("results/binding")


def load_results():
    """Load results."""
    results_path = BINDING_RESULTS_DIR / "results_all_models.json"
    
    if not results_path.exists():
        raise FileNotFoundError("No results. Run run_binding_experiment.py first.")
    
    with open(results_path) as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    df = df[df["is_correct"].notna()].copy()
    df["is_correct"] = df["is_correct"].astype(bool)
    
    return df


def wilson_ci(n_success, n_total, alpha=0.05):
    """Wilson score CI."""
    if n_total == 0:
        return 0, 0, 0
    p = n_success / n_total
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    spread = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return p, max(0, center - spread), min(1, center + spread)


def compute_main_vs_embedded(df):
    """Compute accuracy for main vs embedded bindings."""
    results = []
    
    for model in df["model_key"].unique():
        model_df = df[df["model_key"] == model]
        
        for bind_type in ["main", "embedded"]:
            subset = model_df[model_df["binding_type"] == bind_type]
            n = len(subset)
            n_correct = subset["is_correct"].sum()
            acc, ci_lo, ci_hi = wilson_ci(n_correct, n)
            
            results.append({
                "model": model,
                "binding_type": bind_type,
                "n": n,
                "n_correct": n_correct,
                "accuracy": acc,
                "ci_low": ci_lo,
                "ci_high": ci_hi
            })
    
    return pd.DataFrame(results)


def compute_by_binding_depth(df):
    """Compute accuracy by binding depth (0=main, 1+=embedded)."""
    results = []
    
    for model in df["model_key"].unique():
        model_df = df[df["model_key"] == model]
        
        for bd in sorted(model_df["binding_depth"].unique()):
            subset = model_df[model_df["binding_depth"] == bd]
            n = len(subset)
            n_correct = subset["is_correct"].sum()
            acc, ci_lo, ci_hi = wilson_ci(n_correct, n)
            
            results.append({
                "model": model,
                "binding_depth": bd,
                "n": n,
                "n_correct": n_correct,
                "accuracy": acc,
                "ci_low": ci_lo,
                "ci_high": ci_hi
            })
    
    return pd.DataFrame(results)


def test_main_vs_embedded_difference(df):
    """
    Statistical test: Is embedded accuracy significantly lower than main?
    
    This is the key test for pattern matching vs structural parsing.
    """
    results = []
    
    for model in df["model_key"].unique():
        model_df = df[df["model_key"] == model]
        
        main = model_df[model_df["binding_type"] == "main"]["is_correct"]
        embed = model_df[model_df["binding_type"] == "embedded"]["is_correct"]
        
        main_acc = main.mean()
        embed_acc = embed.mean()
        diff = main_acc - embed_acc
        
        # Chi-square test
        contingency = pd.crosstab(
            model_df["binding_type"],
            model_df["is_correct"]
        )
        
        if contingency.shape == (2, 2):
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
        else:
            chi2, p = None, None
        
        results.append({
            "model": model,
            "main_accuracy": main_acc,
            "embedded_accuracy": embed_acc,
            "difference": diff,
            "chi2": chi2,
            "p_value": p,
            "significant": p < 0.05 if p else None
        })
    
    return pd.DataFrame(results)


def analyze_error_patterns(df):
    """What do models select when wrong on embedded bindings?"""
    errors = df[(~df["is_correct"]) & (df["binding_type"] == "embedded")].copy()
    
    if len(errors) == 0:
        return None
    
    results = []
    
    for model in errors["model_key"].unique():
        model_errors = errors[errors["model_key"] == model]
        total = len(model_errors)
        
        if total == 0:
            continue
        
        # Count error types
        main_subj = (model_errors["error_type"] == "main_subject_error").sum()
        main_obj = (model_errors["error_type"] == "main_object_error").sum()
        wrong_human = (model_errors["error_type"] == "wrong_human_error").sum()
        other = (model_errors["error_type"] == "other_error").sum()
        
        results.append({
            "model": model,
            "total_errors": total,
            "main_subject_errors": main_subj,
            "main_subject_pct": main_subj / total,
            "main_object_errors": main_obj,
            "main_object_pct": main_obj / total,
            "wrong_human_errors": wrong_human,
            "wrong_human_pct": wrong_human / total,
            "other_errors": other
        })
    
    return pd.DataFrame(results)


def plot_binding_results(main_embed_df, by_depth_df, output_dir):
    """Generate plots."""
    sns.set_style("whitegrid")
    colors = {"haiku": "#E07A5F", "sonnet": "#8B5CF6", "gpt4o-mini": "#3D405B"}
    
    # Plot 1: Main vs Embedded comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = sorted(main_embed_df["model"].unique())
    x = np.arange(len(models))
    width = 0.35
    
    main_accs = []
    embed_accs = []
    main_errs = []
    embed_errs = []
    
    for model in models:
        main_row = main_embed_df[(main_embed_df["model"] == model) & 
                                  (main_embed_df["binding_type"] == "main")].iloc[0]
        embed_row = main_embed_df[(main_embed_df["model"] == model) & 
                                   (main_embed_df["binding_type"] == "embedded")].iloc[0]
        
        main_accs.append(main_row["accuracy"])
        embed_accs.append(embed_row["accuracy"])
        main_errs.append([main_row["accuracy"] - main_row["ci_low"],
                         main_row["ci_high"] - main_row["accuracy"]])
        embed_errs.append([embed_row["accuracy"] - embed_row["ci_low"],
                          embed_row["ci_high"] - embed_row["accuracy"]])
    
    main_errs = np.array(main_errs).T
    embed_errs = np.array(embed_errs).T
    
    bars1 = ax.bar(x - width/2, main_accs, width, label="Main Clause",
                   color=[colors.get(m, "gray") for m in models], alpha=0.8,
                   yerr=main_errs, capsize=4)
    bars2 = ax.bar(x + width/2, embed_accs, width, label="Embedded Clauses",
                   color=[colors.get(m, "gray") for m in models], alpha=0.4,
                   yerr=embed_errs, capsize=4)
    
    ax.set_ylabel("Accuracy")
    ax.set_title("Binding Recovery: Main vs Embedded Clauses")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.2, color="gray", linestyle="--", alpha=0.5)
    
    # Add significance markers
    for i, model in enumerate(models):
        diff = main_accs[i] - embed_accs[i]
        if diff > 0.1:  # Notable difference
            ax.annotate(f"Δ={diff:.0%}", xy=(i, max(main_accs[i], embed_accs[i]) + 0.08),
                       ha="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "binding_main_vs_embedded.png", dpi=300)
    print(f"Saved: {output_dir / 'binding_main_vs_embedded.png'}")
    plt.close()
    
    # Plot 2: Accuracy by binding depth
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in models:
        model_data = by_depth_df[by_depth_df["model"] == model].sort_values("binding_depth")
        ax.plot(model_data["binding_depth"], model_data["accuracy"],
                marker="o", linewidth=2, markersize=8,
                color=colors.get(model, "gray"), label=model)
        ax.fill_between(model_data["binding_depth"],
                       model_data["ci_low"], model_data["ci_high"],
                       color=colors.get(model, "gray"), alpha=0.15)
    
    ax.set_xlabel("Binding Depth (0=main, 1+=embedded)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Binding Recovery by Depth")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.axhline(y=0.2, color="gray", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / "binding_by_depth.png", dpi=300)
    print(f"Saved: {output_dir / 'binding_by_depth.png'}")
    plt.close()


def main():
    """Run binding analysis."""
    print("=" * 70)
    print("BINDING RECOVERY ANALYSIS")
    print("=" * 70)
    
    df = load_results()
    print(f"Loaded {len(df)} trials")
    print(f"Models: {df['model_key'].unique().tolist()}")
    
    # Main vs Embedded
    print("\n" + "=" * 70)
    print("MAIN VS EMBEDDED BINDING ACCURACY")
    print("=" * 70)
    
    main_embed_df = compute_main_vs_embedded(df)
    
    for model in sorted(main_embed_df["model"].unique()):
        print(f"\n{model}:")
        model_data = main_embed_df[main_embed_df["model"] == model]
        for _, row in model_data.iterrows():
            ci = f"[{row['ci_low']:.0%}-{row['ci_high']:.0%}]"
            print(f"  {row['binding_type']:10s}: {row['accuracy']:5.0%} {ci} (n={row['n']})")
    
    main_embed_df.to_csv(BINDING_RESULTS_DIR / "main_vs_embedded.csv", index=False)
    
    # Statistical test
    print("\n" + "=" * 70)
    print("STATISTICAL TEST: Main vs Embedded Difference")
    print("=" * 70)
    
    test_df = test_main_vs_embedded_difference(df)
    
    print("\n  Model       | Main | Embed | Diff  | χ²    | p-value | Sig?")
    print("  " + "-" * 60)
    
    for _, row in test_df.iterrows():
        sig = "***" if row["p_value"] and row["p_value"] < 0.001 else \
              "**" if row["p_value"] and row["p_value"] < 0.01 else \
              "*" if row["p_value"] and row["p_value"] < 0.05 else ""
        p_str = f"{row['p_value']:.4f}" if row["p_value"] else "N/A"
        chi_str = f"{row['chi2']:.1f}" if row["chi2"] else "N/A"
        print(f"  {row['model']:12s} | {row['main_accuracy']:.0%}  | {row['embedded_accuracy']:.0%}   | {row['difference']:+.0%}  | {chi_str:5s} | {p_str:7s} | {sig}")
    
    test_df.to_csv(BINDING_RESULTS_DIR / "main_embed_test.csv", index=False)
    
    # By binding depth
    print("\n" + "=" * 70)
    print("ACCURACY BY BINDING DEPTH")
    print("=" * 70)
    
    by_depth_df = compute_by_binding_depth(df)
    
    for model in sorted(by_depth_df["model"].unique()):
        print(f"\n{model}:")
        model_data = by_depth_df[by_depth_df["model"] == model].sort_values("binding_depth")
        for _, row in model_data.iterrows():
            label = "main" if row["binding_depth"] == 0 else f"embed-{int(row['binding_depth'])}"
            bar = "█" * int(row["accuracy"] * 20) + "░" * (20 - int(row["accuracy"] * 20))
            print(f"  {label:8s}: {row['accuracy']:5.0%} {bar}")
    
    by_depth_df.to_csv(BINDING_RESULTS_DIR / "by_binding_depth.csv", index=False)
    
    # Error patterns
    print("\n" + "=" * 70)
    print("ERROR PATTERNS (Embedded Bindings Only)")
    print("=" * 70)
    
    error_df = analyze_error_patterns(df)
    if error_df is not None and len(error_df) > 0:
        print("\nWhen models fail on embedded bindings, what do they select?")
        for _, row in error_df.iterrows():
            print(f"\n{row['model']} ({row['total_errors']} errors on embedded):")
            print(f"  Main subject (first noun): {row['main_subject_pct']:.0%}")
            print(f"  Main object (recency):     {row['main_object_pct']:.0%}")
            print(f"  Wrong human (confusion):   {row['wrong_human_pct']:.0%}")
        
        error_df.to_csv(BINDING_RESULTS_DIR / "error_patterns.csv", index=False)
    else:
        print("\nNo/few errors on embedded bindings!")
    
    # Plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plot_binding_results(main_embed_df, by_depth_df, BINDING_RESULTS_DIR)
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    for _, row in test_df.iterrows():
        model = row["model"]
        diff = row["difference"]
        sig = row["significant"]
        
        print(f"\n{model}:")
        if diff > 0.2 and sig:
            print(f"  Main >> Embedded ({diff:+.0%} difference, p<0.05)")
            print(f"  → Evidence for PATTERN MATCHING (heuristics fail on embedded)")
        elif diff > 0.1:
            print(f"  Main > Embedded ({diff:+.0%} difference)")
            print(f"  → Possible pattern matching tendency")
        elif diff < 0.05:
            print(f"  Main ≈ Embedded ({diff:+.0%} difference)")
            print(f"  → Evidence for STRUCTURAL PARSING (recovers all bindings)")
        else:
            print(f"  Small difference ({diff:+.0%})")
            print(f"  → Inconclusive")
    
    print(f"\nResults saved to: {BINDING_RESULTS_DIR}")


if __name__ == "__main__":
    main()
"""
Analyze structural transfer experiment results.

Key questions:
1. Do models show similar performance across domains? (Transfer → Construction)
2. Or is performance domain-specific? (No transfer → Coverage)
3. What heuristics might explain the pattern?
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from config import ALPHA


TRANSFER_RESULTS_DIR = Path("results/transfer")


def load_results():
    """Load results as DataFrame."""
    results_path = TRANSFER_RESULTS_DIR / "results_all_models.json"
    
    if not results_path.exists():
        raise FileNotFoundError("No results. Run run_transfer_experiment.py first.")
    
    with open(results_path) as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    df = df[df["is_correct"].notna()].copy()
    df["is_correct"] = df["is_correct"].astype(bool)
    
    # Add simplified domain column
    df["domain_short"] = df["domain"].str[0]
    
    return df


def wilson_ci(n_success, n_total, alpha=0.05):
    """Wilson score CI."""
    if n_total == 0:
        return 0, 0, 0
    p = n_success / n_total
    z = stats.norm.ppf(1 - alpha/2)
    denom = 1 + z**2/n_total
    center = (p + z**2/(2*n_total)) / denom
    spread = z * np.sqrt(p*(1-p)/n_total + z**2/(4*n_total**2)) / denom
    return p, max(0, center-spread), min(1, center+spread)


def compute_accuracy_table(df):
    """Compute accuracy by model × domain × depth."""
    results = []
    
    for model in df["model_key"].unique():
        for domain in df["domain"].unique():
            for depth in sorted(df["depth"].unique()):
                subset = df[(df["model_key"]==model) & (df["domain"]==domain) & (df["depth"]==depth)]
                n = len(subset)
                n_correct = subset["is_correct"].sum()
                acc, ci_lo, ci_hi = wilson_ci(n_correct, n)
                
                results.append({
                    "model": model,
                    "domain": domain,
                    "domain_short": domain[0],
                    "depth": depth,
                    "n": n,
                    "n_correct": n_correct,
                    "accuracy": acc,
                    "ci_low": ci_lo,
                    "ci_high": ci_hi
                })
    
    return pd.DataFrame(results)


def test_domain_transfer(accuracy_df, model):
    """
    Test whether performance differs significantly across domains.
    
    If construction: domains should NOT differ significantly
    If coverage: domains SHOULD differ significantly
    """
    model_data = accuracy_df[accuracy_df["model"] == model]
    
    # Get accuracy by domain (averaged across depths)
    domain_accs = model_data.groupby("domain_short")["accuracy"].mean()
    
    # Chi-square test across domains at each depth
    results = []
    df_full = None  # We'll need the raw data
    
    return {
        "model": model,
        "domain_A_mean": domain_accs.get("A", 0),
        "domain_B_mean": domain_accs.get("B", 0),
        "domain_C_mean": domain_accs.get("C", 0),
        "max_diff": domain_accs.max() - domain_accs.min()
    }


def analyze_heuristics(df):
    """
    Test which heuristic best explains errors.
    
    Heuristic predictions:
    - "First noun": Succeeds on A & C, fails on B
    - "Last noun": Succeeds on B, fails on A & C  
    - "Structural": Succeeds on all
    """
    results = []
    
    for model in df["model_key"].unique():
        model_df = df[df["model_key"] == model]
        
        # Accuracy by domain (depth 3+ where it matters)
        deep = model_df[model_df["depth"] >= 3]
        
        acc_A = deep[deep["domain_short"]=="A"]["is_correct"].mean()
        acc_B = deep[deep["domain_short"]=="B"]["is_correct"].mean()
        acc_C = deep[deep["domain_short"]=="C"]["is_correct"].mean()
        
        # Determine pattern
        if acc_A > 0.7 and acc_B > 0.7 and acc_C > 0.7:
            pattern = "structural (all high)"
        elif acc_A > 0.7 and acc_C > 0.7 and acc_B < 0.5:
            pattern = "first_noun heuristic (A&C high, B low)"
        elif acc_B > 0.7 and acc_A < 0.5 and acc_C < 0.5:
            pattern = "last_noun heuristic (B high, A&C low)"
        elif acc_A > 0.7 and acc_B < 0.5 and acc_C < 0.5:
            pattern = "relative_clause specific (only A high)"
        else:
            pattern = "unclear/mixed"
        
        results.append({
            "model": model,
            "acc_A_deep": acc_A,
            "acc_B_deep": acc_B,
            "acc_C_deep": acc_C,
            "inferred_pattern": pattern
        })
    
    return pd.DataFrame(results)


def plot_transfer_results(accuracy_df, output_dir):
    """Generate transfer comparison plots."""
    sns.set_style("whitegrid")
    
    colors = {"haiku": "#E07A5F", "sonnet": "#8B5CF6", "gpt4o-mini": "#3D405B"}
    domain_names = {"A": "Relative Clauses", "B": "Possessives", "C": "Prepositions"}
    
    # Plot 1: Accuracy by depth, faceted by domain
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for idx, domain in enumerate(["A", "B", "C"]):
        ax = axes[idx]
        domain_data = accuracy_df[accuracy_df["domain_short"] == domain]
        
        for model in domain_data["model"].unique():
            model_data = domain_data[domain_data["model"]==model].sort_values("depth")
            ax.plot(model_data["depth"], model_data["accuracy"],
                    marker="o", linewidth=2, markersize=7,
                    color=colors.get(model, "gray"), label=model)
            ax.fill_between(model_data["depth"], model_data["ci_low"], model_data["ci_high"],
                           alpha=0.15, color=colors.get(model, "gray"))
        
        ax.axhline(y=0.2, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Depth")
        ax.set_title(f"Domain {domain}: {domain_names[domain]}")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(sorted(domain_data["depth"].unique()))
        
        if idx == 0:
            ax.set_ylabel("Accuracy")
            ax.legend(loc="lower left")
    
    plt.suptitle("Structural Transfer Test: Performance by Domain", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "transfer_by_domain.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'transfer_by_domain.png'}")
    plt.close()
    
    # Plot 2: Domain comparison at depth 4 (bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    depth_4 = accuracy_df[accuracy_df["depth"] == 4].copy()
    
    x = np.arange(3)  # 3 domains
    width = 0.25
    
    models = sorted(depth_4["model"].unique())
    for i, model in enumerate(models):
        model_data = depth_4[depth_4["model"] == model].sort_values("domain_short")
        ax.bar(x + i*width, model_data["accuracy"], width, 
               label=model, color=colors.get(model, "gray"),
               yerr=[model_data["accuracy"]-model_data["ci_low"], 
                     model_data["ci_high"]-model_data["accuracy"]],
               capsize=3)
    
    ax.set_xlabel("Domain")
    ax.set_ylabel("Accuracy")
    ax.set_title("Domain Comparison at Depth 4")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["A: Relative\nClauses", "B: Possessive\nChains", "C: Prepositional\nChains"])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.2, color="gray", linestyle="--", alpha=0.5, label="_chance")
    
    plt.tight_layout()
    plt.savefig(output_dir / "transfer_domain_comparison_d4.png", dpi=300)
    print(f"Saved: {output_dir / 'transfer_domain_comparison_d4.png'}")
    plt.close()


def main():
    """Run transfer analysis."""
    print("=" * 70)
    print("STRUCTURAL TRANSFER ANALYSIS")
    print("=" * 70)
    
    # Load
    print("\nLoading results...")
    df = load_results()
    print(f"Loaded {len(df)} valid trials")
    print(f"Models: {df['model_key'].unique().tolist()}")
    print(f"Domains: {df['domain'].unique().tolist()}")
    
    # Accuracy table
    print("\n" + "=" * 70)
    print("ACCURACY BY MODEL × DOMAIN × DEPTH")
    print("=" * 70)
    
    accuracy_df = compute_accuracy_table(df)
    
    for model in sorted(accuracy_df["model"].unique()):
        print(f"\n{'='*50}")
        print(f"MODEL: {model}")
        print(f"{'='*50}")
        
        model_data = accuracy_df[accuracy_df["model"] == model]
        
        for domain in ["A", "B", "C"]:
            domain_full = {"A": "Relative Clauses", "B": "Possessives", "C": "Prepositions"}[domain]
            print(f"\n  Domain {domain} ({domain_full}):")
            
            d_data = model_data[model_data["domain_short"] == domain].sort_values("depth")
            for _, row in d_data.iterrows():
                bar = "█" * int(row["accuracy"]*20) + "░" * (20-int(row["accuracy"]*20))
                print(f"    D{row['depth']}: {row['accuracy']:5.0%} {bar}")
    
    accuracy_df.to_csv(TRANSFER_RESULTS_DIR / "transfer_accuracy_table.csv", index=False)
    
    # Heuristic analysis
    print("\n" + "=" * 70)
    print("HEURISTIC ANALYSIS")
    print("=" * 70)
    print("""
    Predictions:
    - "First noun" heuristic: High on A & C, Low on B
    - "Last noun" heuristic: High on B, Low on A & C
    - Structural parsing: High on ALL domains
    """)
    
    heuristic_df = analyze_heuristics(df)
    print("\nResults (Depth 3+ accuracy):")
    print(heuristic_df.to_string(index=False))
    
    heuristic_df.to_csv(TRANSFER_RESULTS_DIR / "heuristic_analysis.csv", index=False)
    
    # Transfer test
    print("\n" + "=" * 70)
    print("TRANSFER TEST")
    print("=" * 70)
    print("""
    Question: Does performance transfer across domains?
    - Transfer (construction): Similar accuracy across A, B, C
    - No transfer (coverage): Domain-specific performance
    """)
    
    for model in sorted(accuracy_df["model"].unique()):
        result = test_domain_transfer(accuracy_df, model)
        print(f"\n{model}:")
        print(f"  Domain A (Relative Clauses): {result['domain_A_mean']:.0%}")
        print(f"  Domain B (Possessives):      {result['domain_B_mean']:.0%}")
        print(f"  Domain C (Prepositions):     {result['domain_C_mean']:.0%}")
        print(f"  Max difference: {result['max_diff']:.0%}")
        
        if result['max_diff'] < 0.15:
            print(f"  → TRANSFER (differences < 15%)")
        else:
            print(f"  → NO TRANSFER (differences ≥ 15%)")
    
    # Plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plot_transfer_results(accuracy_df, TRANSFER_RESULTS_DIR)
    
    # Final interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    # Determine overall pattern
    for _, row in heuristic_df.iterrows():
        model = row["model"]
        pattern = row["inferred_pattern"]
        
        print(f"\n{model}: {pattern}")
        
        if "structural" in pattern:
            print("  → Evidence for CONSTRUCTION (recursive capacity transfers)")
        elif "first_noun" in pattern:
            print("  → Evidence for HEURISTIC (using position, not structure)")
        elif "specific" in pattern:
            print("  → Evidence for COVERAGE (domain-specific training)")
        else:
            print("  → Inconclusive (mixed pattern)")
    
    print(f"\nResults saved to: {TRANSFER_RESULTS_DIR}")


if __name__ == "__main__":
    main()
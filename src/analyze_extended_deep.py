"""
Deep analysis of extended experiment results.

Additional analyses:
1. Curve fitting: Threshold collapse vs. gradual degradation
2. Error position analysis: What heuristic explains errors?
3. Cross-model agreement: Do models fail on same items?
4. Response time analysis: Does processing scale with depth?
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

RESULTS_DIR = Path("results/extended")


def load_results():
    """Load results."""
    with open(RESULTS_DIR / "results_all_models.json") as f:
        results = json.load(f)
    df = pd.DataFrame(results)
    df = df[df["is_correct"].notna()].copy()
    df["is_correct"] = df["is_correct"].astype(bool)
    return df


# =============================================================================
# 1. CURVE FITTING: Threshold vs Gradual
# =============================================================================

def sigmoid(x, L, k, x0):
    """Sigmoid function for threshold collapse."""
    return L / (1 + np.exp(k * (x - x0)))

def linear(x, a, b):
    """Linear function for gradual decline."""
    return a * x + b

def fit_degradation_curves(df):
    """
    Fit sigmoid (threshold) vs linear (gradual) to each model's curve.
    Compare fits using BIC.
    """
    print("=" * 70)
    print("CURVE FITTING: Threshold vs Gradual Degradation")
    print("=" * 70)
    print("""
    Sigmoid fit: Models threshold collapse (sudden drop)
    Linear fit: Models gradual degradation (steady decline)
    
    Lower BIC = better fit (penalizes complexity)
    """)
    
    results = []
    
    for model in df["model_key"].unique():
        model_df = df[df["model_key"] == model]
        
        # Get accuracy by depth
        acc_by_depth = model_df.groupby("depth")["is_correct"].mean()
        x = np.array(acc_by_depth.index, dtype=float)
        y = np.array(acc_by_depth.values)
        n = len(model_df)
        
        # Fit sigmoid
        try:
            # Initial guess: L=1 (max), k=1 (steepness), x0=3 (midpoint)
            popt_sig, _ = curve_fit(sigmoid, x, y, p0=[1, 1, 3], maxfev=5000,
                                     bounds=([0, 0, 0], [1.5, 10, 15]))
            y_pred_sig = sigmoid(x, *popt_sig)
            ss_res_sig = np.sum((y - y_pred_sig) ** 2)
            bic_sig = n * np.log(ss_res_sig / n) + 3 * np.log(n)  # 3 params
            sigmoid_success = True
        except:
            bic_sig = np.inf
            sigmoid_success = False
            popt_sig = [None, None, None]
        
        # Fit linear
        try:
            popt_lin, _ = curve_fit(linear, x, y)
            y_pred_lin = linear(x, *popt_lin)
            ss_res_lin = np.sum((y - y_pred_lin) ** 2)
            bic_lin = n * np.log(ss_res_lin / n) + 2 * np.log(n)  # 2 params
            linear_success = True
        except:
            bic_lin = np.inf
            linear_success = False
            popt_lin = [None, None]
        
        # Determine better fit
        if bic_sig < bic_lin:
            better_fit = "SIGMOID (threshold collapse)"
            delta_bic = bic_lin - bic_sig
        else:
            better_fit = "LINEAR (gradual degradation)"
            delta_bic = bic_sig - bic_lin
        
        print(f"\n{model}:")
        print(f"  Sigmoid BIC: {bic_sig:.1f}" + (" (failed)" if not sigmoid_success else f" (midpoint={popt_sig[2]:.1f})"))
        print(f"  Linear BIC:  {bic_lin:.1f}" + (" (failed)" if not linear_success else f" (slope={popt_lin[0]:.3f})"))
        print(f"  Better fit:  {better_fit} (ΔBIC={delta_bic:.1f})")
        
        results.append({
            "model": model,
            "bic_sigmoid": bic_sig,
            "bic_linear": bic_lin,
            "better_fit": "sigmoid" if bic_sig < bic_lin else "linear",
            "delta_bic": delta_bic,
            "sigmoid_midpoint": popt_sig[2] if sigmoid_success else None,
            "linear_slope": popt_lin[0] if linear_success else None
        })
    
    return pd.DataFrame(results)


# =============================================================================
# 2. ERROR POSITION ANALYSIS
# =============================================================================

def analyze_error_positions(df):
    """
    Analyze where in the sentence the selected (wrong) noun appears.
    
    This reveals what heuristic the model might be using:
    - Recency: Selects nouns near end of sentence
    - Primacy: Selects first nouns encountered
    - Middle: Gets confused in embedding structure
    """
    print("\n" + "=" * 70)
    print("ERROR POSITION ANALYSIS")
    print("=" * 70)
    
    errors = df[~df["is_correct"]].copy()
    
    if len(errors) == 0:
        print("No errors to analyze!")
        return
    
    for model in sorted(df["model_key"].unique()):
        model_errors = errors[errors["model_key"] == model]
        
        if len(model_errors) == 0:
            print(f"\n{model}: No errors")
            continue
        
        print(f"\n{model} ({len(model_errors)} errors):")
        
        # Analyze by error type and depth
        by_depth = model_errors.groupby(["depth", "error_type"]).size().unstack(fill_value=0)
        
        # Calculate proportions
        print("\n  Error type by depth:")
        print("  Depth | Object | Human | Other | Total")
        print("  " + "-" * 45)
        
        for depth in sorted(model_errors["depth"].unique()):
            depth_errors = model_errors[model_errors["depth"] == depth]
            n_obj = (depth_errors["error_type"] == "object_error").sum()
            n_hum = (depth_errors["error_type"] == "human_error").sum()
            n_oth = (depth_errors["error_type"] == "other_error").sum()
            total = len(depth_errors)
            
            if total > 0:
                print(f"  D{depth:2d}   | {n_obj:3d} ({n_obj/total:4.0%}) | {n_hum:3d} ({n_hum/total:4.0%}) | {n_oth:3d} | {total:3d}")
        
        # Overall pattern
        total_obj = (model_errors["error_type"] == "object_error").sum()
        total_hum = (model_errors["error_type"] == "human_error").sum()
        total = len(model_errors)
        
        print(f"\n  Overall: {total_obj/total:.0%} object errors, {total_hum/total:.0%} human errors")
        
        # Interpretation
        if total_obj / total > 0.6:
            print("  → Pattern suggests RECENCY heuristic (picks end of sentence)")
        elif total_hum / total > 0.6:
            print("  → Pattern suggests EMBEDDING CONFUSION (picks wrong agent)")
        else:
            print("  → Mixed pattern (no clear single heuristic)")


# =============================================================================
# 3. CROSS-MODEL AGREEMENT
# =============================================================================

def analyze_cross_model_agreement(df):
    """
    Do models fail on the same sentences?
    
    High agreement = sentences have intrinsic difficulty
    Low agreement = failures are model-specific
    """
    print("\n" + "=" * 70)
    print("CROSS-MODEL AGREEMENT ANALYSIS")
    print("=" * 70)
    
    # Pivot to get model × sentence accuracy
    pivot = df.pivot_table(
        index="trial_id", 
        columns="model_key", 
        values="is_correct",
        aggfunc="first"
    )
    
    models = list(pivot.columns)
    
    print("\nPairwise agreement (% of items where both correct OR both wrong):")
    print()
    
    agreement_matrix = {}
    
    for i, m1 in enumerate(models):
        agreement_matrix[m1] = {}
        for m2 in models:
            if m1 == m2:
                agreement_matrix[m1][m2] = 1.0
            else:
                # Agreement = both correct OR both wrong
                both_correct = (pivot[m1] == True) & (pivot[m2] == True)
                both_wrong = (pivot[m1] == False) & (pivot[m2] == False)
                agreement = (both_correct | both_wrong).mean()
                agreement_matrix[m1][m2] = agreement
    
    agreement_df = pd.DataFrame(agreement_matrix)
    print(agreement_df.round(2).to_string())
    
    # Analyze items that ALL models get wrong
    all_wrong = pivot[pivot.all(axis=1) == False]
    all_correct = pivot[pivot.all(axis=1) == True]
    
    print(f"\nItems ALL models got correct: {len(all_correct)} ({len(all_correct)/len(pivot):.0%})")
    print(f"Items ALL models got wrong: {len(all_wrong)} ({len(all_wrong)/len(pivot):.0%})")
    
    # What depths are the universally hard items?
    if len(all_wrong) > 0:
        wrong_ids = all_wrong.index.tolist()
        wrong_depths = df[df["trial_id"].isin(wrong_ids)]["depth"].value_counts().sort_index()
        print(f"\nUniversally wrong items by depth:")
        for depth, count in wrong_depths.items():
            print(f"  D{depth}: {count}")
    
    # Cohen's Kappa for each pair
    print("\nCohen's Kappa (chance-corrected agreement):")
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i < j:
                # Calculate kappa
                a = pivot[m1].astype(int)
                b = pivot[m2].astype(int)
                
                # Observed agreement
                po = ((a == b).sum()) / len(a)
                
                # Expected agreement
                p1 = a.mean()
                p2 = b.mean()
                pe = p1 * p2 + (1 - p1) * (1 - p2)
                
                kappa = (po - pe) / (1 - pe) if pe < 1 else 1
                
                print(f"  {m1} vs {m2}: κ = {kappa:.3f}")
    
    return agreement_df


# =============================================================================
# 4. RESPONSE TIME ANALYSIS
# =============================================================================

def analyze_response_times(df):
    """
    Does response time scale with depth?
    
    Construction: Time should increase with depth (building structure)
    Lookup: Time should be relatively flat (pattern matching)
    """
    print("\n" + "=" * 70)
    print("RESPONSE TIME ANALYSIS")
    print("=" * 70)
    
    if "response_time" not in df.columns:
        print("No response time data available.")
        return
    
    for model in sorted(df["model_key"].unique()):
        model_df = df[df["model_key"] == model]
        
        print(f"\n{model}:")
        
        # Mean response time by depth
        rt_by_depth = model_df.groupby("depth")["response_time"].agg(["mean", "std"])
        
        print("  Depth | Mean RT (s) | Std")
        print("  " + "-" * 30)
        for depth, row in rt_by_depth.iterrows():
            print(f"  D{depth:2d}   | {row['mean']:6.3f}      | {row['std']:.3f}")
        
        # Correlation between depth and RT
        corr, p = stats.pearsonr(model_df["depth"], model_df["response_time"])
        print(f"\n  Depth-RT correlation: r={corr:.3f}, p={p:.4f}")
        
        if corr > 0.3 and p < 0.05:
            print("  → Response time INCREASES with depth")
        elif corr < -0.3 and p < 0.05:
            print("  → Response time DECREASES with depth (unexpected)")
        else:
            print("  → No significant depth-RT relationship")
        
        # Does RT predict accuracy?
        correct_rt = model_df[model_df["is_correct"]]["response_time"].mean()
        wrong_rt = model_df[~model_df["is_correct"]]["response_time"].mean()
        
        print(f"\n  Mean RT (correct): {correct_rt:.3f}s")
        print(f"  Mean RT (wrong):   {wrong_rt:.3f}s")


# =============================================================================
# 5. VARIANCE ANALYSIS
# =============================================================================

def analyze_variance(df):
    """
    Is performance consistent at each depth, or highly variable?
    
    High variance at failure depths = confused/random
    Low variance = consistent (even if wrong) = systematic heuristic
    """
    print("\n" + "=" * 70)
    print("PERFORMANCE VARIANCE ANALYSIS")
    print("=" * 70)
    
    for model in sorted(df["model_key"].unique()):
        model_df = df[df["model_key"] == model]
        
        print(f"\n{model}:")
        print("  Depth | Accuracy | 95% CI Width | Interpretation")
        print("  " + "-" * 55)
        
        for depth in sorted(model_df["depth"].unique()):
            depth_df = model_df[model_df["depth"] == depth]
            acc = depth_df["is_correct"].mean()
            n = len(depth_df)
            
            # Wilson CI width
            z = 1.96
            p = acc
            denom = 1 + z**2/n
            spread = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
            ci_width = 2 * spread
            
            # Interpretation
            if acc > 0.9:
                interp = "ceiling"
            elif acc < 0.25:
                interp = "floor/chance"
            elif ci_width > 0.3:
                interp = "high variance"
            else:
                interp = "stable"
            
            print(f"  D{depth:2d}   | {acc:5.0%}    | {ci_width:5.0%}         | {interp}")


# =============================================================================
# 6. INDIVIDUAL ITEM DIFFICULTY
# =============================================================================

def analyze_item_difficulty(df):
    """
    Which specific sentences are hardest?
    What makes them hard?
    """
    print("\n" + "=" * 70)
    print("ITEM DIFFICULTY ANALYSIS")
    print("=" * 70)
    
    # Calculate difficulty per item (across all models)
    item_difficulty = df.groupby("trial_id").agg({
        "is_correct": "mean",
        "depth": "first",
        "sentence": "first"
    }).rename(columns={"is_correct": "accuracy"})
    
    # Hardest items (lowest accuracy)
    hardest = item_difficulty.nsmallest(10, "accuracy")
    
    print("\nTop 10 hardest items (lowest accuracy across models):")
    print("-" * 70)
    
    for trial_id, row in hardest.iterrows():
        print(f"\n{trial_id} (D{row['depth']}, acc={row['accuracy']:.0%}):")
        sent = row['sentence']
        if len(sent) > 80:
            print(f"  {sent[:80]}...")
        else:
            print(f"  {sent}")
    
    # Is difficulty just depth, or are some items harder than expected?
    print("\n\nDifficulty vs Depth:")
    depth_difficulty = item_difficulty.groupby("depth")["accuracy"].agg(["mean", "std", "min", "max"])
    print(depth_difficulty.round(2).to_string())


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("DEEP ANALYSIS: EXTENDED EXPERIMENT RESULTS")
    print("=" * 70)
    
    df = load_results()
    print(f"Loaded {len(df)} trials")
    
    # Run all analyses
    curve_results = fit_degradation_curves(df)
    curve_results.to_csv(RESULTS_DIR / "curve_fitting.csv", index=False)
    
    analyze_error_positions(df)
    
    agreement_df = analyze_cross_model_agreement(df)
    agreement_df.to_csv(RESULTS_DIR / "model_agreement.csv")
    
    analyze_response_times(df)
    
    analyze_variance(df)
    
    analyze_item_difficulty(df)
    
    # Final summary
    print("\n" + "=" * 70)
    print("DEEP ANALYSIS SUMMARY")
    print("=" * 70)
    
    print("\n1. CURVE FITTING:")
    for _, row in curve_results.iterrows():
        print(f"   {row['model']}: {row['better_fit'].upper()} fits better (ΔBIC={row['delta_bic']:.1f})")
    
    print("\n2. KEY FINDINGS:")
    print("   - See detailed output above for error patterns, agreement, timing")
    
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
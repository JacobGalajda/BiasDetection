"""
significance_tests.py
---------------------
Statistical significance testing for the bias detection results.

Tests performed:
  1. Mann-Whitney U -- LLMxLLM vs HumanxHuman LCS distributions
  2. Mann-Whitney U -- LLMxDuped vs HumanxHuman LCS (fine-tuning memorization)
  3. Mann-Whitney U -- LLMxTrain vs HumanxHuman LCS (broad corpus)
  4. Run-stratified test -- high-LCS runs vs low-LCS runs
  5. Effect sizes (rank-biserial correlation r)
  6. Bootstrap confidence intervals on mean LCS differences

All results are printed and saved to a JSON for the paper.

Usage:
    python scripts/significance_tests.py results/report_full_v2.json
    python scripts/significance_tests.py results/report_full_v2.json \
        --out results/significance/ \
        --alpha 0.05
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PAIR_TYPE_MAP = {
    ("llm_output",     "llm_output"):     "LLM x LLM",
    ("human_baseline", "human_baseline"): "Human x Human",
    ("llm_output",     "human_baseline"): "LLM x Human",
    ("human_baseline", "llm_output"):     "LLM x Human",
    ("llm_output",     "train_corpus"):   "LLM x Train",
    ("train_corpus",   "llm_output"):     "LLM x Train",
    ("llm_output",     "duped"):          "LLM x Duped",
    ("duped",          "llm_output"):     "LLM x Duped",
    ("human_baseline", "duped"):          "Human x Duped",
    ("duped",          "human_baseline"): "Human x Duped",
    ("duped",          "duped"):          "Duped x Duped",
    ("train_corpus",   "train_corpus"):   "Train x Train",
}


def pair_label(ga, gb):
    return PAIR_TYPE_MAP.get((ga.lower(), gb.lower()), f"{ga} x {gb}")


def load_lcs_by_type(report_path: str) -> dict:
    """Load LCS scores grouped by pair type. Skips zero-LCS pairs for
    cross-instrumentation groups where the filter was applied."""
    print(f"Loading {report_path} ...")
    with open(report_path) as f:
        report = json.load(f)

    buckets = defaultdict(list)
    for pair in report.get("pair_results", []):
        label = pair_label(pair["group_a"], pair["group_b"])
        buckets[label].append(pair["lcs_score"])

    print(f"Pair types loaded: {sorted(buckets)}")
    for label, scores in buckets.items():
        nz = sum(1 for s in scores if s > 0)
        print(f"  {label:<35} n={len(scores):>8,}  non-zero={nz:>6,}  "
              f"mean={np.mean(scores):.5f}  max={np.max(scores):.4f}")
    return buckets


def load_run_lcs(report_path: str) -> dict:
    """Extract within-run LCS scores for LLMxLLM pairs."""
    import re
    with open(report_path) as f:
        report = json.load(f)

    run_scores = defaultdict(list)
    for pair in report.get("pair_results", []):
        if pair["group_a"] != "llm_output" or pair["group_b"] != "llm_output":
            continue
        ma = re.search(r"_R(\d+)_", pair["piece_a"])
        mb = re.search(r"_R(\d+)_", pair["piece_b"])
        if ma and mb and ma.group(1) == mb.group(1):
            run_scores[f"R{ma.group(1)}"].append(pair["lcs_score"])
    return run_scores


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def mann_whitney(a: list, b: list, label_a: str, label_b: str,
                 alpha: float = 0.05) -> dict:
    """
    Two-sided Mann-Whitney U test with rank-biserial effect size.

    rank-biserial r = 2U / (n_a * n_b) - 1
    Interpretation: |r| < 0.1 negligible, 0.1-0.3 small,
                    0.3-0.5 medium, > 0.5 large
    """
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)

    if len(a_arr) < 2 or len(b_arr) < 2:
        return {"error": "insufficient data"}

    u_stat, p_val = stats.mannwhitneyu(a_arr, b_arr, alternative="two-sided")

    # Rank-biserial correlation
    n_a, n_b  = len(a_arr), len(b_arr)
    r         = (2 * u_stat) / (n_a * n_b) - 1

    # Effect size interpretation
    abs_r = abs(r)
    if abs_r < 0.1:
        effect = "negligible"
    elif abs_r < 0.3:
        effect = "small"
    elif abs_r < 0.5:
        effect = "medium"
    else:
        effect = "large"

    result = {
        "comparison":     f"{label_a} vs {label_b}",
        "n_a":            n_a,
        "n_b":            n_b,
        "mean_a":         float(np.mean(a_arr)),
        "mean_b":         float(np.mean(b_arr)),
        "median_a":       float(np.median(a_arr)),
        "median_b":       float(np.median(b_arr)),
        "u_statistic":    float(u_stat),
        "p_value":        float(p_val),
        "significant":    bool(p_val < alpha),
        "alpha":          alpha,
        "effect_r":       float(r),
        "effect_size":    effect,
        "direction":      "A > B" if np.mean(a_arr) > np.mean(b_arr) else "A < B",
    }
    return result


def bootstrap_mean_diff(a: list, b: list,
                         n_boot: int = 2000,
                         ci: float = 0.95) -> dict:
    """
    Bootstrap confidence interval on mean(A) - mean(B).
    Uses random subsampling if arrays are very large (>50k) for speed.
    """
    MAX_SAMPLE = 50_000
    rng = np.random.default_rng(42)

    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)

    if len(a_arr) > MAX_SAMPLE:
        a_arr = rng.choice(a_arr, MAX_SAMPLE, replace=False)
    if len(b_arr) > MAX_SAMPLE:
        b_arr = rng.choice(b_arr, MAX_SAMPLE, replace=False)

    diffs = []
    for _ in range(n_boot):
        boot_a = rng.choice(a_arr, len(a_arr), replace=True)
        boot_b = rng.choice(b_arr, len(b_arr), replace=True)
        diffs.append(float(np.mean(boot_a) - np.mean(boot_b)))

    lo = float(np.percentile(diffs, (1 - ci) / 2 * 100))
    hi = float(np.percentile(diffs, (1 + ci) / 2 * 100))
    observed = float(np.mean(a_arr) - np.mean(b_arr))

    return {
        "observed_diff": observed,
        f"ci_{int(ci*100)}_lo": lo,
        f"ci_{int(ci*100)}_hi": hi,
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


def kruskal_wallis(groups: dict, label: str) -> dict:
    """Kruskal-Wallis H test across multiple groups."""
    arrays = [np.array(v) for v in groups.values()]
    arrays = [a for a in arrays if len(a) >= 2]
    if len(arrays) < 2:
        return {"error": "need >=2 groups"}
    h, p = stats.kruskal(*arrays)
    return {
        "test":        "Kruskal-Wallis H",
        "comparison":  label,
        "n_groups":    len(arrays),
        "H_statistic": float(h),
        "p_value":     float(p),
        "significant": bool(p < 0.05),
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

W = 70

def _print_result(res: dict, title: str):
    print(f"\n  {title}")
    print(f"  {'-'*60}")
    if "error" in res:
        print(f"  ERROR: {res['error']}")
        return
    if "H_statistic" in res:
        print(f"  H={res['H_statistic']:.2f}  p={res['p_value']:.2e}  "
              f"significant={'YES' if res['significant'] else 'NO'}")
        return
    print(f"  {res['comparison']}")
    print(f"  n_A={res['n_a']:,}  mean_A={res['mean_a']:.5f}  "
          f"median_A={res['median_a']:.5f}")
    print(f"  n_B={res['n_b']:,}  mean_B={res['mean_b']:.5f}  "
          f"median_B={res['median_b']:.5f}")
    print(f"  U={res['u_statistic']:.0f}  p={res['p_value']:.2e}  "
          f"{'*** SIGNIFICANT ***' if res['significant'] else 'not significant'}")
    print(f"  Effect r={res['effect_r']:.3f}  ({res['effect_size']})  "
          f"direction: {res['direction']}")


def _print_bootstrap(bs: dict, label: str):
    print(f"  Bootstrap CI ({label}):  "
          f"diff={bs['observed_diff']:.5f}  "
          f"95% CI [{bs['ci_95_lo']:.5f}, {bs['ci_95_hi']:.5f}]  "
          f"{'excludes zero [ok]' if bs['ci_excludes_zero'] else 'includes zero'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", help="Path to report JSON")
    parser.add_argument("--out",   default="results/significance",
                        help="Output directory (default: results/significance)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level (default: 0.05)")
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="Bootstrap iterations (default: 2000)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets   = load_lcs_by_type(args.report)
    run_lcs   = load_run_lcs(args.report)

    null      = buckets.get("Human x Human", [])
    llm_self  = buckets.get("LLM x LLM",    [])
    llm_duped = buckets.get("LLM x Duped",  [])
    llm_train = buckets.get("LLM x Train",  [])
    ft_self   = buckets.get("Duped x Duped",[])

    all_results = {}

    print("\n" + "=" * W)
    print("  STATISTICAL SIGNIFICANCE TESTS")
    print(f"  alpha = {args.alpha}   bootstrap n = {args.n_boot}")
    print("=" * W)

    # -- Test 1: LLM self-similarity vs Human null -------------------------
    print("\n-- 1. Intra-model self-similarity --")
    t1 = mann_whitney(llm_self, null,
                      "LLMxLLM", "HumanxHuman", args.alpha)
    _print_result(t1, "LLMxLLM vs HumanxHuman LCS")
    b1 = bootstrap_mean_diff(llm_self, null, args.n_boot)
    _print_bootstrap(b1, "LLMxLLM - HumanxHuman")
    all_results["llm_self_vs_null"] = {**t1, "bootstrap": b1}

    # -- Test 2: Fine-tuning memorization ---------------------------------
    print("\n-- 2. Fine-tuning corpus memorization --")
    t2 = mann_whitney(llm_duped, null,
                      "LLMxDuped", "HumanxHuman", args.alpha)
    _print_result(t2, "LLMxDuped (fine-tune) vs HumanxHuman LCS")
    b2 = bootstrap_mean_diff(llm_duped, null, args.n_boot)
    _print_bootstrap(b2, "LLMxDuped - HumanxHuman")
    all_results["llm_duped_vs_null"] = {**t2, "bootstrap": b2}

    # -- Test 3: Broad corpus (negative control) ---------------------------
    print("\n-- 3. Broad training corpus (negative control) --")
    t3 = mann_whitney(llm_train, null,
                      "LLMxTrain", "HumanxHuman", args.alpha)
    _print_result(t3, "LLMxTrain vs HumanxHuman LCS")
    b3 = bootstrap_mean_diff(llm_train, null, args.n_boot)
    _print_bootstrap(b3, "LLMxTrain - HumanxHuman")
    all_results["llm_train_vs_null"] = {**t3, "bootstrap": b3}

    # -- Test 4: Fine-tune vs broad corpus ---------------------------------
    print("\n-- 4. Fine-tune corpus vs broad corpus --")
    if llm_duped and llm_train:
        t4 = mann_whitney(llm_duped, llm_train,
                          "LLMxDuped", "LLMxTrain", args.alpha)
        _print_result(t4, "LLMxDuped vs LLMxTrain")
        all_results["llm_duped_vs_train"] = t4

    # -- Test 5: Run-stratified analysis -----------------------------------
    print("\n-- 5. Run-stratified self-similarity --")
    THRESHOLD = 0.05   # runs above this mean are "high-similarity"
    high_runs = {r: v for r, v in run_lcs.items()
                 if np.mean(v) >= THRESHOLD and len(v) >= 3}
    low_runs  = {r: v for r, v in run_lcs.items()
                 if np.mean(v) < THRESHOLD and len(v) >= 3}

    high_scores = [s for v in high_runs.values() for s in v]
    low_scores  = [s for v in low_runs.values()  for s in v]

    print(f"  High-LCS runs (mean>={THRESHOLD}): "
          f"{sorted(high_runs)} "
          f"(n_pairs={len(high_scores):,})")
    print(f"  Low-LCS  runs (mean<{THRESHOLD}): "
          f"{sorted(low_runs)[:8]}... "
          f"(n_pairs={len(low_scores):,})")

    if high_scores and low_scores:
        t5 = mann_whitney(high_scores, low_scores,
                          "High-LCS runs", "Low-LCS runs", args.alpha)
        _print_result(t5, "High-LCS runs vs Low-LCS runs")
        b5 = bootstrap_mean_diff(high_scores, low_scores, args.n_boot)
        _print_bootstrap(b5, "High - Low runs")

        # Also test high-LCS runs vs human null
        t5b = mann_whitney(high_scores, null,
                           "High-LCS runs", "HumanxHuman", args.alpha)
        _print_result(t5b, "High-LCS runs vs HumanxHuman null")
        all_results["run_stratified"] = {
            "high_vs_low":  {**t5, "bootstrap": b5},
            "high_vs_null": t5b,
            "high_runs":    sorted(high_runs),
            "low_runs":     sorted(low_runs),
            "threshold":    THRESHOLD,
        }

    # -- Test 6: Kruskal-Wallis across all groups --------------------------
    print("\n-- 6. Overall group difference (Kruskal-Wallis) --")
    kw_groups = {
        "LLMxLLM":     llm_self,
        "LLMxDuped":   llm_duped,
        "LLMxTrain":   llm_train,
        "HumanxHuman": null,
    }
    kw = kruskal_wallis(kw_groups, "All groups")
    _print_result(kw, "Kruskal-Wallis across LLMxLLM, LLMxDuped, LLMxTrain, HxH")
    all_results["kruskal_wallis"] = kw

    # -- Summary for paper -------------------------------------------------
    print("\n" + "=" * W)
    print("  SUMMARY FOR PAPER")
    print("=" * W)

    def _sig(res):
        if "error" in res:
            return "N/A"
        p = res.get("p_value", 1.0)
        if p < 0.001: return f"p<0.001 ***"
        if p < 0.01:  return f"p={p:.3f} **"
        if p < 0.05:  return f"p={p:.3f} *"
        return f"p={p:.3f} n.s."

    rows = [
        ("LLMxLLM > HumanxHuman",
         all_results.get("llm_self_vs_null", {})),
        ("LLMxDuped > HumanxHuman",
         all_results.get("llm_duped_vs_null", {})),
        ("LLMxTrain vs HumanxHuman",
         all_results.get("llm_train_vs_null", {})),
        ("LLMxDuped > LLMxTrain",
         all_results.get("llm_duped_vs_train", {})),
    ]

    print(f"\n  {'Hypothesis':<35} {'Result':>12}  {'Effect':>10}  {'Sig':>14}")
    print("  " + "-" * 66)
    for hyp, res in rows:
        if not res or "error" in res:
            print(f"  {hyp:<35} {'no data':>12}")
            continue
        sig = "[ok]" if res.get("significant") else "[rm]"
        r   = res.get("effect_r", float("nan"))
        eff = res.get("effect_size", "")
        print(f"  {hyp:<35} {sig:>12}  "
              f"r={r:+.3f} {eff:<8}  {_sig(res):>14}")

    print()

    # -- Save --------------------------------------------------------------
    out_path = out_dir / "significance_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved -> {out_path}")


if __name__ == "__main__":
    main()

"""
significance_tests_extended.py
================================
Extends the original significance_tests.py to cover all three metric families:
  - Best-window LCS (absolute-pitch)
  - TF-IDF pairwise cosine distance
  - Jaccard n-gram overlap (n=5)

All tests use Mann-Whitney U (two-sided, alpha=0.05).
Effect sizes reported as rank-biserial correlation r.
Bootstrap 95% CIs on mean difference (n_boot=2000).

Usage
-----
    python significance_tests_extended.py \
        --report results/report_full_v2.json \
        --out    results/significance_extended.json

The report JSON must contain per-pair records with keys:
    lcs_score, tfidf_cosine_dist, jaccard_n5, pair_type

pair_type values expected:
    "LLM x LLM", "LLM x Duped", "LLM x Train", "Human x Human"
"""

import argparse
import collections
import json
import numpy as np
from scipy.stats import mannwhitneyu, kruskal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALPHA = 0.05
N_BOOT = 2000
SUBSAMPLE_LIMIT = 50_000   # cap large groups before bootstrapping
RNG_SEED = 42

EFFECT_THRESHOLDS = [
    (0.1,  "negligible"),
    (0.3,  "small"),
    (0.5,  "medium"),
    (1.0,  "large"),
]

PRIMARY_PAIRS = ["LLM x LLM", "LLM x Duped", "LLM x Train", "Human x Human"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def effect_label(r: float) -> str:
    for threshold, label in EFFECT_THRESHOLDS:
        if abs(r) < threshold:
            return label
    return "large"


def rank_biserial(u: float, n_a: int, n_b: int) -> float:
    return 2 * u / (n_a * n_b) - 1


def bootstrap_ci(a: np.ndarray, b: np.ndarray,
                 n_boot: int = N_BOOT, seed: int = RNG_SEED):
    rng = np.random.default_rng(seed)
    obs_diff = a.mean() - b.mean()
    diffs = []
    for _ in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs.append(sa.mean() - sb.mean())
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "observed_diff": float(obs_diff),
        "ci_95_lo":      float(lo),
        "ci_95_hi":      float(hi),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


def subsample(arr: np.ndarray, limit: int, seed: int = RNG_SEED) -> np.ndarray:
    if len(arr) <= limit:
        return arr
    rng = np.random.default_rng(seed)
    return rng.choice(arr, size=limit, replace=False)


def mw_test(a: np.ndarray, b: np.ndarray,
            label_a: str, label_b: str,
            do_bootstrap: bool = True) -> dict:
    """Run Mann-Whitney U and return a result dict."""
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    r = rank_biserial(u, len(a), len(b))
    result = {
        "comparison":   f"{label_a} vs {label_b}",
        "n_a":          len(a),
        "n_b":          len(b),
        "mean_a":       float(a.mean()),
        "mean_b":       float(b.mean()),
        "median_a":     float(np.median(a)),
        "median_b":     float(np.median(b)),
        "u_statistic":  float(u),
        "p_value":      float(p),
        "significant":  bool(p < ALPHA),
        "alpha":        ALPHA,
        "effect_r":     float(r),
        "effect_size":  effect_label(r),
        "direction":    "A > B" if a.mean() > b.mean() else "B > A",
    }
    if do_bootstrap:
        a_s = subsample(a, SUBSAMPLE_LIMIT)
        b_s = subsample(b, SUBSAMPLE_LIMIT)
        result["bootstrap"] = bootstrap_ci(a_s, b_s)
    return result


def kruskal_test(groups: dict) -> dict:
    arrays = list(groups.values())
    labels = list(groups.keys())
    h, p = kruskal(*arrays)
    return {
        "test":       "Kruskal-Wallis H",
        "groups":     labels,
        "n_groups":   len(arrays),
        "H_statistic": float(h),
        "p_value":    float(p),
        "significant": bool(p < ALPHA),
    }


# ---------------------------------------------------------------------------
# Load pairs from report JSON
# ---------------------------------------------------------------------------

def derive_pair_type(group_a: str, group_b: str) -> str:
    """
    Map (group_a, group_b) from CorpusGroup enum values to the canonical
    pair_type labels used throughout this study.

    CorpusGroup values (from pipeline.py):
        "llm_output", "train_corpus", "duped", "human_baseline"
    """
    a = group_a.lower().strip()
    b = group_b.lower().strip()
    key = frozenset([a, b])
    pair = (a, b)

    # self-pairs
    if a == b:
        if a == "llm_output":      return "LLM x LLM"
        if a == "human_baseline":  return "Human x Human"
        if a == "duped":           return "LLM x Duped"   # duped self = fine-tune proxy
        if a == "train_corpus":    return "LLM x Train"   # train self

    # cross-pairs -- canonical label has LLM first when applicable
    if key == frozenset(["llm_output", "human_baseline"]): return "LLM x Human"
    if key == frozenset(["llm_output", "duped"]):          return "LLM x Duped"
    if key == frozenset(["llm_output", "train_corpus"]):   return "LLM x Train"
    if key == frozenset(["human_baseline", "duped"]):      return "Human x Duped"
    if key == frozenset(["human_baseline", "train_corpus"]):return "Human x Train"
    if key == frozenset(["duped", "train_corpus"]):        return "Duped x Train"

    # fallback: alphabetical
    return f"{a} x {b}"


def load_scores(report_path: str):
    """
    Parse report JSON produced by pipeline.py AnalysisReport.save().

    Expected structure:
        {
          "pair_results": [
            {
              "group_a": "llm_output",
              "group_b": "human_baseline",
              "lcs_score": 0.012,
              "tfidf_cosine_dist": 0.87,
              "ngram_scores": {"jaccard_n5": 0.0, "jaccard_n3": 0.0, ...},
              ...
            },
            ...
          ],
          ...
        }
    """
    print(f"  Reading {report_path} ...")
    with open(report_path) as f:
        report = json.load(f)

    if "pair_results" not in report:
        raise KeyError(
            f"Key 'pair_results' not found in {report_path}. "
            f"Top-level keys: {list(report.keys())}"
        )

    raw = report["pair_results"]
    print(f"  Found {len(raw):,} raw pair records.")

    scores = {p: {"lcs": [], "tfidf": [], "jaccard_n5": []}
              for p in PRIMARY_PAIRS}

    type_counter = collections.Counter()
    skipped = 0

    for rec in raw:
        ga = rec.get("group_a", "")
        gb = rec.get("group_b", "")
        pt = derive_pair_type(ga, gb)
        type_counter[pt] += 1

        if pt not in scores:
            skipped += 1
            continue

        scores[pt]["lcs"].append(rec.get("lcs_score", 0.0))

        td = rec.get("tfidf_cosine_dist")
        scores[pt]["tfidf"].append(td if td is not None else np.nan)

        ng = rec.get("ngram_scores") or {}
        # Support both "jaccard_n5" and bare "jaccard" keys
        j5 = ng.get("jaccard_n5") or ng.get("jaccard") or rec.get("jaccard_n5", 0.0)
        scores[pt]["jaccard_n5"].append(j5)

    print(f"  Pair type distribution:")
    for pt, n in sorted(type_counter.items(), key=lambda x: -x[1]):
        marker = "  [ok]" if pt in scores else "  (skipped)"
        print(f"    {pt:<30} {n:>8,}{marker}")

    if skipped:
        print(f"  [info] {skipped:,} pairs mapped to non-primary types -- excluded from tests.")

    # Convert to numpy, drop NaN for tfidf
    out = {}
    for pt, d in scores.items():
        out[pt] = {
            "lcs":        np.array(d["lcs"],        dtype=float),
            "tfidf":      np.array([x for x in d["tfidf"] if not np.isnan(x)], dtype=float),
            "jaccard_n5": np.array(d["jaccard_n5"], dtype=float),
        }
    return out


# ---------------------------------------------------------------------------
# Run all tests for a given metric
# ---------------------------------------------------------------------------

def run_metric_tests(scores: dict, metric: str) -> dict:
    null = scores["Human x Human"][metric]
    results = {}

    # --- global Kruskal-Wallis ---
    results["kruskal_wallis"] = kruskal_test({
        pt: scores[pt][metric] for pt in PRIMARY_PAIRS
    })

    # --- pairwise vs null ---
    for pt in ["LLM x LLM", "LLM x Duped", "LLM x Train"]:
        key = pt.lower().replace(" x ", "_vs_")
        results[f"{key}_vs_null"] = mw_test(
            scores[pt][metric], null,
            pt, "Human x Human",
        )

    # --- LLM x Duped vs LLM x Train (fine-tuning signal) ---
    results["llm_duped_vs_train"] = mw_test(
        scores["LLM x Duped"][metric],
        scores["LLM x Train"][metric],
        "LLM x Duped", "LLM x Train",
        do_bootstrap=False,
    )

    # --- LLM x LLM vs LLM x Duped (self-similarity vs fine-tune) ---
    results["llm_self_vs_duped"] = mw_test(
        scores["LLM x LLM"][metric],
        scores["LLM x Duped"][metric],
        "LLM x LLM", "LLM x Duped",
        do_bootstrap=False,
    )

    return results


# ---------------------------------------------------------------------------
# Note on TF-IDF directionality
# ---------------------------------------------------------------------------
# TF-IDF cosine DISTANCE: higher = more different.
# For LCS and Jaccard: higher = more similar (flag if high).
# For TF-IDF: lower distance = more similar (flag if low).
# The Mann-Whitney direction ("A > B") means A has HIGHER values.
# For TF-IDF: "A > B" means A is MORE DIFFERENT from B, i.e. less similar.
# The null (HxH, 0.518) has the LOWEST TF-IDF distance -- most homogeneous.
# LLMxLLM (0.702) > HxH (0.518): LLM outputs are MORE spread out than
# MAESTRO pairs, which makes sense (multi-key vs single-instrument).
# The key comparison is LLMxDuped (0.778) < LLMxTrain (0.843):
# LLM outputs are CLOSER to the fine-tune proxy than the broad corpus.


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True,
                    help="Path to pipeline report JSON (must contain raw pairs)")
    ap.add_argument("--out", default="significance_extended.json",
                    help="Output path for results JSON")
    args = ap.parse_args()

    print(f"Loading scores from {args.report} ...")
    scores = load_scores(args.report)

    for pt in PRIMARY_PAIRS:
        n = len(scores[pt]["lcs"])
        print(f"  {pt}: {n:,} pairs")

    output = {}

    for metric in ("lcs", "tfidf", "jaccard_n5"):
        print(f"\nRunning tests for metric: {metric}")
        output[metric] = run_metric_tests(scores, metric)

        # Quick summary print
        for test_key, res in output[metric].items():
            if test_key == "kruskal_wallis":
                print(f"  Kruskal-Wallis H={res['H_statistic']:.1f}, "
                      f"p={res['p_value']:.2e}, sig={res['significant']}")
            else:
                print(f"  {res['comparison']}: "
                      f"p={res['p_value']:.2e}, "
                      f"r={res['effect_r']:.4f} ({res['effect_size']}), "
                      f"sig={res['significant']}")

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {args.out}")
    _print_summary(output)


def _print_summary(output: dict):
    print("\n" + "="*65)
    print("SUMMARY -- effect sizes across metrics")
    print("="*65)
    print(f"{'Test':<35} {'LCS r':>8} {'TF-IDF r':>10} {'Jac r':>8}")
    print("-"*65)

    test_keys = [
        "llm_vs_llm_vs_null",
        "llm_vs_duped_vs_null",
        "llm_vs_train_vs_null",
        "llm_duped_vs_train",
        "llm_self_vs_duped",
    ]
    labels = [
        "LLMxLLM vs HxH",
        "LLMxDuped vs HxH",
        "LLMxTrain vs HxH",
        "LLMxDuped vs LLMxTrain",
        "LLMxLLM vs LLMxDuped",
    ]

    for key, label in zip(test_keys, labels):
        row = []
        for metric in ("lcs", "tfidf", "jaccard_n5"):
            res = output[metric].get(key, {})
            r = res.get("effect_r", float("nan"))
            row.append(f"{r:>8.4f}")
        print(f"{label:<35} {'  '.join(row)}")


if __name__ == "__main__":
    main()
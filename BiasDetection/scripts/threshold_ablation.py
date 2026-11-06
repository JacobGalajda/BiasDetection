"""
threshold_ablation.py
---------------------
Sweeps LCS and n-gram flag thresholds across a range and reports how
flag rates change per pair type. Demonstrates that the core findings
are robust (or shows where they break down).

Produces:
  1. A table of flag rates at each threshold value
  2. A sensitivity plot -- flag rate vs threshold for each pair type
  3. An annotated JSON with the full ablation results

Usage:
    python scripts/threshold_ablation.py results/report_full_v2.json
    python scripts/threshold_ablation.py results/report_full_v2.json \
        --out results/ablation/ \
        --lcs-range 0.15 0.50 --steps 15
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Pair type bucketing
# ---------------------------------------------------------------------------

PAIR_TYPE_MAP = {
    ("llm_output",     "llm_output"):     "LLM x LLM",
    ("human_baseline", "human_baseline"): "Human x Human",
    ("llm_output",     "human_baseline"): "LLM x Human",
    ("human_baseline", "llm_output"):     "LLM x Human",
    ("llm_output",     "train_corpus"):   "LLM x Train",
    ("train_corpus",   "llm_output"):     "LLM x Train",
    ("human_baseline", "train_corpus"):   "Human x Train",
    ("train_corpus",   "human_baseline"): "Human x Train",
    ("llm_output",     "duped"):          "LLM x SQ Human",
    ("duped",          "llm_output"):     "LLM x SQ Human",
    ("human_baseline", "duped"):          "MAESTRO x SQ Human",
    ("duped",          "human_baseline"): "MAESTRO x SQ Human",
    ("duped",          "duped"):          "SQ Human x SQ Human",
    ("duped",          "train_corpus"):   "SQ Human x Train",
    ("train_corpus",   "duped"):          "SQ Human x Train",
}

COLORS = {
    "LLM x LLM":           "#e74c3c",
    "Human x Human":        "#2ecc71",
    "LLM x Human":          "#e09a30",
    "LLM x Train":          "#9b59b6",
    "Human x Train":        "#1abc9c",
    "LLM x SQ Human":       "#e67e22",
    "MAESTRO x SQ Human":   "#3498db",
    "SQ Human x SQ Human":  "#f39c12",
    "SQ Human x Train":     "#16a085",
}

# The pair types that matter most for the paper
KEY_PAIRS = ["LLM x LLM", "Human x Human", "LLM x Train",
             "LLM x SQ Human", "SQ Human x SQ Human"]


def pair_label(ga: str, gb: str) -> str:
    return PAIR_TYPE_MAP.get((ga.lower(), gb.lower()),
                             f"{ga} x {gb}")


# ---------------------------------------------------------------------------
# Load and bucket pairs
# ---------------------------------------------------------------------------

def load_pairs(report_path: str) -> dict:
    with open(report_path) as f:
        report = json.load(f)
    buckets = defaultdict(list)
    for pair in report.get("pair_results", []):
        label = pair_label(pair["group_a"], pair["group_b"])
        buckets[label].append(pair)
    return buckets


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep_lcs(buckets: dict,
              thresholds: np.ndarray) -> dict:
    """
    For each threshold T, compute flag rate per pair type.
    Returns {pair_type: [flag_rate_at_T0, flag_rate_at_T1, ...]}
    """
    results = {}
    for label, pairs in buckets.items():
        rates = []
        lcs_scores = np.array([p["lcs_score"] for p in pairs],
                               dtype=np.float32)
        for T in thresholds:
            rates.append(float(np.mean(lcs_scores >= T)))
        results[label] = rates
    return results


def sweep_ngram(buckets: dict,
                thresholds: np.ndarray,
                n: int = 5) -> dict:
    key = f"jaccard_n{n}"
    results = {}
    for label, pairs in buckets.items():
        rates = []
        scores = np.array(
            [p["ngram_scores"].get(key, 0.0) for p in pairs],
            dtype=np.float32,
        )
        for T in thresholds:
            rates.append(float(np.mean(scores >= T)))
        results[label] = rates
    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_sensitivity(thresholds: np.ndarray,
                     sweep_results: dict,
                     calibrated_threshold: float,
                     metric_name: str,
                     out_path: str,
                     key_pairs: list = None):
    """Line plot of flag rate vs threshold for each pair type."""
    key_pairs = key_pairs or KEY_PAIRS

    fig, ax = plt.subplots(figsize=(10, 5))

    for label, rates in sweep_results.items():
        if label not in key_pairs:
            continue
        color = COLORS.get(label, "#888")
        ax.plot(thresholds, rates, label=label, color=color,
                linewidth=2.0, alpha=0.85)
        # Mark the rate at the calibrated threshold
        idx = np.argmin(np.abs(thresholds - calibrated_threshold))
        ax.scatter([thresholds[idx]], [rates[idx]],
                   color=color, s=80, zorder=5)

    ax.axvline(calibrated_threshold, color="black", linestyle="--",
               linewidth=1.2,
               label=f"Calibrated threshold ({calibrated_threshold:.3f})")

    # Shade the "robust region" -- where conclusions don't change
    ax.set_xlabel(f"{metric_name} threshold", fontsize=11)
    ax.set_ylabel("Flag rate", fontsize=11)
    ax.set_title(f"Threshold Sensitivity -- {metric_name}", fontsize=12)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_table(thresholds: np.ndarray,
                sweep_results: dict,
                calibrated_threshold: float,
                metric_name: str):
    """Print a condensed flag-rate table to stdout."""
    # Show only a subset of thresholds for readability
    step = max(1, len(thresholds) // 8)
    show_idx = list(range(0, len(thresholds), step))
    # Always include the calibrated threshold index
    cal_idx = int(np.argmin(np.abs(thresholds - calibrated_threshold)))
    if cal_idx not in show_idx:
        show_idx.append(cal_idx)
    show_idx = sorted(show_idx)
    show_T   = thresholds[show_idx]

    print(f"\n-- {metric_name} threshold sensitivity --")
    header = f"{'Pair type':<28}" + "".join(
        f"  {T:.3f}{'*' if abs(T - calibrated_threshold) < 1e-4 else ' ':1s}"
        for T in show_T
    )
    print(header)
    print("-" * len(header))

    for label in KEY_PAIRS:
        if label not in sweep_results:
            continue
        rates = sweep_results[label]
        row = f"{label:<28}" + "".join(
            f"  {rates[i]:>5.1%} "
            for i in show_idx
        )
        print(row)
    print("  * = calibrated threshold")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", help="Path to report JSON")
    parser.add_argument("--out", default="results/ablation",
                        help="Output directory (default: results/ablation)")
    parser.add_argument("--lcs-range", nargs=2, type=float,
                        default=[0.10, 0.55],
                        metavar=("MIN", "MAX"),
                        help="LCS threshold sweep range (default: 0.10 0.55)")
    parser.add_argument("--ngram-range", nargs=2, type=float,
                        default=[0.001, 0.20],
                        metavar=("MIN", "MAX"),
                        help="N-gram threshold range (default: 0.001 0.20)")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of threshold steps (default: 20)")
    parser.add_argument("--lcs-calibrated", type=float, default=0.377,
                        help="Calibrated LCS threshold to mark (default: 0.377)")
    parser.add_argument("--ngram-calibrated", type=float, default=0.0125,
                        help="Calibrated n-gram threshold (default: 0.0125)")
    parser.add_argument("--save-json", default=None,
                        help="Save full ablation results to JSON")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.report}")
    buckets = load_pairs(args.report)
    print(f"Pair types found: {sorted(buckets)}")
    print(f"Total pairs: {sum(len(v) for v in buckets.values()):,}")

    # LCS sweep
    lcs_T   = np.linspace(args.lcs_range[0], args.lcs_range[1], args.steps)
    lcs_res = sweep_lcs(buckets, lcs_T)
    print_table(lcs_T, lcs_res, args.lcs_calibrated, "LCS")
    plot_sensitivity(lcs_T, lcs_res, args.lcs_calibrated, "LCS",
                     str(out_dir / "sensitivity_lcs.png"))

    # N-gram sweep
    ng_T   = np.linspace(args.ngram_range[0], args.ngram_range[1], args.steps)
    ng_res = sweep_ngram(buckets, ng_T, n=5)
    print_table(ng_T, ng_res, args.ngram_calibrated, "Jaccard n=5")
    plot_sensitivity(ng_T, ng_res, args.ngram_calibrated, "Jaccard n=5",
                     str(out_dir / "sensitivity_ngram.png"))

    # Key finding summary
    print("\n-- Robustness summary --")
    print("  Does LLM x Train stay below Human x Human across the full range?")
    for i, T in enumerate(lcs_T):
        llm_train  = lcs_res.get("LLM x Train",  [0]*len(lcs_T))[i]
        human_hum  = lcs_res.get("Human x Human", [0]*len(lcs_T))[i]
        sq_sq      = lcs_res.get("SQ Human x SQ Human", [0]*len(lcs_T))[i]
        marker     = " <- calibrated" if abs(T - args.lcs_calibrated) < 1e-4 else ""
        status     = "YES" if llm_train <= human_hum else "NO "
        print(f"  T={T:.3f}  LLMxTrain={llm_train:.1%}  "
              f"HxH={human_hum:.1%}  SQxSQ={sq_sq:.1%}  "
              f"robust={status}{marker}")

    if args.save_json:
        out = {
            "lcs_thresholds":   lcs_T.tolist(),
            "lcs_flag_rates":   lcs_res,
            "ngram_thresholds": ng_T.tolist(),
            "ngram_flag_rates": ng_res,
            "calibrated_lcs":   args.lcs_calibrated,
            "calibrated_ngram": args.ngram_calibrated,
        }
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nAblation JSON saved: {args.save_json}")


if __name__ == "__main__":
    main()

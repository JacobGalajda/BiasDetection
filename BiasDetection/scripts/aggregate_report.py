"""
aggregate_report.py
-------------------
Aggregates one or more bias-detection report JSON files produced by
music_bias_analyzer.pipeline.AnalysisReport.save() and answers:

  1. How often does each pair type get flagged?            (flag rate table)
  2. What do the score distributions look like per group?  (box plots)
  3. How do LLM self-copies compare to human self-copies?  (LCS histogram)
  4. Is there a style-collapse signal?                     (score heatmap)
  5. Per-run breakdown for LLM output                      (if R0/R1/... in names)

Usage
-----
    # Single report, figures saved to results/figures_agg/
    python aggregate_report.py results/report.json

    # Merge two reports
    python aggregate_report.py results/report.json results/report_fast.json \\
        --out results/figures_agg/ \\
        --save-json results/agg_summary.json

    # Summary only (no figures)
    python aggregate_report.py results/report.json --no-figures
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants -- mirror pipeline.py / README thresholds
# ---------------------------------------------------------------------------

_PAIR_TYPE_MAP = {
    ("llm_output",     "llm_output"):     "LLM x LLM",
    ("human_baseline", "human_baseline"): "Human x Human",
    ("llm_output",     "human_baseline"): "LLM x Human",
    ("human_baseline", "llm_output"):     "LLM x Human",
    ("llm_output",     "train_corpus"):   "LLM x Train",
    ("train_corpus",   "llm_output"):     "LLM x Train",
    ("human_baseline", "train_corpus"):   "Human x Train",
    ("train_corpus",   "human_baseline"): "Human x Train",
    ("llm_output",     "duped"):          "LLM x Duped",
    ("duped",          "llm_output"):     "LLM x Duped",
}

PAIR_TYPE_ORDER = [
    "LLM x LLM",
    "Human x Human",
    "LLM x Human",
    "LLM x Train",
    "Human x Train",
    "LLM x Duped",
]

COLORS = {
    "LLM x LLM":     "#e05252",
    "Human x Human": "#4a9ede",
    "LLM x Human":   "#e09a30",
    "LLM x Train":   "#a855f7",
    "Human x Train": "#22c55e",
    "LLM x Duped":   "#ef4444",
}

# From README / PipelineConfig defaults
THRESHOLDS = {
    "lcs_score":              ("gt", 0.35),
    "tfidf_cosine_dist":      ("lt", 0.25),
    "histogram_cosine_dist":  ("lt", 0.10),
    "jaccard_n5":             ("gt", 0.15),
    "bleu_n5":                ("gt", 0.20),
}

SCORE_FIELDS = ["lcs_score", "tfidf_cosine_dist", "histogram_cosine_dist"]
NGRAM_FIELDS = ["jaccard_n3", "bleu_n3", "jaccard_n5", "bleu_n5",
                "jaccard_n8", "bleu_n8"]
FLAG_FIELDS  = ["flagged_ngram", "flagged_lcs", "flagged_ensemble"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PairTypeBucket:
    label: str
    pairs: list = field(default_factory=list)
    _stats: Optional[dict] = field(default=None, repr=False)

    def add(self, pair: dict):
        self.pairs.append(pair)
        self._stats = None

    def stats(self) -> dict:
        if self._stats is not None:
            return self._stats

        n = len(self.pairs)
        if n == 0:
            self._stats = {"n": 0}
            return self._stats

        s: dict = {"n": n}

        # Flag rates
        for flag in FLAG_FIELDS:
            vals = [bool(p.get(flag, False)) for p in self.pairs]
            s[f"{flag}_count"] = sum(vals)
            s[f"{flag}_rate"]  = sum(vals) / n

        any_flag = [any(bool(p.get(f, False)) for f in FLAG_FIELDS)
                    for p in self.pairs]
        s["any_flag_count"] = sum(any_flag)
        s["any_flag_rate"]  = sum(any_flag) / n

        # Continuous score distributions
        for fn in SCORE_FIELDS:
            vals = [p[fn] for p in self.pairs if fn in p]
            if vals:
                arr = np.array(vals)
                s[f"{fn}_mean"]   = float(arr.mean())
                s[f"{fn}_std"]    = float(arr.std())
                s[f"{fn}_median"] = float(np.median(arr))
                s[f"{fn}_p05"]    = float(np.percentile(arr, 5))
                s[f"{fn}_p95"]    = float(np.percentile(arr, 95))

        # N-gram scores (may be absent in fast report)
        for fn in NGRAM_FIELDS:
            vals = [p["ngram_scores"][fn]
                    for p in self.pairs
                    if "ngram_scores" in p and fn in p["ngram_scores"]]
            if vals:
                arr = np.array(vals)
                s[f"{fn}_mean"]   = float(arr.mean())
                s[f"{fn}_std"]    = float(arr.std())
                s[f"{fn}_median"] = float(np.median(arr))

        self._stats = s
        return s

    def score_array(self, field_name: str) -> list:
        if field_name in SCORE_FIELDS:
            return [p[field_name] for p in self.pairs if field_name in p]
        return [p["ngram_scores"][field_name]
                for p in self.pairs
                if "ngram_scores" in p and field_name in p["ngram_scores"]]


# ---------------------------------------------------------------------------
# Loading and aggregation
# ---------------------------------------------------------------------------

def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def pair_type_label(group_a: str, group_b: str) -> str:
    key = (group_a.lower(), group_b.lower())
    return _PAIR_TYPE_MAP.get(key, f"{group_a} x {group_b}")


def aggregate(reports: list) -> dict:
    buckets: dict[str, PairTypeBucket] = {}
    for report in reports:
        for pair in report.get("pair_results", []):
            label = pair_type_label(pair["group_a"], pair["group_b"])
            if label not in buckets:
                buckets[label] = PairTypeBucket(label=label)
            buckets[label].add(pair)
    return buckets


def group_stats_combined(reports: list) -> dict:
    combined: dict = defaultdict(list)
    for report in reports:
        for group, stats in report.get("group_stats", {}).items():
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    combined[f"{group}.{k}"].append(v)
    return {k: float(np.mean(v)) for k, v in combined.items()}


def calibration_status(reports: list) -> str:
    statuses = []
    for r in reports:
        ct = r.get("calibrated_thresholds")
        if ct is None:
            statuses.append("manual (not calibrated)")
        else:
            statuses.append(ct.get("calibration_source", "unknown"))
    return ", ".join(sorted(set(statuses)))


def extract_run_breakdown(bucket: PairTypeBucket) -> dict:
    """
    For LLM x LLM pairs, group by run tag (R0, R1, ...) extracted from
    notagen filenames: notagen_R<N>_C<M>_...
    Only counts same-run pairs (both pieces share the same run tag).
    """
    run_pairs: dict = defaultdict(list)
    for pair in bucket.pairs:
        def _run(name):
            m = re.search(r"_R(\d+)_", name)
            return f"R{m.group(1)}" if m else None
        ra = _run(pair.get("piece_a", ""))
        rb = _run(pair.get("piece_b", ""))
        if ra and rb and ra == rb:
            run_pairs[ra].append(pair)
    if not run_pairs:
        return {}
    result = {}
    for run, pairs in sorted(run_pairs.items()):
        n = len(pairs)
        any_flag = sum(any(bool(p.get(f, False)) for f in FLAG_FIELDS)
                       for p in pairs)
        lcs_vals = [p["lcs_score"] for p in pairs if "lcs_score" in p]
        result[run] = {
            "n":             n,
            "any_flag_rate": any_flag / n if n else 0,
            "lcs_mean":      float(np.mean(lcs_vals)) if lcs_vals else 0.0,
        }
    return result


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

W = 76

def print_summary(buckets: dict, group_stats: dict,
                  cal_status: str, n_reports: int):
    print("=" * W)
    print("  BIAS DETECTION -- AGGREGATED REPORT")
    print(f"  Reports merged : {n_reports}")
    print(f"  Thresholds     : {cal_status}")
    if "manual" in cal_status:
        print("  WARNING: Thresholds NOT calibrated to your null distribution.")
        print("  Re-run with ThresholdCalibrator() for publication-quality flags.")
    print("=" * W)

    # Group-level stats
    groups_seen = sorted(set(k.split(".")[0] for k in group_stats))
    if groups_seen:
        print("\n  -- Group statistics --")
        for group in groups_seen:
            print(f"\n  [{group.upper()}]")
            for k, v in group_stats.items():
                if k.startswith(group + ".") and k != f"{group}.group":
                    print(f"    {k.split('.', 1)[1]:<38} {v:.4f}")

    # Flag rate table
    print(f"\n  -- Flag rates --\n")
    print(f"  {'Pair type':<22} {'N':>6}  {'Any flag':>10}  "
          f"{'N-gram':>8}  {'LCS':>8}  {'Ensemble':>10}")
    print("  " + "-" * (W - 2))

    ordered = [l for l in PAIR_TYPE_ORDER if l in buckets] + \
              [l for l in buckets if l not in PAIR_TYPE_ORDER]
    for label in ordered:
        s = buckets[label].stats()
        if s["n"] == 0:
            continue
        print(f"  {label:<22} {s['n']:>6}  "
              f"{s['any_flag_rate']:>9.1%}  "
              f"{s['flagged_ngram_rate']:>7.1%}  "
              f"{s['flagged_lcs_rate']:>7.1%}  "
              f"{s['flagged_ensemble_rate']:>9.1%}")

    # Score distributions
    print(f"\n  -- Score distributions (mean +/- std) --\n")
    print(f"  {'Pair type':<22}  {'LCS':>16}  {'TF-IDF dist':>16}  {'Hist dist':>16}")
    print("  " + "-" * (W - 2))

    def fmt(s, prefix):
        m  = s.get(f"{prefix}_mean",  float("nan"))
        sd = s.get(f"{prefix}_std",   float("nan"))
        return f"{m:.3f} +/- {sd:.3f}"

    for label in ordered:
        s = buckets[label].stats()
        if s["n"] == 0:
            continue
        print(f"  {label:<22}  {fmt(s, 'lcs_score'):>18}  "
              f"{fmt(s, 'tfidf_cosine_dist'):>18}  "
              f"{fmt(s, 'histogram_cosine_dist'):>18}")

    # Key comparison
    print(f"\n  -- Key comparison: self-copy rates --")
    for label in ["LLM x LLM", "Human x Human"]:
        if label in buckets:
            s = buckets[label].stats()
            print(f"    {label:<22}  LCS mean={s.get('lcs_score_mean', float('nan')):.3f}  "
                  f"p95={s.get('lcs_score_p95', float('nan')):.3f}  "
                  f"LCS-flag={s.get('flagged_lcs_rate', float('nan')):.1%}")

    # Per-run breakdown
    if "LLM x LLM" in buckets:
        runs = extract_run_breakdown(buckets["LLM x LLM"])
        if runs:
            print(f"\n  -- LLM x LLM per-run breakdown --")
            print(f"    {'Run':<8}  {'Pairs':>6}  {'Any-flag':>10}  {'LCS mean':>10}")
            for run, rd in runs.items():
                print(f"    {run:<8}  {rd['n']:>6}  "
                      f"{rd['any_flag_rate']:>9.1%}  "
                      f"{rd['lcs_mean']:>10.3f}")

    print("=" * W)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _ordered_active(buckets: dict) -> list:
    ordered = [l for l in PAIR_TYPE_ORDER if l in buckets] + \
              [l for l in buckets if l not in PAIR_TYPE_ORDER]
    return [(l, buckets[l]) for l in ordered if buckets[l].stats()["n"] > 0]


def _save(fig, out_dir: str, fname: str):
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_bias_incidence_overview(buckets: dict, out_dir: str):
    active = _ordered_active(buckets)
    if not active:
        return
    labels = [l for l, _ in active]
    rates  = [b.stats()["any_flag_rate"]  for _, b in active]
    counts = [b.stats()["any_flag_count"] for _, b in active]
    totals = [b.stats()["n"]              for _, b in active]
    colors = [COLORS.get(l, "#888") for l in labels]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.75 + 1.2)))
    y = np.arange(len(labels))
    ax.barh(y, rates, color=colors, alpha=0.82, height=0.55, edgecolor="white")
    for i, (rate, cnt, tot) in enumerate(zip(rates, counts, totals)):
        ax.text(rate + 0.008, i, f"{rate:.1%}  ({cnt}/{tot})",
                va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Any-flag incidence rate")
    ax.set_title("Bias Incidence Overview (Any Flag Triggered)", fontsize=12)
    ax.set_xlim(0, min(1.0, max(rates or [0]) * 1.45 + 0.06))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save(fig, out_dir, "bias_incidence_overview.png")


def plot_flag_rates(buckets: dict, out_dir: str):
    active = _ordered_active(buckets)
    if not active:
        return
    labels      = [l for l, _ in active]
    flag_keys   = ["flagged_ngram", "flagged_lcs", "flagged_ensemble", "any_flag"]
    flag_names  = ["N-gram", "LCS", "Ensemble (AND)", "Any flag"]
    flag_colors = ["#6baed6", "#fd8d3c", "#74c476", "#e74c3c"]

    x = np.arange(len(labels))
    w = 0.18
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.5), 5))
    for i, (key, name, color) in enumerate(zip(flag_keys, flag_names, flag_colors)):
        rates  = [buckets[l].stats().get(f"{key}_rate", 0) for l in labels]
        offset = (i - 1.5) * w
        ax.bar(x + offset, rates, w, label=name, color=color,
               alpha=0.82, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Flag rate")
    ax.set_title("Bias Flag Rates by Pair Type", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, out_dir, "flag_rates.png")


def plot_lcs_comparison(buckets: dict, out_dir: str):
    """
    Overlaid LCS density histograms for LLM x LLM, Human x Human, LLM x Human.
    The core 'do LLMs self-copy more than humans?' visualization.
    """
    targets = ["LLM x LLM", "Human x Human", "LLM x Human"]
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for label in targets:
        if label not in buckets:
            continue
        vals = buckets[label].score_array("lcs_score")
        if not vals:
            continue
        ax.hist(vals, bins=50, alpha=0.52, color=COLORS[label],
                label=f"{label}  (n={len(vals)})", density=True, edgecolor="none")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    _, thresh = THRESHOLDS["lcs_score"]
    ax.axvline(thresh, color="#c0392b", linestyle="--", linewidth=1.5,
               label=f"Default flag threshold ({thresh})")
    ax.set_xlabel("LCS Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("LCS Score Distribution: LLM vs Human (self-copy comparison)",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, out_dir, "lcs_llm_vs_human.png")


def plot_score_distributions(buckets: dict, out_dir: str):
    configs = [
        ("lcs_score",             "LCS Score",             (0.0, 1.0)),
        ("tfidf_cosine_dist",     "TF-IDF Cosine Distance",(0.0, 1.0)),
        ("histogram_cosine_dist", "Histogram Cosine Dist", (0.0, 0.6)),
    ]
    active = _ordered_active(buckets)
    if not active:
        return
    for field_name, title, ylim in configs:
        data, colors, tick_labels = [], [], []
        for label, bucket in active:
            vals = bucket.score_array(field_name)
            if not vals:
                continue
            data.append(vals)
            colors.append(COLORS.get(label, "#888"))
            tick_labels.append(label)
        if not data:
            continue
        fig, ax = plt.subplots(figsize=(max(7, len(data) * 1.2), 5))
        bp = ax.boxplot(data, patch_artist=True, notch=False, showfliers=True,
                        flierprops=dict(marker=".", alpha=0.35, markersize=4))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
        if field_name in THRESHOLDS:
            direction, val = THRESHOLDS[field_name]
            ax.axhline(val, color="#c0392b", linestyle="--", linewidth=1.2,
                       label=f"Default threshold ({val})")
            ax.legend(fontsize=9)
        ax.set_xticks(range(1, len(tick_labels) + 1))
        ax.set_xticklabels(tick_labels, rotation=18, ha="right")
        ax.set_ylabel(title)
        ax.set_title(f"{title} by Pair Type", fontsize=12)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        _save(fig, out_dir, f"dist_{field_name}.png")


def plot_score_heatmap(buckets: dict, out_dir: str):
    score_cols = ["lcs_score", "tfidf_cosine_dist", "histogram_cosine_dist",
                  "jaccard_n5", "bleu_n5"]
    col_labels = ["LCS", "TF-IDF dist", "Hist dist", "Jaccard n5", "BLEU n5"]
    active = _ordered_active(buckets)
    if not active:
        return
    row_labels = [l for l, _ in active]
    matrix     = np.full((len(active), len(score_cols)), np.nan)
    for i, (label, bucket) in enumerate(active):
        s = bucket.stats()
        for j, col in enumerate(score_cols):
            if f"{col}_mean" in s:
                matrix[i, j] = s[f"{col}_mean"]

    fig, ax = plt.subplots(figsize=(10, max(3, len(row_labels) * 0.9 + 1.8)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.5)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color="black",
                        fontweight="bold" if val > 0.3 else "normal")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Mean score")
    ax.set_title("Mean Score Heatmap by Pair Type", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, "score_heatmap.png")


def plot_style_collapse_comparison(group_stats_raw: dict, out_dir: str):
    group_colors = {
        "llm_output":     "#e74c3c",
        "train_corpus":   "#3498db",
        "duped":          "#f39c12",
        "human_baseline": "#2ecc71",
    }
    groups = sorted(set(k.split(".")[0] for k in group_stats_raw))
    tfidf_vals, hist_vals, labels, colors = [], [], [], []
    for g in groups:
        tv = group_stats_raw.get(f"{g}.style_collapse_tfidf")
        hv = group_stats_raw.get(f"{g}.style_collapse_histogram")
        if tv is None:
            continue
        labels.append(g)
        tfidf_vals.append(tv)
        hist_vals.append(hv or 0)
        colors.append(group_colors.get(g, "#888"))
    if not labels:
        return
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    b1 = ax.bar(x - w/2, tfidf_vals, w, label="TF-IDF",    color=colors, alpha=0.8)
    b2 = ax.bar(x + w/2, hist_vals,  w, label="Histogram", color=colors,
                alpha=0.45, edgecolor="white", linewidth=0.8)
    ax.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Mean intra-group cosine distance (lower = collapsed)")
    ax.set_title("Style Collapse Score by Group", fontsize=12)
    ax.set_ylim(0, max(tfidf_vals + hist_vals) * 1.3 if tfidf_vals else 1)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, out_dir, "style_collapse_comparison.png")


def save_all_figures(buckets: dict, group_stats: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nGenerating figures -> {out_dir}/")
    plot_bias_incidence_overview(buckets, out_dir)
    plot_flag_rates(buckets, out_dir)
    plot_lcs_comparison(buckets, out_dir)
    plot_score_distributions(buckets, out_dir)
    plot_score_heatmap(buckets, out_dir)
    plot_style_collapse_comparison(group_stats, out_dir)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_json(buckets: dict, group_stats: dict,
                reports: list, out_path: str):
    def _s(v):
        if isinstance(v, (np.float32, np.float64)):
            return float(v)
        if isinstance(v, (np.int32, np.int64)):
            return int(v)
        return v

    summary = {
        "n_reports_merged":     len(reports),
        "timestamps":           [r.get("timestamp") for r in reports],
        "calibration_status":   calibration_status(reports),
        "group_stats_averaged": {k: _s(v) for k, v in group_stats.items()},
        "pair_type_summary": {
            label: {k: _s(v) for k, v in bucket.stats().items()}
            for label, bucket in buckets.items()
        },
    }
    if "LLM x LLM" in buckets:
        runs = extract_run_breakdown(buckets["LLM x LLM"])
        if runs:
            summary["llm_per_run_breakdown"] = runs

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAggregated JSON saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", nargs="+",
                        help="Path(s) to report JSON file(s)")
    parser.add_argument("--out", default="results/figures_agg",
                        help="Output directory for figures")
    parser.add_argument("--save-json", default=None, metavar="PATH",
                        help="Save aggregated JSON summary here")
    parser.add_argument("--no-figures", action="store_true",
                        help="Print summary only, skip figures")
    args = parser.parse_args()

    reports = []
    for path in args.reports:
        print(f"Loading: {path}")
        reports.append(load_report(path))

    buckets     = aggregate(reports)
    group_stats = group_stats_combined(reports)
    cal_status  = calibration_status(reports)

    print_summary(buckets, group_stats, cal_status, len(reports))

    if not args.no_figures:
        save_all_figures(buckets, group_stats, args.out)

    if args.save_json:
        export_json(buckets, group_stats, reports, args.save_json)


if __name__ == "__main__":
    main()
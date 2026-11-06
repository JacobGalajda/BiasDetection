"""
transposed_matches.py
---------------------
Identifies pairs flagged exclusively by interval-based LCS but not by
absolute-pitch LCS -- candidate transposed memorisation events.

For each such pair, produces:
  - A summary table (CSV + console)
  - A sliding-window LCS profile plot (Appendix C figures)
  - A JSON record with all metadata for supplemental material

Usage:
    python scripts/transposed_matches.py \
        results/report_full_v2.json \
        results/report_full_v3_intervals.json \
        --out results/transposed/ \
        --top 20

The first argument is the absolute-pitch report, the second is the
interval report. Both must cover the same corpus (same filenames).
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pairs_by_key(report_path: str) -> dict:
    """Load pairs keyed by (piece_a, piece_b) -- canonical sorted order."""
    print(f"Loading {report_path} ...")
    with open(report_path) as f:
        report = json.load(f)

    pairs = {}
    for p in report.get("pair_results", []):
        key = tuple(sorted([p["piece_a"], p["piece_b"]]))
        pairs[key] = p
    print(f"  Loaded {len(pairs):,} pairs")
    return pairs, report.get("calibrated_thresholds", {})


def pair_label(ga: str, gb: str) -> str:
    pairs_map = {
        frozenset(["llm_output", "llm_output"]):        "LLM x LLM",
        frozenset(["llm_output", "duped"]):             "LLM x Duped",
        frozenset(["llm_output", "train_corpus"]):      "LLM x Train",
        frozenset(["llm_output", "human_baseline"]):    "LLM x Human",
        frozenset(["human_baseline", "human_baseline"]):"Human x Human",
        frozenset(["duped", "train_corpus"]):           "Duped x Train",
        frozenset(["duped", "duped"]):                  "Duped x Duped",
    }
    return pairs_map.get(frozenset([ga, gb]), f"{ga} x {gb}")


def run_id(filename: str) -> str:
    m = re.search(r"_R(\d+)_", filename)
    return f"R{m.group(1)}" if m else "unknown"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def find_transposed_matches(abs_pairs: dict, int_pairs: dict,
                             abs_thr: float, int_thr: float,
                             focus_groups: list = None) -> list:
    """
    Returns pairs where:
      - interval LCS >= int_thr  (flagged by interval)
      - abs-pitch LCS < abs_thr  (NOT flagged by abs-pitch)

    focus_groups: if set, only return pairs where at least one piece
                  is in one of these groups. E.g. ["llm_output", "duped"]
    """
    results = []
    common_keys = set(abs_pairs) & set(int_pairs)
    print(f"Common pairs: {len(common_keys):,}")

    for key in common_keys:
        ap = abs_pairs[key]
        ip = int_pairs[key]

        abs_lcs = ap["lcs_score"]
        int_lcs = ip["lcs_score"]

        # Core filter: interval flags, absolute-pitch does not
        if int_lcs < int_thr:
            continue
        if abs_lcs >= abs_thr:
            continue

        ga, gb = ap["group_a"], ap["group_b"]
        label  = pair_label(ga, gb)

        if focus_groups:
            if ga not in focus_groups and gb not in focus_groups:
                continue

        results.append({
            "piece_a":           ap["piece_a"],
            "piece_b":           ap["piece_b"],
            "group_a":           ga,
            "group_b":           gb,
            "pair_type":         label,
            "run_a":             run_id(ap["piece_a"]),
            "run_b":             run_id(ap["piece_b"]),
            "abs_lcs":           round(abs_lcs, 5),
            "int_lcs":           round(int_lcs, 5),
            "lift":              round((int_lcs - abs_lcs) / max(abs_lcs, 1e-9), 3),
            "tfidf_cosine_dist": round(ap.get("tfidf_cosine_dist", 0), 5),
            "hist_cosine_dist":  round(ap.get("histogram_cosine_dist", 0), 5),
            "flagged_abs":       False,
            "flagged_int":       True,
        })

    results.sort(key=lambda x: -x["int_lcs"])
    return results


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(matches: list, top: int = 20):
    print(f"\n{'='*78}")
    print(f"  TRANSPOSED MATCHES -- interval-flagged, abs-pitch-negative")
    print(f"  Total: {len(matches)}  |  Showing top {min(top, len(matches))}")
    print(f"{'='*78}")

    by_type = defaultdict(list)
    for m in matches:
        by_type[m["pair_type"]].append(m)

    print(f"\n  By pair type:")
    for pt, ms in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"    {pt:<30} {len(ms):>4} pairs")

    print(f"\n  {'#':<4} {'Pair type':<20} {'Runs':<12} {'Abs LCS':>8} {'Int LCS':>8} {'Lift':>7}  Piece A")
    print(f"  {'-'*90}")
    for i, m in enumerate(matches[:top], 1):
        runs = f"{m['run_a']}/{m['run_b']}" if m['run_a'] != m['run_b'] else m['run_a']
        name_a = m["piece_a"][:35] + "..." if len(m["piece_a"]) > 36 else m["piece_a"]
        print(f"  {i:<4} {m['pair_type']:<20} {runs:<12} "
              f"{m['abs_lcs']:>8.5f} {m['int_lcs']:>8.5f} "
              f"{m['lift']:>6.1f}x  {name_a}")


# ---------------------------------------------------------------------------
# Profile plot
# ---------------------------------------------------------------------------

def sliding_window_lcs_profile(seq_a: list, seq_b: list,
                                window_size: int = None) -> list:
    """
    Slide seq_a (shorter) across seq_b, compute LCS ratio in each window.
    Returns list of (position, score).
    """
    if len(seq_a) > len(seq_b):
        seq_a, seq_b = seq_b, seq_a

    w = window_size or len(seq_a)
    if w > len(seq_b):
        w = len(seq_b)

    results = []
    for start in range(0, len(seq_b) - w + 1, max(1, w // 20)):
        window = seq_b[start:start + w]
        lcs_len = _lcs_length(seq_a[:w], window)
        results.append((start, lcs_len / w))
    return results


def _lcs_length(a: list, b: list) -> int:
    """Standard DP LCS length."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(m):
        curr = [0] * (n + 1)
        for j in range(n):
            if a[i] == b[j]:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(prev[j + 1], curr[j])
        prev = curr
    return prev[n]


def plot_profile(profile: list, match: dict,
                 abs_thr: float, int_thr: float,
                 out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [skip plot -- matplotlib not available]")
        return

    positions = [p[0] for p in profile]
    scores    = [p[1] for p in profile]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [3, 1]})

    # Left: sliding window profile
    ax = axes[0]
    ax.plot(positions, scores, linewidth=1.2, color="#3498db", label="Interval LCS sim")
    ax.fill_between(positions, scores, alpha=0.12, color="#3498db")
    ax.axhline(int_thr,  linestyle="--", color="#9b59b6", linewidth=1,
               label=f"Interval threshold ({int_thr:.3f})")
    ax.axhline(abs_thr,  linestyle=":",  color="#e74c3c",  linewidth=1,
               label=f"Abs-pitch threshold ({abs_thr:.3f})")

    if scores:
        peak_pos = positions[np.argmax(scores)]
        peak_val = max(scores)
        ax.annotate(f"peak={peak_val:.3f}",
                    xy=(peak_pos, peak_val),
                    xytext=(peak_pos, min(peak_val + 0.08, 0.98)),
                    fontsize=8, color="#9b59b6",
                    arrowprops=dict(arrowstyle="->", color="#9b59b6"))

    name_a = match["piece_a"][:40]
    name_b = match["piece_b"][:40]
    ax.set_title(f"Sliding-window interval LCS\n{name_a}  <->  {name_b}",
                 fontsize=9)
    ax.set_xlabel("Reference token position (window start)", fontsize=9)
    ax.set_ylabel("Interval LCS similarity", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    # Right: summary bar
    ax2 = axes[1]
    metrics = ["Abs-pitch\nLCS", "Interval\nLCS", "TF-IDF\ndist", "Hist\ndist"]
    values  = [match["abs_lcs"], match["int_lcs"],
               match["tfidf_cosine_dist"], match["hist_cosine_dist"]]
    colors  = ["#e74c3c", "#9b59b6", "#3498db", "#2ecc71"]
    bars = ax2.bar(metrics, values, color=colors, alpha=0.8,
                   edgecolor="white", linewidth=0.6)
    ax2.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
    ax2.set_title("Metric values", fontsize=9)
    ax2.set_ylim(0, max(values) * 1.4 if values else 1.0)
    ax2.axhline(int_thr, linestyle="--", color="#9b59b6",
                linewidth=0.8, alpha=0.6)
    ax2.grid(axis="y", alpha=0.2)

    fig.suptitle(f"Transposed match candidate  [{match['pair_type']}]  "
                 f"int_lcs={match['int_lcs']:.4f}  abs_lcs={match['abs_lcs']:.4f}",
                 fontsize=10, y=1.01)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Token loader
# ---------------------------------------------------------------------------

def load_tokens_from_report(report_path: str) -> dict:
    """
    We can't re-tokenise without the original files, but we can extract
    token sequences from the report if they were stored -- they usually aren't.
    Returns empty dict; caller falls back to profile-less output.
    """
    return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("abs_report",
                        help="Absolute-pitch report JSON (report_full_v2.json)")
    parser.add_argument("int_report",
                        help="Interval report JSON (report_full_v3_intervals.json)")
    parser.add_argument("--out",    default="results/transposed",
                        help="Output directory (default: results/transposed)")
    parser.add_argument("--top",    type=int, default=20,
                        help="Number of top matches to show/save (default: 20)")
    parser.add_argument("--abs-thr", type=float, default=0.387,
                        help="Abs-pitch LCS flag threshold (default: 0.387)")
    parser.add_argument("--int-thr", type=float, default=0.387,
                        help="Interval LCS flag threshold (default: 0.387)")
    parser.add_argument("--focus",  nargs="+",
                        default=["llm_output", "duped", "train_corpus"],
                        help="Limit to pairs involving these groups")
    parser.add_argument("--all-groups", action="store_true",
                        help="Include all group combinations (ignores --focus)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    abs_pairs, abs_thr_obj = load_pairs_by_key(args.abs_report)
    int_pairs, int_thr_obj = load_pairs_by_key(args.int_report)

    # Use calibrated thresholds from report if not overridden
    abs_thr = abs_thr_obj.get("lcs_strict", args.abs_thr)
    int_thr = int_thr_obj.get("lcs_strict", args.int_thr)
    print(f"\nUsing thresholds:  abs={abs_thr:.4f}  interval={int_thr:.4f}")

    focus = None if args.all_groups else args.focus
    matches = find_transposed_matches(abs_pairs, int_pairs,
                                       abs_thr, int_thr, focus)

    print_summary(matches, args.top)

    if not matches:
        print("\n  No transposed matches found at these thresholds.")
        print("  Try --abs-thr 0.20 --int-thr 0.20 for a relaxed search,")
        print("  or --all-groups to include all pair types.")
        return

    # Save CSV
    csv_path = out_dir / "transposed_matches.csv"
    fieldnames = ["rank", "pair_type", "run_a", "run_b",
                  "abs_lcs", "int_lcs", "lift",
                  "tfidf_cosine_dist", "hist_cosine_dist",
                  "piece_a", "piece_b", "group_a", "group_b"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for i, m in enumerate(matches[:args.top], 1):
            w.writerow({"rank": i, **m})
    print(f"\n  CSV saved: {csv_path}")

    # Save JSON (full supplemental record)
    json_path = out_dir / "transposed_matches.json"
    with open(json_path, "w") as f:
        json.dump({
            "abs_report":       args.abs_report,
            "int_report":       args.int_report,
            "abs_threshold":    abs_thr,
            "int_threshold":    int_thr,
            "n_total_matches":  len(matches),
            "focus_groups":     focus,
            "matches":          matches[:args.top],
        }, f, indent=2)
    print(f"  JSON saved: {json_path}")

    # Relaxed search for supplemental stats (lower threshold)
    print("\n-- Relaxed search (int_lcs >= 0.10, abs_lcs < 0.10) --")
    relaxed = find_transposed_matches(abs_pairs, int_pairs,
                                       0.10, 0.10, focus)
    print(f"  {len(relaxed)} pairs have interval LCS >=0.10 but abs-pitch LCS <0.10")
    if relaxed:
        by_type = defaultdict(int)
        for m in relaxed:
            by_type[m["pair_type"]] += 1
        for pt, n in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"    {pt:<30} {n:>4}")

        relaxed_path = out_dir / "transposed_relaxed.csv"
        with open(relaxed_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for i, m in enumerate(relaxed, 1):
                w.writerow({"rank": i, **m})
        print(f"  Relaxed CSV: {relaxed_path}")

    # Profile plots -- only possible if we can retokenise
    # Point user to the right command
    print(f"""
-- Appendix C profile plots --
To generate sliding-window LCS profiles for these pairs, run the
tokeniser on the specific files and call plot_lcs_profile() from
visualise.py. The match filenames are in:

    {json_path}

Quick command to regenerate profiles for the top match:
    python - << 'EOF'
import json
from pathlib import Path
from music_bias_analyzer.tokenizer import file_to_tokens
from music_bias_analyzer.visualise import plot_lcs_profile

with open("{json_path}") as f:
    data = json.load(f)

match = data["matches"][0]
# Load tokens from your data directories
toks_a = file_to_tokens(Path("data/llm_output") / match["piece_a"],
                         use_intervals=True)
toks_b = file_to_tokens(Path("data/human_baseline_sq") / match["piece_b"],
                         use_intervals=True)
# Sliding window: slide shorter over longer
if len(toks_a) > len(toks_b):
    toks_a, toks_b = toks_b, toks_a
w = len(toks_a)
profile = []
from scripts.transposed_matches import _lcs_length
for start in range(0, len(toks_b) - w + 1, max(1, w // 20)):
    window = toks_b[start:start + w]
    score  = _lcs_length(toks_a, window) / w
    profile.append((start, score))
fig = plot_lcs_profile(profile,
                        query_name=match["piece_a"],
                        reference_name=match["piece_b"],
                        flag_threshold={int_thr:.3f})
fig.savefig("results/transposed/appendix_c_top1.png", dpi=150)
EOF
""")


if __name__ == "__main__":
    main()

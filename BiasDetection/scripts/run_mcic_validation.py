"""
run_mcic_validation.py
-----------------------
Validate the bias-detection protocol against the MCIC dataset
(Park et al., ISMIR 2024 -- 116 adjudicated copyright infringement cases).

Runs best-window LCS (absolute-pitch and interval) and Jaccard n=5
on every usable MCIC pair (Infringed=32, Denied=66; Settled=18 excluded).
Reports Precision, Recall, F1, and Accuracy for three flagging strategies:
  1. Absolute-pitch LCS only
  2. Interval LCS only
  3. Either metric (union) -- recommended

Usage
-----
    python scripts/run_mcic_validation.py \
        --labels  data/mcic/labels.csv \
        --out     results/validation/mcic_validation.json \
        [--lcs_threshold  0.387] \
        [--jac_threshold  0.022] \
        [--workers 4]

Thresholds default to the Pareto-calibrated values from the NotaGen audit.
Pass --lcs_threshold / --jac_threshold to sweep alternatives.

Output
------
    results/validation/mcic_validation.json   -- full results
    results/validation/mcic_validation.csv    -- per-pair scores
    Printed summary table to stdout
"""

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Attempt to import pipeline tokenizer and LCS
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from music_bias_analyzer.tokenizer import file_to_tokens
    from music_bias_analyzer.lcs import lcs_density_score, lcs_similarity
    from music_bias_analyzer.ngram import jaccard_overlap
except ImportError as e:
    print(f"[error] Could not import music_bias_analyzer: {e}")
    print("        Run from repo root or ensure music_bias_analyzer is on PYTHONPATH.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Interval tokenization helper
# ---------------------------------------------------------------------------

def to_interval_tokens(abs_tokens: list[str]) -> list[str]:
    """
    Convert absolute-pitch token list to interval tokens.
    Each token expected as "PITCH_DURATION", e.g. "G4_quarter".
    Interval token = signed semitone difference + "_" + duration.
    Rests (pitch -1 or token starting with "R") produce interval token "0_<dur>".
    """
    NOTE_TO_MIDI = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
        "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
        "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
    }

    def token_to_midi(tok: str) -> int | None:
        """Return MIDI pitch from token string, or None for rests/unknowns."""
        if not tok or tok.startswith("R"):
            return None
        parts = tok.split("_", 1)
        pitch_str = parts[0]
        # Try to extract note name and octave, e.g. "G4", "C#5"
        import re
        m = re.match(r'^([A-Ga-g][#b]?)(-?\d+)$', pitch_str)
        if not m:
            return None
        note, octave = m.group(1), int(m.group(2))
        semitone = NOTE_TO_MIDI.get(note.capitalize())
        if semitone is None:
            return None
        return semitone + (octave + 1) * 12

    interval_tokens = []
    prev_midi = None
    for tok in abs_tokens:
        parts = tok.split("_", 1)
        dur = parts[1] if len(parts) > 1 else "unknown"
        midi = token_to_midi(tok)
        if midi is None or prev_midi is None:
            interval_tokens.append(f"0_{dur}")
        else:
            interval_tokens.append(f"{midi - prev_midi}_{dur}")
        if midi is not None:
            prev_midi = midi
    return interval_tokens


# ---------------------------------------------------------------------------
# Single-pair scoring
# ---------------------------------------------------------------------------

def score_pair(args):
    """Worker function -- score one MCIC pair."""
    case_id, file_a, file_b, outcome, lcs_thr, jac_thr = args

    result = {
        "case_id":     case_id,
        "outcome":     outcome,
        "outcome_bin": 1 if outcome == "Infringed" else 0,
        "error":       None,
        # Absolute-pitch
        "lcs_abs":        None,
        "flagged_abs":    None,
        "jaccard_n5":     None,
        "flagged_jac":    None,
        # Interval
        "lcs_int":        None,
        "flagged_int":    None,
        # Union
        "flagged_union":  None,
    }

    try:
        toks_a = file_to_tokens(file_a)
        toks_b = file_to_tokens(file_b)
    except Exception as e:
        result["error"] = f"tokenize: {e}"
        return result

    if not toks_a or not toks_b:
        result["error"] = "empty token sequence"
        return result

    try:
        # Absolute-pitch LCS -- WAR/WAQ density score (recommended)
        lcs_abs_d    = lcs_density_score(toks_a, toks_b)
        lcs_abs      = lcs_abs_d.get("density_score", 0.0)
        result["lcs_abs"]          = round(lcs_abs, 6)
        result["flagged_abs"]      = int(lcs_abs >= lcs_thr)

        # Absolute-pitch LCS -- standard normalized LCS/max(len) (for comparison)
        lcs_abs_norm = lcs_similarity(toks_a, toks_b)
        result["lcs_abs_norm"]     = round(lcs_abs_norm, 6)
        result["flagged_abs_norm"] = int(lcs_abs_norm >= lcs_thr)

        # Jaccard n=5
        jac = jaccard_overlap(toks_a, toks_b, n=5)
        result["jaccard_n5"]  = round(jac, 6)
        result["flagged_jac"] = int(jac >= jac_thr)

        # Interval LCS -- WAR/WAQ density score (recommended)
        int_a     = to_interval_tokens(toks_a)
        int_b     = to_interval_tokens(toks_b)
        lcs_int_d = lcs_density_score(int_a, int_b)
        lcs_int   = lcs_int_d.get("density_score", 0.0)
        result["lcs_int"]          = round(lcs_int, 6)
        result["flagged_int"]      = int(lcs_int >= lcs_thr)

        # Interval LCS -- standard normalized (for comparison)
        lcs_int_norm = lcs_similarity(int_a, int_b)
        result["lcs_int_norm"]     = round(lcs_int_norm, 6)
        result["flagged_int_norm"] = int(lcs_int_norm >= lcs_thr)

        # Union flag (WAR/WAQ variants only -- canonical)
        result["flagged_union"] = int(
            result["flagged_abs"] or result["flagged_int"] or result["flagged_jac"]
        )

    except Exception as e:
        result["error"] = f"scoring: {e}"

    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def prf1(results: list[dict], flag_col: str) -> dict:
    """Compute P, R, F1, Accuracy for binary classification."""
    tp = fp = fn = tn = 0
    for r in results:
        if r["error"]:
            continue
        pred  = r[flag_col]
        truth = r["outcome_bin"]
        if pred == 1 and truth == 1:  tp += 1
        elif pred == 1 and truth == 0: fp += 1
        elif pred == 0 and truth == 1: fn += 1
        else:                          tn += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc  = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "accuracy":  round(acc,  4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels",        default="data/mcic/labels.csv",
                    help="Path to MCIC labels.csv")
    ap.add_argument("--out",           default="results/validation/mcic_validation.json",
                    help="Output JSON path")
    ap.add_argument("--lcs_threshold", type=float, default=0.387,
                    help="LCS flagging threshold (default: Pareto-calibrated 0.387)")
    ap.add_argument("--jac_threshold", type=float, default=0.022,
                    help="Jaccard n=5 threshold (default: Pareto-calibrated 0.022)")
    ap.add_argument("--workers",       type=int, default=max(1, mp.cpu_count() - 1),
                    help="Parallel workers (default: cpu_count - 1)")
    ap.add_argument("--include_settled", action="store_true",
                    help="Include settled cases (outcome_bin=-1) in evaluation")
    args = ap.parse_args()

    labels_path = Path(args.labels)
    out_path    = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- load labels ---
    print(f"\nLoading labels from {labels_path} ...")
    rows = []
    with open(labels_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ob = row.get("outcome_bin", "")
            if ob == "-1" and not args.include_settled:
                continue   # exclude settled
            if ob not in ("0", "1"):
                continue   # skip blank / unknown
            rows.append(row)

    n_inf  = sum(1 for r in rows if r["outcome_bin"] == "1")
    n_den  = sum(1 for r in rows if r["outcome_bin"] == "0")
    print(f"  Usable pairs: {len(rows)} "
          f"(Infringed={n_inf}, Denied={n_den})")
    print(f"  Thresholds: LCS>={args.lcs_threshold}, Jaccard>={args.jac_threshold}")
    print(f"  Workers: {args.workers}")

    # --- build worker args ---
    worker_args = [
        (r["case_id"], r["file_a"], r["file_b"], r["outcome"],
         args.lcs_threshold, args.jac_threshold)
        for r in rows
    ]

    # --- run scoring ---
    print(f"\nScoring {len(worker_args)} pairs ...")
    t0 = time.time()

    if args.workers == 1:
        results = [score_pair(a) for a in worker_args]
    else:
        with mp.Pool(args.workers) as pool:
            results = pool.map(score_pair, worker_args)

    elapsed = time.time() - t0
    errors  = [r for r in results if r["error"]]
    print(f"  Done in {elapsed:.1f}s -- {len(errors)} errors")
    if errors:
        print("  Errors:")
        for r in errors[:5]:
            print(f"    Case {r['case_id']}: {r['error']}")

    # --- compute metrics ---
    usable = [r for r in results if not r["error"]]

    metrics = {
        "absolute_lcs_warwaq":  prf1(usable, "flagged_abs"),
        "absolute_lcs_normalized":    prf1(usable, "flagged_abs_norm"),
        "interval_lcs_warwaq":  prf1(usable, "flagged_int"),
        "interval_lcs_normalized":    prf1(usable, "flagged_int_norm"),
        "jaccard_n5":           prf1(usable, "flagged_jac"),
        "union":                prf1(usable, "flagged_union"),
    }

    # --- print summary table ---
    print(f"\n{'='*62}")
    print(f"MCIC Validation Results  "
          f"(n={len(usable)}, Inf={n_inf}, Den={n_den})")
    print(f"{'='*62}")
    print(f"{'Strategy':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} "
          f"{'Acc':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    print(f"{'-'*62}")
    labels_map = {
        "absolute_lcs_warwaq": "Abs LCS (WAR/WAQ)",
        "absolute_lcs_normalized":   "Abs LCS (normalized)",
        "interval_lcs_warwaq": "Int LCS (WAR/WAQ)",
        "interval_lcs_normalized":   "Int LCS (normalized)",
        "jaccard_n5":          "Jaccard n=5",
        "union":               "Union (any)",
    }
    for key, m in metrics.items():
        print(f"{labels_map[key]:<20} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['accuracy']:>6.3f} "
              f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['tn']:>4}")
    print(f"{'='*62}")

    # --- transposition analysis ---
    abs_only_fn  = [r for r in usable
                    if r["outcome_bin"] == 1
                    and r["flagged_abs"] == 0
                    and r["flagged_int"] == 1]
    int_only_tp  = [r for r in usable
                    if r["outcome_bin"] == 1
                    and r["flagged_abs"] == 0
                    and r["flagged_int"] == 1]

    print(f"\nTransposition analysis:")
    print(f"  Infringed pairs caught by interval but NOT abs-pitch: "
          f"{len(int_only_tp)}")
    if int_only_tp:
        print(f"  Cases: {[r['case_id'] for r in int_only_tp]}")

    # --- save outputs ---
    output = {
        "dataset":    "MCIC (Park et al., ISMIR 2024)",
        "n_pairs":    len(usable),
        "n_infringed": n_inf,
        "n_denied":   n_den,
        "thresholds": {
            "lcs":     args.lcs_threshold,
            "jaccard": args.jac_threshold,
        },
        "metrics":    metrics,
        "per_pair":   results,
        "errors":     len(errors),
        "elapsed_s":  round(elapsed, 2),
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {out_path}")

    # --- per-pair CSV ---
    csv_path = out_path.with_suffix(".csv")
    fieldnames = [
        "case_id", "outcome", "outcome_bin",
        "lcs_abs", "lcs_abs_norm",
        "lcs_int", "lcs_int_norm",
        "jaccard_n5",
        "flagged_abs", "flagged_abs_norm",
        "flagged_int", "flagged_int_norm",
        "flagged_jac", "flagged_union",
        "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Per-pair scores saved to {csv_path}")


if __name__ == "__main__":
    main()

"""
run_smp_validation.py
---------------------
Validate the bias-detection protocol against the SMP dataset
(Go 2025 / Go & Kim ICASSP 2026 -- 70 similar music pairs with timestamps).

Scores each transcribed segment pair using LCS (absolute-pitch and interval)
and Jaccard n=5. Reports segment-level and pair-level Precision/Recall/F1.

Relation labels:
  plag        -> positive (1)
  remake      -> positive (1)
  plag_doubt  -> excluded (-1)

Usage
-----
    python scripts/run_smp_validation.py \
        --labels  data/smp/labels.csv \
        --out     results/validation/smp_validation.json \
        [--lcs_threshold 0.387] \
        [--jac_threshold 0.022]
"""

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from music_bias_analyzer.tokenizer import file_to_tokens
    from music_bias_analyzer.lcs import lcs_density_score
    from music_bias_analyzer.ngram import jaccard_overlap
except ImportError as e:
    print(f"[error] {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Interval tokenisation (same as run_mcic_validation.py)
# ---------------------------------------------------------------------------

def to_interval_tokens(abs_tokens):
    NOTE_TO_MIDI = {
        "C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,
        "E":4,"F":5,"F#":6,"Gb":6,"G":7,"G#":8,
        "Ab":8,"A":9,"A#":10,"Bb":10,"B":11,
    }
    import re as _re

    def tok_to_midi(tok):
        m = _re.match(r'^([A-Ga-g][#b]?)(-?\d+)', tok.split("_")[0])
        if not m:
            return None
        note, oct_ = m.group(1).capitalize(), int(m.group(2))
        s = NOTE_TO_MIDI.get(note)
        return None if s is None else s + (oct_ + 1) * 12

    out, prev = [], None
    for tok in abs_tokens:
        parts = tok.split("_", 1)
        dur   = parts[1] if len(parts) > 1 else "unknown"
        midi  = tok_to_midi(tok)
        if midi is None or prev is None:
            out.append(f"0_{dur}")
        else:
            out.append(f"{midi-prev}_{dur}")
        if midi is not None:
            prev = midi
    return out


# ---------------------------------------------------------------------------
# Score one segment pair
# ---------------------------------------------------------------------------

def score_segment(args):
    seg_id, file_ori, file_comp, outcome_bin, lcs_thr, jac_thr = args

    result = {
        "seg_id":       seg_id,
        "outcome_bin":  outcome_bin,
        "error":        None,
        "lcs_abs":      None, "flagged_abs": None,
        "lcs_int":      None, "flagged_int": None,
        "jaccard_n5":   None, "flagged_jac": None,
        "flagged_union": None,
    }

    if not file_ori or not file_comp:
        result["error"] = "missing file"
        return result

    try:
        toks_a = file_to_tokens(file_ori)
        toks_b = file_to_tokens(file_comp)
    except Exception as e:
        result["error"] = f"tokenize: {e}"
        return result

    if not toks_a or not toks_b:
        result["error"] = "empty tokens"
        return result

    try:
        d_abs = lcs_density_score(toks_a, toks_b)
        lcs_abs = d_abs.get("density_score", 0.0)
        result["lcs_abs"]     = round(lcs_abs, 6)
        result["flagged_abs"] = int(lcs_abs >= lcs_thr)

        jac = jaccard_overlap(toks_a, toks_b, n=5)
        result["jaccard_n5"]  = round(jac, 6)
        result["flagged_jac"] = int(jac >= jac_thr)

        int_a = to_interval_tokens(toks_a)
        int_b = to_interval_tokens(toks_b)
        d_int = lcs_density_score(int_a, int_b)
        lcs_int = d_int.get("density_score", 0.0)
        result["lcs_int"]     = round(lcs_int, 6)
        result["flagged_int"] = int(lcs_int >= lcs_thr)

        result["flagged_union"] = int(
            result["flagged_abs"] or result["flagged_int"] or result["flagged_jac"]
        )
    except Exception as e:
        result["error"] = f"scoring: {e}"

    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def prf1(results, flag_col):
    tp = fp = fn = tn = 0
    for r in results:
        if r["error"] or r[flag_col] is None:
            continue
        pred, truth = r[flag_col], r["outcome_bin"]
        if pred == 1 and truth == 1:   tp += 1
        elif pred == 1 and truth == 0: fp += 1
        elif pred == 0 and truth == 1: fn += 1
        else:                          tn += 1
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn) > 0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    acc  = (tp+tn)/(tp+fp+fn+tn) if (tp+fp+fn+tn) > 0 else 0.0
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,
            "precision":round(prec,4),"recall":round(rec,4),
            "f1":round(f1,4),"accuracy":round(acc,4)}


def pair_level_metrics(seg_results, flag_col):
    """
    Aggregate segment-level scores to pair level.
    A pair is flagged if ANY of its segments is flagged.
    """
    pairs = defaultdict(lambda: {"flagged":0, "outcome_bin": None, "n_segs":0})
    for r in seg_results:
        if r["error"] or r[flag_col] is None:
            continue
        pid = r["seg_id"].rsplit("_seg", 1)[0]
        pairs[pid]["outcome_bin"] = r["outcome_bin"]
        pairs[pid]["n_segs"] += 1
        if r[flag_col] == 1:
            pairs[pid]["flagged"] = 1

    pair_list = [{"outcome_bin": v["outcome_bin"],
                  "error": None,
                  flag_col: v["flagged"]}
                 for v in pairs.values()
                 if v["outcome_bin"] is not None]
    return prf1(pair_list, flag_col)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels",        default="data/smp/labels.csv")
    ap.add_argument("--out",           default="results/validation/smp_validation.json")
    ap.add_argument("--lcs_threshold", type=float, default=0.387)
    ap.add_argument("--jac_threshold", type=float, default=0.022)
    ap.add_argument("--workers",       type=int,
                    default=max(1, mp.cpu_count()-1))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading labels from {args.labels} ...")
    rows = []
    with open(args.labels, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ob = int(row.get("outcome_bin", -1))
            if ob == -1:
                continue  # exclude plag_doubt
            if not row.get("ok", "True") in ("True", "1", "true"):
                continue  # exclude failed transcriptions
            rows.append(row)

    n_pos = sum(1 for r in rows if int(r["outcome_bin"]) == 1)
    n_neg = sum(1 for r in rows if int(r["outcome_bin"]) == 0)
    print(f"  Usable segments: {len(rows)}  (pos={n_pos}, neg={n_neg})")
    print(f"  Thresholds: LCS>={args.lcs_threshold}, Jaccard>={args.jac_threshold}")

    worker_args = [
        (r["seg_id"], r["file_ori"], r["file_comp"],
         int(r["outcome_bin"]), args.lcs_threshold, args.jac_threshold)
        for r in rows
    ]

    print(f"\nScoring {len(worker_args)} segments ...")
    t0 = time.time()
    if args.workers == 1:
        results = [score_segment(a) for a in worker_args]
    else:
        with mp.Pool(args.workers) as pool:
            results = pool.map(score_segment, worker_args)
    elapsed = time.time() - t0

    errors  = [r for r in results if r["error"]]
    usable  = [r for r in results if not r["error"]]
    print(f"  Done in {elapsed:.1f}s -- {len(errors)} errors")

    # Segment-level metrics
    seg_metrics = {
        "absolute_lcs": prf1(usable, "flagged_abs"),
        "interval_lcs": prf1(usable, "flagged_int"),
        "jaccard_n5":   prf1(usable, "flagged_jac"),
        "union":        prf1(usable, "flagged_union"),
    }

    # Pair-level metrics
    pair_metrics = {
        "absolute_lcs": pair_level_metrics(usable, "flagged_abs"),
        "interval_lcs": pair_level_metrics(usable, "flagged_int"),
        "jaccard_n5":   pair_level_metrics(usable, "flagged_jac"),
        "union":        pair_level_metrics(usable, "flagged_union"),
    }

    labels_map = {
        "absolute_lcs": "Abs-pitch LCS",
        "interval_lcs": "Interval LCS",
        "jaccard_n5":   "Jaccard n=5",
        "union":        "Union (any)",
    }

    print(f"\n{'='*65}")
    print(f"SMP Validation -- SEGMENT level (n={len(usable)} segments)")
    print(f"{'='*65}")
    print(f"{'Strategy':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} "
          f"{'Acc':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    print("-"*65)
    for k, m in seg_metrics.items():
        print(f"{labels_map[k]:<20} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['accuracy']:>6.3f} "
              f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['tn']:>4}")

    print(f"\n{'='*65}")
    print(f"SMP Validation -- PAIR level (any-segment union per pair)")
    print(f"{'='*65}")
    print(f"{'Strategy':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} "
          f"{'Acc':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    print("-"*65)
    for k, m in pair_metrics.items():
        print(f"{labels_map[k]:<20} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['accuracy']:>6.3f} "
              f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['tn']:>4}")

    # Transposition analysis
    trans_only = [r for r in usable
                  if r["outcome_bin"] == 1
                  and r["flagged_abs"] == 0
                  and r["flagged_int"] == 1]
    print(f"\nTransposition (interval-only detections): {len(trans_only)}")

    output = {
        "dataset":        "SMP (Go 2025 / ICASSP 2026)",
        "n_segments":     len(usable),
        "thresholds":     {"lcs": args.lcs_threshold, "jaccard": args.jac_threshold},
        "segment_metrics": seg_metrics,
        "pair_metrics":   pair_metrics,
        "transposition_only": len(trans_only),
        "errors":         len(errors),
        "elapsed_s":      round(elapsed, 2),
        "per_segment":    results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    csv_path = out_path.with_suffix(".csv")
    fieldnames = ["seg_id","outcome_bin","lcs_abs","lcs_int","jaccard_n5",
                  "flagged_abs","flagged_int","flagged_jac","flagged_union","error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {out_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
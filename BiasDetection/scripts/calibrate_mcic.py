"""
calibrate_mcic.py
-----------------
Find Pareto-optimal thresholds for each metric on the MCIC dataset,
using the same approach as the NotaGen audit (S.3.4) but with a
MCIC-specific null distribution.

Null distribution:  Denied pairs  (court-confirmed non-infringement)
Candidate positive: Infringed pairs (court-confirmed infringement)

Usage
-----
    python scripts/calibrate_mcic.py \
        --scores  results/validation/mcic_validation.csv \
        --out     results/validation/mcic_calibrated_thresholds.json

Then re-run validation with calibrated thresholds:
    python scripts/run_mcic_validation.py \
        --labels        data/mcic/labels.csv \
        --out           results/validation/mcic_validation_calibrated.json \
        --lcs_threshold <calibrated> \
        --jac_threshold <calibrated>
"""

import argparse
import csv
import json
import numpy as np
from pathlib import Path


def pareto_threshold(pos_scores, neg_scores, metric_name,
                     direction="high", n_steps=200):
    all_scores = np.concatenate([pos_scores, neg_scores])
    lo, hi = float(all_scores.min()), float(all_scores.max())
    thresholds = np.linspace(lo, hi, n_steps)

    results = []
    for thr in thresholds:
        if direction == "high":
            tp = int((pos_scores >= thr).sum())
            fp = int((neg_scores >= thr).sum())
        else:
            tp = int((pos_scores <= thr).sum())
            fp = int((neg_scores <= thr).sum())

        fn = len(pos_scores) - tp
        tn = len(neg_scores) - fp
        tpr  = tp / len(pos_scores) if len(pos_scores) > 0 else 0.0
        fpr  = fp / len(neg_scores) if len(neg_scores) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1   = 2*prec*tpr / (prec+tpr) if (prec+tpr) > 0 else 0.0
        acc  = (tp+tn)/(tp+fp+fn+tn) if (tp+fp+fn+tn) > 0 else 0.0

        results.append({
            "threshold": float(thr),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "tpr": round(tpr, 4), "fpr": round(fpr, 4),
            "precision": round(prec, 4),
            "recall": round(tpr, 4),
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
        })

    # Pareto front: for each TPR level, keep lowest FPR
    pareto = {}
    for r in results:
        tpr = r["tpr"]
        if tpr not in pareto or r["fpr"] < pareto[tpr]["fpr"]:
            pareto[tpr] = r

    best_f1  = max(pareto.values(), key=lambda x: x["f1"])
    best_acc = max(pareto.values(), key=lambda x: x["accuracy"])
    best_tpr = max(pareto.values(), key=lambda x: (x["tpr"], -x["fpr"]))

    print(f"\n  {metric_name} ({direction}):")
    print(f"    Best F1:       thr={best_f1['threshold']:.4f}  "
          f"P={best_f1['precision']:.3f}  R={best_f1['recall']:.3f}  "
          f"F1={best_f1['f1']:.3f}  Acc={best_f1['accuracy']:.3f}  "
          f"TP={best_f1['tp']}  FP={best_f1['fp']}")
    print(f"    Best Accuracy: thr={best_acc['threshold']:.4f}  "
          f"P={best_acc['precision']:.3f}  R={best_acc['recall']:.3f}  "
          f"F1={best_acc['f1']:.3f}  Acc={best_acc['accuracy']:.3f}  "
          f"TP={best_acc['tp']}  FP={best_acc['fp']}")
    print(f"    Best Recall:   thr={best_tpr['threshold']:.4f}  "
          f"P={best_tpr['precision']:.3f}  R={best_tpr['recall']:.3f}  "
          f"F1={best_tpr['f1']:.3f}  Acc={best_tpr['accuracy']:.3f}  "
          f"TP={best_tpr['tp']}  FP={best_tpr['fp']}")

    return {
        "metric": metric_name,
        "direction": direction,
        "n_pos": int(len(pos_scores)),
        "n_neg": int(len(neg_scores)),
        "best_f1":  best_f1,
        "best_acc": best_acc,
        "best_tpr": best_tpr,
        "pareto_front": sorted(pareto.values(), key=lambda x: x["tpr"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="results/validation/mcic_validation.csv")
    ap.add_argument("--out",    default="results/validation/mcic_calibrated_thresholds.json")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading scores from {args.scores} ...")
    pos = {"lcs_abs": [], "lcs_int": [], "jaccard_n5": []}
    neg = {"lcs_abs": [], "lcs_int": [], "jaccard_n5": []}

    with open(args.scores, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("error"):
                continue
            ob = int(row["outcome_bin"])
            try:
                vals = {
                    "lcs_abs":    float(row["lcs_abs"]),
                    "lcs_int":    float(row["lcs_int"]),
                    "jaccard_n5": float(row["jaccard_n5"]),
                }
            except (ValueError, KeyError):
                continue
            bucket = pos if ob == 1 else neg if ob == 0 else None
            if bucket is not None:
                for k, v in vals.items():
                    bucket[k].append(v)

    for k in pos:
        pos[k] = np.array(pos[k])
        neg[k] = np.array(neg[k])

    print(f"  Infringed: {len(pos['lcs_abs'])}  Denied: {len(neg['lcs_abs'])}")

    print(f"\n{'='*60}")
    print("Pareto calibration (null=Denied, positive=Infringed)")
    print(f"{'='*60}")

    output = {
        "dataset":  "MCIC (Park et al., ISMIR 2024)",
        "null":     "Denied pairs (n=66)",
        "positive": "Infringed pairs (n=32)",
        "metrics":  {},
    }

    for metric in ["lcs_abs", "lcs_int", "jaccard_n5"]:
        output["metrics"][metric] = pareto_threshold(
            pos[metric], neg[metric], metric, direction="high"
        )

    # Recommended thresholds -- best F1 per metric
    recs_f1  = {m: output["metrics"][m]["best_f1"]["threshold"]  for m in output["metrics"]}
    recs_acc = {m: output["metrics"][m]["best_acc"]["threshold"] for m in output["metrics"]}

    output["recommended_best_f1"]  = recs_f1
    output["recommended_best_acc"] = recs_acc

    print(f"\n{'='*60}")
    print("SUMMARY -- recommended thresholds")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Best-F1 thr':>12} {'Best-Acc thr':>14}")
    print("-"*42)
    for m in recs_f1:
        print(f"{m:<15} {recs_f1[m]:>12.4f} {recs_acc[m]:>14.4f}")

    lcs_thr = recs_f1.get("lcs_int", recs_f1.get("lcs_abs", 0.2))
    jac_thr = recs_f1.get("jaccard_n5", 0.022)

    print(f"\nRe-run validation with calibrated thresholds:")
    print(f"  python scripts/run_mcic_validation.py \\")
    print(f"      --labels        data/mcic/labels.csv \\")
    print(f"      --out           results/validation/mcic_validation_calibrated.json \\")
    print(f"      --lcs_threshold {lcs_thr:.4f} \\")
    print(f"      --jac_threshold {jac_thr:.4f}")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

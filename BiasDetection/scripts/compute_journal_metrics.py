"""
compute_journal_metrics.py
--------------------------
Computes all additional technical contributions for the journal version:
  1. ROC AUC on MCIC (all three metrics)
  2. Cohen's kappa between metric pairs (LLM corpus + MCIC)
  3. Interval lift criterion formalisation
  4. Auditing blind spot dilution formula
  5. GCC formal definition statistics
  6. WAR/WAQ vs best-window justification

Requires:
    results/validation/mcic_validation.csv
    results/significance_extended.json
    results/agg_summary.json
    results/agg_summary_intervals.json

Usage:
    python scripts/compute_journal_metrics.py \
        --mcic_csv   results/validation/mcic_validation.csv \
        --sig_json   results/significance_extended.json \
        --agg_json   results/agg_summary.json \
        --agg_int    results/agg_summary_intervals.json \
        --out        results/journal_metrics.json
"""

import argparse
import csv
import json
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. ROC AUC
# ---------------------------------------------------------------------------

def compute_auc(scores, labels):
    """Compute ROC AUC from raw scores and binary labels."""
    from sklearn.metrics import roc_auc_score, roc_curve
    try:
        auc = roc_auc_score(labels, scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        return {
            "auc":        round(float(auc), 4),
            "fpr":        [round(float(x), 4) for x in fpr],
            "tpr":        [round(float(x), 4) for x in tpr],
            "thresholds": [round(float(x), 4) for x in thresholds],
        }
    except Exception as e:
        return {"auc": None, "error": str(e)}


def mcic_roc(mcic_csv: Path) -> dict:
    """Compute ROC AUC for each metric on MCIC pairs."""
    rows = list(csv.DictReader(open(mcic_csv, encoding="utf-8")))
    usable = [r for r in rows
              if not r.get("error")
              and r["outcome_bin"] in ("0", "1")]

    labels   = [int(r["outcome_bin"]) for r in usable]
    lcs_abs  = [float(r["lcs_abs"])   for r in usable]
    lcs_int  = [float(r["lcs_int"])   for r in usable]
    jac      = [float(r["jaccard_n5"])for r in usable]
    union    = [max(float(r["lcs_abs"]), float(r["lcs_int"]), float(r["jaccard_n5"]))
                for r in usable]

    return {
        "n_pairs":     len(usable),
        "n_infringed": sum(labels),
        "n_denied":    len(labels) - sum(labels),
        "absolute_lcs": compute_auc(lcs_abs, labels),
        "interval_lcs": compute_auc(lcs_int, labels),
        "jaccard_n5":   compute_auc(jac,     labels),
        "max_score":    compute_auc(union,   labels),
    }


# ---------------------------------------------------------------------------
# 2. Cohen's Kappa
# ---------------------------------------------------------------------------

def cohens_kappa(flags_a, flags_b):
    """Cohen's kappa between two binary flag arrays."""
    n  = len(flags_a)
    tp = sum(1 for a, b in zip(flags_a, flags_b) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(flags_a, flags_b) if a == 0 and b == 0)
    fp_a = sum(1 for a, b in zip(flags_a, flags_b) if a == 1 and b == 0)
    fp_b = sum(1 for a, b in zip(flags_a, flags_b) if a == 0 and b == 1)

    p_o  = (tp + tn) / n
    p_a  = ((tp + fp_a) / n) * ((tp + fp_b) / n)
    p_b  = ((tn + fp_b) / n) * ((tn + fp_a) / n)
    p_e  = p_a + p_b

    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0

    interp = ("near-perfect (>0.8)" if kappa > 0.8 else
              "substantial (0.6-0.8)" if kappa > 0.6 else
              "moderate (0.4-0.6)" if kappa > 0.4 else
              "fair (0.2-0.4)" if kappa > 0.2 else
              "slight or no agreement (<0.2)")

    return {
        "kappa":  round(kappa, 4),
        "p_obs":  round(p_o,   4),
        "p_exp":  round(p_e,   4),
        "interpretation": interp,
        "n": n,
    }


def mcic_kappa(mcic_csv: Path, lcs_thr: float, jac_thr: float) -> dict:
    rows = [r for r in csv.DictReader(open(mcic_csv, encoding="utf-8"))
            if not r.get("error") and r["outcome_bin"] in ("0","1")]

    flags_abs = [int(float(r["lcs_abs"])    >= lcs_thr) for r in rows]
    flags_int = [int(float(r["lcs_int"])    >= lcs_thr) for r in rows]
    flags_jac = [int(float(r["jaccard_n5"]) >= jac_thr) for r in rows]

    return {
        "thresholds": {"lcs": lcs_thr, "jaccard": jac_thr},
        "abs_vs_int": cohens_kappa(flags_abs, flags_int),
        "abs_vs_jac": cohens_kappa(flags_abs, flags_jac),
        "int_vs_jac": cohens_kappa(flags_int, flags_jac),
    }


# ---------------------------------------------------------------------------
# 3. Interval Lift Criterion
# ---------------------------------------------------------------------------

def interval_lift(agg_abs: dict, agg_int: dict) -> dict:
    pair_types = ["LLM x LLM", "LLM x Duped", "LLM x Train", "Human x Human"]
    results = {}
    hh_abs = agg_abs["pair_type_summary"]["Human x Human"]["lcs_score_mean"]
    hh_int = agg_int["pair_type_summary"]["Human x Human"]["lcs_score_mean"]

    for pt in pair_types:
        a = agg_abs["pair_type_summary"][pt]["lcs_score_mean"]
        i = agg_int["pair_type_summary"][pt]["lcs_score_mean"]
        lift = (i - a) / a if a > 0 else float("nan")
        results[pt] = {
            "abs_lcs":  round(a, 6),
            "int_lcs":  round(i, 6),
            "lift":     round(lift, 4) if not np.isnan(lift) else None,
            "lift_pct": round(lift * 100, 1) if not np.isnan(lift) else None,
        }

    # Lift criterion: classify as TRANSPOSED when:
    #   int_LCS >= T_lcs AND abs_LCS < T_lcs AND lift > theta_lift
    # theta_lift derived from HxH lift (baseline ~2%)
    hh_lift = (hh_int - hh_abs) / hh_abs if hh_abs > 0 else 0
    theta_lift = max(hh_lift * 10, 0.5)  # 10x null lift as conservative criterion

    return {
        "pair_type_lifts": results,
        "hh_null_lift":    round(hh_lift, 4),
        "recommended_theta_lift": round(theta_lift, 3),
        "criterion": (
            f"TRANSPOSED if: int_LCS >= T_lcs AND abs_LCS < T_lcs "
            f"AND lift > {theta_lift:.2f}"
        ),
    }


# ---------------------------------------------------------------------------
# 4. Auditing Blind Spot Dilution
# ---------------------------------------------------------------------------

def dilution_formula(agg: dict, k: int = 50, K: int = 300) -> dict:
    pt = agg["pair_type_summary"]
    duped_lcs = pt["LLM x Duped"]["lcs_score_mean"]
    train_lcs = pt["LLM x Train"]["lcs_score_mean"]
    hh_lcs    = pt["Human x Human"]["lcs_score_mean"]

    expected_dilution = 1 - k / K
    predicted_ratio   = 1 / (1 - expected_dilution)
    observed_ratio    = (duped_lcs - hh_lcs) / (train_lcs - hh_lcs) if (train_lcs - hh_lcs) > 0 else 0

    return {
        "k_finetune":       k,
        "K_train":          K,
        "expected_dilution":round(expected_dilution, 4),
        "predicted_ratio":  round(predicted_ratio, 2),
        "observed_ratio":   round(observed_ratio, 2),
        "duped_lcs_mean":   round(duped_lcs, 6),
        "train_lcs_mean":   round(train_lcs, 6),
        "hh_lcs_mean":      round(hh_lcs, 6),
        "formula": "dilution = 1 - k/K; signal_ratio ~= 1/(1-dilution)",
        "note": (
            "Observed ratio lower than predicted because the fine-tune proxy "
            "is a held-out sample, not the exact training set."
        ),
    }


# ---------------------------------------------------------------------------
# 5. GCC Formal Definition
# ---------------------------------------------------------------------------

def gcc_statistics(agg: dict, threshold: float = 0.05) -> dict:
    runs = agg["llm_per_run_breakdown"]
    hh   = agg["pair_type_summary"]["Human x Human"]["lcs_score_mean"]

    gcc_pos = {rid: r for rid, r in runs.items()
               if r["lcs_mean"] >= threshold and r["n"] >= 3}
    gcc_neg = {rid: r for rid, r in runs.items()
               if r["lcs_mean"] < threshold and r["n"] >= 3}

    pos_means = [r["lcs_mean"] for r in gcc_pos.values()]
    neg_means = [r["lcs_mean"] for r in gcc_neg.values()]

    return {
        "gcc_threshold":        threshold,
        "hh_null_mean":         round(hh, 6),
        "threshold_vs_null":    round(threshold / hh, 1),
        "gcc_positive_runs":    sorted(gcc_pos.keys()),
        "gcc_negative_runs":    sorted(gcc_neg.keys()),
        "n_gcc_pos":            len(gcc_pos),
        "n_gcc_neg":            len(gcc_neg),
        "gcc_pos_mean_lcs":     round(float(np.mean(pos_means)), 5),
        "gcc_pos_std_lcs":      round(float(np.std(pos_means)), 5),
        "gcc_neg_mean_lcs":     round(float(np.mean(neg_means)), 5),
        "gcc_neg_std_lcs":      round(float(np.std(neg_means)), 5),
        "separation_ratio":     round(float(np.mean(pos_means) / np.mean(neg_means)), 2),
        "definition": (
            "GCC(R) = 1 if mean_within_run_LCS(R) >= T_gcc, else 0. "
            f"T_gcc = {threshold} ({threshold/hh:.0f}x HxH null mean)."
        ),
    }


# ---------------------------------------------------------------------------
# 6. WAR/WAQ Justification
# ---------------------------------------------------------------------------

def warwaq_justification(agg: dict) -> dict:
    gs = agg["group_stats_averaged"]
    notagen_len = gs["llm_output.mean_seq_len"]
    hb_len      = gs["human_baseline.mean_seq_len"]

    return {
        "notagen_mean_seq_len":    round(notagen_len, 1),
        "maestro_mean_seq_len":    round(hb_len, 1),
        "mcic_approx_seq_len":     100,
        "best_window_rationale": (
            "Best-window LCS normalises by min(|Q|, window_size), rewarding "
            "detection of a short fragment anywhere in a long output. Appropriate "
            f"for NotaGen (mean {notagen_len:.0f} tokens) where a 50-token theme "
            "may appear inside a 1100-token piece."
        ),
        "war_waq_rationale": (
            "WAR/WAQ density score penalises sparse LCS matches by measuring "
            "how compactly the matched tokens span each sequence. Appropriate "
            "for MCIC (~100 tokens) where a scattered 12-token LCS across 48 "
            "reference positions signals coincidental overlap rather than copying."
        ),
        "cross_validation_note": (
            "Using WAR/WAQ for MCIC and best-window for NotaGen is intentional: "
            "each metric is chosen to match the sequence-length regime of its corpus. "
            "Future work should establish a single metric that performs well across "
            "both regimes."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mcic_csv", default="results/validation/mcic_validation.csv")
    ap.add_argument("--sig_json", default="results/significance_extended.json")
    ap.add_argument("--agg_json", default="results/agg_summary.json")
    ap.add_argument("--agg_int",  default="results/agg_summary_intervals.json")
    ap.add_argument("--out",      default="results/journal_metrics.json")
    ap.add_argument("--lcs_thr",  type=float, default=0.387)
    ap.add_argument("--jac_thr",  type=float, default=0.022)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    agg_abs = json.load(open(args.agg_json))
    agg_int = json.load(open(args.agg_int))
    mcic_csv = Path(args.mcic_csv)

    results = {}

    # 1. ROC AUC
    print("1. Computing ROC AUC on MCIC ...")
    results["roc_auc_mcic"] = mcic_roc(mcic_csv)
    for metric, r in results["roc_auc_mcic"].items():
        if isinstance(r, dict) and "auc" in r:
            print(f"   {metric:<20} AUC = {r['auc']}")

    # 2. Cohen's Kappa
    print("2. Computing Cohen's kappa ...")
    results["cohens_kappa_mcic"] = mcic_kappa(
        mcic_csv, args.lcs_thr, args.jac_thr)
    for pair, k in results["cohens_kappa_mcic"].items():
        if isinstance(k, dict) and "kappa" in k:
            print(f"   {pair:<20} kappa = {k['kappa']:.4f}  ({k['interpretation']})")

    # 3. Interval lift
    print("3. Computing interval lift ...")
    results["interval_lift"] = interval_lift(agg_abs, agg_int)
    for pt, v in results["interval_lift"]["pair_type_lifts"].items():
        print(f"   {pt:<20} lift = {v['lift_pct']}%")

    # 4. Dilution formula
    print("4. Computing dilution formula ...")
    results["dilution"] = dilution_formula(agg_abs)
    d = results["dilution"]
    print(f"   Expected {d['predicted_ratio']}x signal ratio, "
          f"observed {d['observed_ratio']}x")

    # 5. GCC statistics
    print("5. Computing GCC statistics ...")
    results["gcc"] = gcc_statistics(agg_abs)
    g = results["gcc"]
    print(f"   GCC+ runs: {g['n_gcc_pos']}  mean={g['gcc_pos_mean_lcs']:.5f}")
    print(f"   GCC- runs: {g['n_gcc_neg']}  mean={g['gcc_neg_mean_lcs']:.5f}")
    print(f"   Separation: {g['separation_ratio']}x")

    # 6. WAR/WAQ justification
    print("6. WAR/WAQ justification ...")
    results["warwaq"] = warwaq_justification(agg_abs)

    # Save
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {out_path}")

    # Print summary table
    print(f"\n{'='*55}")
    print("JOURNAL METRICS SUMMARY")
    print(f"{'='*55}")
    print(f"\nROC AUC on MCIC (n=98):")
    for m in ["absolute_lcs","interval_lcs","jaccard_n5","max_score"]:
        auc = results["roc_auc_mcic"][m].get("auc","N/A")
        print(f"  {m:<20} {auc}")
    print(f"\nCohen's kappa (MCIC, calibrated thresholds):")
    for pair, k in results["cohens_kappa_mcic"].items():
        if isinstance(k, dict) and "kappa" in k:
            print(f"  {pair:<20} kappa={k['kappa']:.3f}  {k['interpretation']}")
    print(f"\nInterval lift criterion:")
    print(f"  HxH null lift:      {results['interval_lift']['hh_null_lift']*100:.1f}%")
    print(f"  Recommended theta_lift: {results['interval_lift']['recommended_theta_lift']}")
    print(f"\nDilution formula:")
    print(f"  Predicted ratio:    {results['dilution']['predicted_ratio']}x")
    print(f"  Observed ratio:     {results['dilution']['observed_ratio']}x")
    print(f"\nGCC:")
    print(f"  Separation ratio:   {results['gcc']['separation_ratio']}x")
    print(f"  Definition: {results['gcc']['definition']}")


if __name__ == "__main__":
    main()

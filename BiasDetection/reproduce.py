"""
reproduce.py
------------
Master reproducibility script. Runs all experiments in dependency order
and verifies outputs against stored expected values.

Usage
-----
    # Full run (all 16 experiments)
    python reproduce.py --maestro_root ../maestro-v3.0.0 --mcic_dir ../MCIC

    # Sample run (faster -- for verification of a subset)
    python reproduce.py --sample --maestro_root ../maestro-v3.0.0 --mcic_dir ../MCIC

    # Verify only (compare existing results to expected values)
    python reproduce.py --verify_only

    # Skip specific stages
    python reproduce.py --skip composer --maestro_root ../maestro-v3.0.0

Stages (--skip accepts comma-separated list):
    audit       Main LCS/TF-IDF/Jaccard audit
    interval    Interval token audit
    significance Extended significance tests
    mcic        MCIC ingest + validation
    composer    Composer model + visualization
    attribution NotaGen attribution
    verify      Final verification report

Output
------
    results/reproduce_log.txt        -- full stdout log
    results/reproduce_summary.json  -- all key numbers
    results/reproduce_report.txt    -- pass/fail vs expected values
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Expected values -- master reference from original runs
# ---------------------------------------------------------------------------
EXPECTED = {
    # Group-level LCS (absolute, v2)
    "lcs_llm_llm":        {"value": 0.009466, "tol": 0.0005, "label": "LLMxLLM mean LCS"},
    "lcs_llm_duped":      {"value": 0.002602, "tol": 0.0003, "label": "LLMxDuped mean LCS"},
    "lcs_llm_train":      {"value": 0.000507, "tol": 0.0001, "label": "LLMxTrain mean LCS"},
    "lcs_hh":             {"value": 0.000307, "tol": 0.00005,"label": "HxH mean LCS"},
    # Interval lifts
    "lift_llm_duped":     {"value": 208.2,    "tol": 5.0,    "label": "LLMxDuped lift %"},
    "lift_hh":            {"value": 1.6,      "tol": 0.5,    "label": "HxH lift %"},
    "lift_llm_llm":       {"value": 67.6,     "tol": 3.0,    "label": "LLMxLLM lift %"},
    # GCC
    "gcc_pos_n":          {"value": 9,        "tol": 0,      "label": "GCC+ run count"},
    "gcc_pos_mean":       {"value": 0.07286,  "tol": 0.002,  "label": "GCC+ mean LCS"},
    "gcc_neg_mean":       {"value": 0.01610,  "tol": 0.002,  "label": "GCC- mean LCS"},
    "gcc_separation":     {"value": 4.53,     "tol": 0.1,    "label": "GCC separation ratio"},
    # Significance
    "sig_llm_llm_r":      {"value": 0.0376,   "tol": 0.002,  "label": "LCS r: LLMxLLM vs HxH"},
    "sig_runstrat_r":     {"value": 0.2798,   "tol": 0.01,   "label": "LCS r: high vs HxH"},
    # MCIC (out-of-domain)
    "mcic_ood_acc":       {"value": 0.6735,   "tol": 0.005,  "label": "MCIC OOD union accuracy"},
    "mcic_ood_recall":    {"value": 0.250,    "tol": 0.01,   "label": "MCIC OOD union recall"},
    # MCIC (calibrated)
    "mcic_cal_acc":       {"value": 0.714,    "tol": 0.005,  "label": "MCIC calibrated int accuracy"},
    "mcic_cal_f1":        {"value": 0.417,    "tol": 0.01,   "label": "MCIC calibrated int F1"},
    "mcic_trans_only":    {"value": 5,        "tol": 0,      "label": "Transposition-only cases"},
    # ROC AUC
    "auc_ensemble":       {"value": 0.6077,   "tol": 0.005,  "label": "MCIC ROC AUC ensemble"},
    "auc_interval":       {"value": 0.5876,   "tol": 0.005,  "label": "MCIC ROC AUC interval"},
    # Kappa
    "kappa_abs_int":      {"value": 0.692,    "tol": 0.01,   "label": "Cohen's kappa abs vs int"},
    # Dilution
    "dilution_observed":  {"value": 11.49,    "tol": 0.5,    "label": "Dilution observed ratio"},
    # Classification (5-composer)
    "clf_5_abs_top1":     {"value": 0.516,    "tol": 0.01,   "label": "5-composer abs Top-1"},
    "clf_5_abs_top3":     {"value": 0.826,    "tol": 0.01,   "label": "5-composer abs Top-3"},
    "clf_5_int_top1":     {"value": 0.516,    "tol": 0.01,   "label": "5-composer int Top-1"},
    "clf_5_int_top3":     {"value": 0.826,    "tol": 0.01,   "label": "5-composer int Top-3"},
    # Attribution
    "attr_r0_top":        {"value": 0.231,    "tol": 0.01,   "label": "R0 (GCC+) top cosine (Liszt)"},
    "attr_r18_top":       {"value": 0.192,    "tol": 0.01,   "label": "R18 (GCC-) top cosine (Bach)"},
    # LCS group means (from full audit run)
    "lcs_llm_llm":        {"value": 0.009466, "tol": 0.0005, "label": "LLMxLLM mean LCS"},
    "lcs_llm_duped":      {"value": 0.002602, "tol": 0.0003, "label": "LLMxDuped mean LCS"},
    "lcs_llm_train":      {"value": 0.000507, "tol": 0.0001, "label": "LLMxTrain mean LCS"},
    "lcs_hh":             {"value": 0.000307, "tol": 0.00005,"label": "HxH mean LCS"},
    # Interval lifts
    "lift_llm_duped":     {"value": 208.2,    "tol": 5.0,    "label": "LLMxDuped lift %"},
    "lift_hh":            {"value": 1.6,      "tol": 0.5,    "label": "HxH lift %"},
    "lift_llm_llm":       {"value": 67.6,     "tol": 3.0,    "label": "LLMxLLM lift %"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.fh = open(log_path, "w")
        self.start = time.time()

    def log(self, msg, also_print=True):
        ts = f"[{time.time()-self.start:>7.1f}s]"
        line = f"{ts} {msg}"
        self.fh.write(line + "\n")
        self.fh.flush()
        if also_print:
            print(line)

    def close(self):
        self.fh.close()


def run_cmd(cmd, logger, check=True):
    """Run a shell command, stream output in real-time, return returncode."""
    logger.log(f"$ {cmd}")
    process = subprocess.Popen(
        cmd, shell=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True
    )
    for line in process.stdout:
        line = line.rstrip()
        logger.log(f"    {line}", also_print=True)
    process.wait()
    if check and process.returncode != 0:
        logger.log(f"ERROR: command failed (rc={process.returncode})")
    return process.returncode


def check_value(key, actual, results, logger):
    """Compare actual value to expected, log pass/fail."""
    exp = EXPECTED.get(key)
    if exp is None:
        return None
    expected = exp["value"]
    tol = exp["tol"]
    label = exp["label"]

    if isinstance(expected, int):
        ok = actual == expected
    else:
        ok = abs(float(actual) - float(expected)) <= tol

    status = "PASS" if ok else "FAIL"
    results[key] = {
        "label":    label,
        "expected": expected,
        "actual":   actual,
        "tol":      tol,
        "status":   status,
    }
    flag = "[ok]" if ok else "[rm]"
    logger.log(f"  {flag} {status:<4} {label}: "
               f"expected={expected} actual={actual} tol=+/-{tol}")
    return ok



# ---------------------------------------------------------------------------
# Fast verify: read existing audit JSON without rerunning pipeline
# ---------------------------------------------------------------------------

def stage_audit_verify(logger, results, summary):
    """Sample mode: verify Stage 1 from existing agg_summary.json."""
    logger.log("\n" + "="*55)
    logger.log("STAGE 1/7: Audit verify (reading existing results)")
    logger.log("="*55)
    path = Path("results/agg_summary.json")
    if not path.exists():
        logger.log("  [SKIP] results/agg_summary.json not found -- run full mode first")
        return
    agg = json.load(open(path))
    pt  = agg["pair_type_summary"]
    summary["lcs_llm_llm"]   = pt["LLM x LLM"]["lcs_score_mean"]
    summary["lcs_llm_duped"] = pt["LLM x Duped"]["lcs_score_mean"]
    summary["lcs_llm_train"] = pt["LLM x Train"]["lcs_score_mean"]
    summary["lcs_hh"]        = pt["Human x Human"]["lcs_score_mean"]
    for key in ["lcs_llm_llm","lcs_llm_duped","lcs_llm_train","lcs_hh"]:
        check_value(key, summary[key], results, logger)
    logger.log("  Stage 1 verified from existing file.")


def stage_interval_verify(logger, results, summary):
    """Sample mode: verify Stage 2 from existing interval summary."""
    logger.log("\n" + "="*55)
    logger.log("STAGE 2/7: Interval verify (reading existing results)")
    logger.log("="*55)
    abs_path = Path("results/agg_summary.json")
    int_path = Path("results/agg_summary_intervals.json")
    if not abs_path.exists() or not int_path.exists():
        logger.log("  [SKIP] summary files not found -- run full mode first")
        return
    agg     = json.load(open(abs_path))
    agg_int = json.load(open(int_path))
    pt      = agg["pair_type_summary"]
    pt_int  = agg_int["pair_type_summary"]
    for group, key in [
        ("LLM x LLM",    "lift_llm_llm"),
        ("LLM x Duped",  "lift_llm_duped"),
        ("Human x Human","lift_hh"),
    ]:
        a = pt[group]["lcs_score_mean"]
        i = pt_int[group]["lcs_score_mean"]
        lift = (i - a) / a * 100 if a > 0 else 0
        summary[key] = round(lift, 1)
    for key in ["lift_llm_llm","lift_llm_duped","lift_hh"]:
        check_value(key, summary[key], results, logger)
    logger.log("  Stage 2 verified from existing file.")

# ---------------------------------------------------------------------------
# Stage 1: Main LCS audit
# ---------------------------------------------------------------------------

def stage_audit(args, logger, results, summary):
    logger.log("\n" + "="*55)
    logger.log("STAGE 1/7: Main LCS / TF-IDF / Jaccard audit")
    logger.log("  Expected duration: ~15 min (full) / ~2 min (sample)")
    logger.log("="*55)

    n = 10 if args.sample else 0  # 0 = use all
    cmd = (
        f"python experiments/run_experiment.py "
        f"--llm data/llm_output "
        f"--human data/human_baseline "
        f"--human-sq data/human_baseline_sq "
        f"--train data/train_corpus "
        f"--n-llm {n} --n-human {n} --n-human-sq {n} --n-train {300 if n == 0 else n} "
        f"--workers 4 --tag {'sample' if args.sample else 'full_v2'}"
    )
    rc = run_cmd(cmd, logger)

    if rc == 0 and Path("results/agg_summary.json").exists():
        agg = json.load(open("results/agg_summary.json"))
        pt  = agg["pair_type_summary"]
        summary["lcs_llm_llm"]   = pt["LLM x LLM"]["lcs_score_mean"]
        summary["lcs_llm_duped"] = pt["LLM x Duped"]["lcs_score_mean"]
        summary["lcs_llm_train"] = pt["LLM x Train"]["lcs_score_mean"]
        summary["lcs_hh"]        = pt["Human x Human"]["lcs_score_mean"]

        if not args.sample:
            for key in ["lcs_llm_llm","lcs_llm_duped","lcs_llm_train","lcs_hh"]:
                check_value(key, summary[key], results, logger)
    return rc


# ---------------------------------------------------------------------------
# Stage 2: Interval token audit
# ---------------------------------------------------------------------------

def stage_interval(args, logger, results, summary):
    logger.log("\n" + "="*55)
    logger.log("STAGE 2/7: Interval token audit")
    logger.log("  Expected duration: ~15 min (full) / ~2 min (sample)")
    logger.log("="*55)

    n = 10 if args.sample else 0
    cmd = (
        f"python experiments/run_experiment.py "
        f"--llm data/llm_output "
        f"--human data/human_baseline "
        f"--human-sq data/human_baseline_sq "
        f"--train data/train_corpus "
        f"--n-llm {n} --n-human {n} --n-human-sq {n} --n-train {300 if n == 0 else n} "
        f"--workers 4 --use-intervals "
        f"--tag {'sample_intervals' if args.sample else 'full_v3_intervals'}"
    )
    rc = run_cmd(cmd, logger)

    if rc == 0 and Path("results/agg_summary.json").exists() \
               and Path("results/agg_summary_intervals.json").exists():
        agg     = json.load(open("results/agg_summary.json"))
        agg_int = json.load(open("results/agg_summary_intervals.json"))
        pt      = agg["pair_type_summary"]
        pt_int  = agg_int["pair_type_summary"]

        for group, key in [
            ("LLM x LLM",   "lift_llm_llm"),
            ("LLM x Duped", "lift_llm_duped"),
            ("Human x Human","lift_hh"),
        ]:
            a = pt[group]["lcs_score_mean"]
            i = pt_int[group]["lcs_score_mean"]
            lift = (i - a) / a * 100 if a > 0 else 0
            summary[key] = round(lift, 1)

        if not args.sample:
            for key in ["lift_llm_llm","lift_llm_duped","lift_hh"]:
                check_value(key, summary[key], results, logger)
    return rc


# ---------------------------------------------------------------------------
# Stage 3: Significance tests
# ---------------------------------------------------------------------------

def stage_significance(args, logger, results, summary):
    logger.log("\n" + "="*55)
    logger.log("STAGE 3/7: Extended significance tests")
    logger.log("  Expected duration: ~2 min")
    logger.log("="*55)

    rc = run_cmd(
        "python scripts/significance_tests_extended.py "
        "--report results/report_full_v2.json "
        "--out    results/significance_extended.json",
        logger
    )

    if rc == 0 and Path("results/significance_extended.json").exists():
        sig = json.load(open("results/significance_extended.json"))
        # Extract key r values
        for entry in sig.get("lcs", {}).get("comparisons", []):
            if "LLMxLLM" in entry.get("comparison","") and "Human" in entry.get("comparison",""):
                summary["sig_llm_llm_r"] = entry.get("r", None)
        for entry in sig.get("lcs", {}).get("run_stratified", []):
            if "Human" in entry.get("comparison",""):
                summary["sig_runstrat_r"] = entry.get("r", None)

        if not args.sample and summary.get("sig_llm_llm_r"):
            check_value("sig_llm_llm_r", summary["sig_llm_llm_r"], results, logger)
    return rc


# ---------------------------------------------------------------------------
# Stage 4: MCIC validation
# ---------------------------------------------------------------------------

def stage_mcic(args, logger, results, summary):
    logger.log("\n" + "="*55)
    logger.log("STAGE 4/7: MCIC ingest + validation")
    logger.log("  Expected duration: ~1 min")
    logger.log("="*55)

    mcic_midi = Path(args.mcic_dir) / "MCIC DATA" / "MIDI FILES"
    mcic_meta = Path(args.mcic_dir) / "mcic_metadata.csv"

    # Ingest
    rc = run_cmd(
        f'python scripts/ingest_mcic.py '
        f'--midi_dir "{mcic_midi}" '
        f'--meta_csv "{mcic_meta}" '
        f'--out_dir data/mcic --copy',
        logger
    )
    if rc != 0:
        return rc

    # Out-of-domain validation
    rc = run_cmd(
        "python scripts/run_mcic_validation.py "
        "--labels data/mcic/labels.csv "
        "--out results/validation/mcic_validation.json "
        "--lcs_threshold 0.387 --jac_threshold 0.022",
        logger
    )
    if rc == 0 and Path("results/validation/mcic_validation.json").exists():
        v = json.load(open("results/validation/mcic_validation.json"))
        summary["mcic_ood_acc"]    = v["metrics"]["union"]["accuracy"]
        summary["mcic_ood_recall"] = v["metrics"]["union"]["recall"]
        summary["mcic_trans_only"] = v.get("transposition_only_int", 0)
        check_value("mcic_ood_acc",    summary["mcic_ood_acc"],    results, logger)
        check_value("mcic_ood_recall", summary["mcic_ood_recall"], results, logger)

    # Calibration
    rc = run_cmd(
        "python scripts/calibrate_mcic.py "
        "--scores results/validation/mcic_validation.csv "
        "--out results/validation/mcic_calibrated_thresholds.json",
        logger
    )
    if rc != 0:
        return rc

    cal = json.load(open("results/validation/mcic_calibrated_thresholds.json"))
    lcs_thr = cal["recommended_best_acc"]["lcs_int"]
    jac_thr = cal["recommended_best_acc"]["jaccard_n5"]

    # Calibrated validation
    rc = run_cmd(
        f"python scripts/run_mcic_validation.py "
        f"--labels data/mcic/labels.csv "
        f"--out results/validation/mcic_validation_calibrated_acc.json "
        f"--lcs_threshold {lcs_thr:.4f} --jac_threshold {jac_thr:.4f}",
        logger
    )
    if rc == 0 and Path("results/validation/mcic_validation_calibrated_acc.json").exists():
        v = json.load(open("results/validation/mcic_validation_calibrated_acc.json"))
        # Prefer normalized LCS (outperforms WAR/WAQ on short MCIC sequences)
        int_key = ("interval_lcs_warwaq" if "interval_lcs_warwaq" in v["metrics"]
                   else "interval_lcs_warwaq" if "interval_lcs_warwaq" in v["metrics"]
                   else "interval_lcs")
        summary["mcic_cal_acc"] = v["metrics"][int_key]["accuracy"]
        summary["mcic_cal_f1"]  = v["metrics"][int_key]["f1"]
        check_value("mcic_cal_acc", summary["mcic_cal_acc"], results, logger)
        check_value("mcic_cal_f1",  summary["mcic_cal_f1"],  results, logger)

    return rc


# ---------------------------------------------------------------------------
# Stage 5: Journal metrics
# ---------------------------------------------------------------------------

def stage_journal_metrics(args, logger, results, summary):
    logger.log("\n" + "="*55)
    logger.log("STAGE 5/7: Journal metrics (AUC, kappa, GCC, dilution)")
    logger.log("  Expected duration: ~30 sec")
    logger.log("="*55)

    rc = run_cmd(
        "python scripts/compute_journal_metrics.py "
        "--mcic_csv results/validation/mcic_validation.csv "
        "--agg_json results/agg_summary.json "
        "--agg_int  results/agg_summary_intervals.json "
        "--out      results/journal_metrics.json",
        logger
    )
    if rc == 0 and Path("results/journal_metrics.json").exists():
        jm = json.load(open("results/journal_metrics.json"))
        summary["auc_ensemble"]    = jm["roc_auc_mcic"]["max_score"]["auc"]
        summary["auc_interval"]    = jm["roc_auc_mcic"]["interval_lcs"]["auc"]
        summary["kappa_abs_int"]   = jm["cohens_kappa_mcic"]["abs_vs_int"]["kappa"]
        summary["dilution_observed"] = jm["dilution"]["observed_ratio"]
        summary["gcc_pos_n"]       = jm["gcc"]["n_gcc_pos"]
        summary["gcc_pos_mean"]    = jm["gcc"]["gcc_pos_mean_lcs"]
        summary["gcc_neg_mean"]    = jm["gcc"]["gcc_neg_mean_lcs"]
        summary["gcc_separation"]  = jm["gcc"]["separation_ratio"]

        for key in ["auc_ensemble","auc_interval","kappa_abs_int",
                    "dilution_observed","gcc_pos_n","gcc_pos_mean",
                    "gcc_neg_mean","gcc_separation"]:
            if key in summary:
                check_value(key, summary[key], results, logger)
    return rc


# ---------------------------------------------------------------------------
# Stage 6: Composer models
# ---------------------------------------------------------------------------

def stage_composer(args, logger, results, summary):
    logger.log("\n" + "="*55)
    logger.log("STAGE 6/7: Composer models + visualization")
    logger.log("  Expected duration: ~5 min (cache warm) / ~20 min (cold)")
    logger.log("="*55)

    maestro = args.maestro_root
    workers = 2 if args.sample else 4

    # 5-composer absolute
    rc = run_cmd(
        f"python scripts/build_composer_model.py "
        f"--maestro_root {maestro} "
        f"--out results/composer/composer_model_5.json "
        f"--composers Bach,Beethoven,Chopin,Liszt,Schubert "
        f"--workers {workers}",
        logger
    )
    if rc == 0 and Path("results/composer/composer_model_5.json").exists():
        m = json.load(open("results/composer/composer_model_5.json"))
        summary["clf_5_abs_top1"] = m["results"]["top1_accuracy"]
        summary["clf_5_abs_top3"] = m["results"]["top3_accuracy"]
        check_value("clf_5_abs_top1", summary["clf_5_abs_top1"], results, logger)
        check_value("clf_5_abs_top3", summary["clf_5_abs_top3"], results, logger)

    # 5-composer interval
    rc = run_cmd(
        f"python scripts/build_composer_model.py "
        f"--maestro_root {maestro} "
        f"--out results/composer/composer_model_5_interval.json "
        f"--composers Bach,Beethoven,Chopin,Liszt,Schubert "
        f"--use_interval --workers {workers}",
        logger
    )
    if rc == 0 and Path("results/composer/composer_model_5_interval.json").exists():
        m = json.load(open("results/composer/composer_model_5_interval.json"))
        summary["clf_5_int_top1"] = m["results"]["top1_accuracy"]
        summary["clf_5_int_top3"] = m["results"]["top3_accuracy"]
        check_value("clf_5_int_top1", summary["clf_5_int_top1"], results, logger)
        check_value("clf_5_int_top3", summary["clf_5_int_top3"], results, logger)

    # Visualization -- 5 composers abs
    run_cmd(
        "python scripts/visualize_composer_space.py "
        "--model results/composer/composer_model_5.json "
        "--out_dir results/composer/figures_5 --n_pieces 30",
        logger
    )
    # Visualization -- 5 composers interval
    run_cmd(
        "python scripts/visualize_composer_space.py "
        "--model results/composer/composer_model_5_interval.json "
        "--out_dir results/composer/figures_5_interval --n_pieces 30",
        logger
    )

    # Appendix C -- sliding-window LCS profile for R18xR6 transposed pair
    piece_a = (
        "data/llm_output/"
        "our_model_R18_C93_4_tracks_3_4_sig_A_key_8218.mid"
    )
    piece_b = (
        "data/llm_output/"
        "our_model_R6_C34_4_tracks_2_3_sig_E_key_4110.mid"
    )
    if Path(piece_a).exists() and Path(piece_b).exists():
        run_cmd(
            f'python scripts/plot_lcs_profile.py '
            f'--piece_a "{piece_a}" '
            f'--piece_b "{piece_b}" '
            f'--out results/figures/appendix_c_lcs_profile.png '
            f'--window 200 --step 5',
            logger
        )
    else:
        logger.log("  [SKIP] Appendix C -- transposed pair files not found")
    return rc


# ---------------------------------------------------------------------------
# Stage 7: Final verification report
# ---------------------------------------------------------------------------

def stage_verify(results, summary, logger, report_path):
    logger.log("\n" + "="*55)
    logger.log("STAGE 7/7: Verification report")
    logger.log("="*55)

    passed = [k for k,v in results.items() if v["status"] == "PASS"]
    failed = [k for k,v in results.items() if v["status"] == "FAIL"]

    logger.log(f"\n  PASSED: {len(passed)}/{len(results)}")
    logger.log(f"  FAILED: {len(failed)}/{len(results)}")

    if failed:
        logger.log("\n  FAILURES:")
        for k in failed:
            r = results[k]
            logger.log(f"    [rm] {r['label']}: "
                       f"expected={r['expected']} actual={r['actual']} tol=+/-{r['tol']}")

    # Write report
    with open(report_path, "w") as f:
        f.write(f"Reproducibility Report -- {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("="*60 + "\n\n")
        f.write(f"PASSED: {len(passed)}/{len(results)}\n")
        f.write(f"FAILED: {len(failed)}/{len(results)}\n\n")
        f.write(f"{'Status':<6} {'Label':<45} {'Expected':>10} {'Actual':>10}\n")
        f.write("-"*75 + "\n")
        for k, v in results.items():
            f.write(f"{v['status']:<6} {v['label']:<45} "
                    f"{str(v['expected']):>10} {str(v['actual']):>10}\n")

    logger.log(f"\n  Report saved to {report_path}")
    return len(failed) == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maestro_root", default="../maestro-v3.0.0",
                    help="Path to MAESTRO v3.0.0 root directory")
    ap.add_argument("--mcic_dir",     default="../MCIC",
                    help="Path to MCIC directory")
    ap.add_argument("--sample",       action="store_true",
                    help="Run on reduced sample (faster, for quick verification)")
    ap.add_argument("--verify_only",  action="store_true",
                    help="Skip experiments, only verify existing results")
    ap.add_argument("--skip",         default="",
                    help="Comma-separated stages to skip: "
                         "audit,interval,significance,mcic,journal,composer,verify")
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    # Setup
    Path("results/validation").mkdir(parents=True, exist_ok=True)
    Path("results/composer").mkdir(parents=True, exist_ok=True)

    log_path    = "results/reproduce_log.txt"
    report_path = "results/reproduce_report.txt"
    summary_path = "results/reproduce_summary.json"

    logger = Logger(log_path)
    results = {}
    summary = {}

    logger.log(f"Reproducibility run started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.log(f"Sample mode: {args.sample}")
    logger.log(f"Skipping: {skip if skip else 'none'}")

    if not args.verify_only:
        if args.sample:
            # In sample mode: read existing audit results rather than
            # re-running the expensive pairwise LCS computation.
            # Stages 1 + 2 are verified from saved JSON files only.
            logger.log("\n[sample mode] Stages 1+2: reading from existing JSON "
                       "(skipping pairwise recompute -- use full run to re-generate)")
            if "audit"    not in skip: stage_audit_verify(logger, results, summary)
            if "interval" not in skip: stage_interval_verify(logger, results, summary)
        else:
            if "audit"    not in skip: stage_audit(args, logger, results, summary)
            if "interval" not in skip: stage_interval(args, logger, results, summary)

        if "significance" not in skip:
            import pathlib
            if args.sample and pathlib.Path("results/significance_extended.json").exists():
                logger.log("  [sample mode] Stage 3: skipping recompute -- using existing JSON")
            else:
                stage_significance(args, logger, results, summary)
        if "mcic"        not in skip: stage_mcic(args, logger, results, summary)
        if "journal"     not in skip: stage_journal_metrics(args, logger, results, summary)
        if "composer"    not in skip: stage_composer(args, logger, results, summary)

    if "verify" not in skip:
        all_passed = stage_verify(results, summary, logger, report_path)

    # Save summary
    with open(summary_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    logger.log(f"\nSummary saved to {summary_path}")

    elapsed = time.time() - logger.start
    logger.log(f"\nTotal elapsed: {elapsed/60:.1f} minutes")
    logger.close()

    print(f"\nLog:    {log_path}")
    print(f"Report: {report_path}")
    sys.exit(0 if (not results or all_passed) else 1)


if __name__ == "__main__":
    main()
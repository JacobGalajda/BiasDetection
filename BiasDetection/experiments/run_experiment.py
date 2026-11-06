"""
run_experiment.py
-----------------
Full bias-detection experiment with dual-baseline design:

  llm_output       : NotaGen string quartet outputs (ABC)
  human_baseline   : MAESTRO piano (instrumentation-diverse null)
  human_baseline_sq: PDMX 4-track pieces (instrumentation-matched null)
  train_corpus     : PDMX <=4 tracks, excl. SQ split (suspected source)

The Pareto calibrator uses human_baseline_sq as the null when provided,
since it is instrumentation-matched to the LLM output. MAESTRO is loaded
as a comparison group but not used for threshold calibration.

Usage:
    # Quick test
    python experiments/run_experiment.py --n-samples 20 --workers 8

    # Full dual-baseline run
    python experiments/run_experiment.py \\
        --llm         data/llm_output \\
        --human       data/human_baseline \\
        --human-sq    data/human_baseline_sq \\
        --train       data/train_corpus \\
        --n-llm   0 --n-human 0 --n-human-sq 0 --n-train 300 \\
        --workers 8 --tag full_v2
"""

import warnings
warnings.filterwarnings("ignore", category=Warning, module="requests")
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*charset_normalizer.*")

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt = "%H:%M:%S",
)

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from music_bias_analyzer.pipeline import (
    CorpusGroup, PipelineConfig,
    ThresholdCalibrator, EnsembleConfig,
)
from music_bias_analyzer.visualise import save_all_figures
from pipeline_lean import AnalysisPipelineLean as AnalysisPipelineFast, FastConfig


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Corpus paths
    p.add_argument("--llm",        default="data/llm_output")
    p.add_argument("--human",      default="data/human_baseline",
                   help="MAESTRO piano baseline")
    p.add_argument("--human-sq",   default="data/human_baseline_sq",
                   help="PDMX 4-track matched baseline (optional -- "
                        "skip if directory doesn't exist)")
    p.add_argument("--train",      default="data/train_corpus")
    p.add_argument("--duped",      default=None)

    # Sample limits (0 = no limit)
    p.add_argument("--n-samples",  type=int, default=0,
                   help="Default limit for all groups (0 = no limit)")
    p.add_argument("--n-llm",      type=int, default=None)
    p.add_argument("--n-human",    type=int, default=None)
    p.add_argument("--n-human-sq", type=int, default=None)
    p.add_argument("--n-train",    type=int, default=None)

    # Pipeline settings
    p.add_argument("--workers",     type=int, default=0)
    p.add_argument("--max-seq-len", type=int, default=2500)
    p.add_argument("--min-jaccard", type=float, default=0.02)
    p.add_argument("--no-calibrate",  action="store_true")
    p.add_argument("--no-ensemble",   action="store_true")
    p.add_argument("--use-intervals", action="store_true",
                   help="Transposition-invariant interval tokenisation")

    # Output
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default="")
    return p.parse_args()


def limit(global_default, override):
    return override if override is not None else global_default


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag          = f"_{args.tag}" if args.tag else ""
    report_path  = out_dir / f"report{tag}.json"
    figures_dir  = out_dir / f"figures{tag}"

    # -- Config ---------------------------------------------------------------
    config = PipelineConfig(
        ngram_ns             = [3, 5, 8],
        ngram_flag_threshold = 0.15,
        lcs_cross_group_only = False,
        lcs_flag_threshold   = 0.35,
        lcs_max_seq_len      = args.max_seq_len,
        lcs_ignore_duration  = False,
        lcs_ignore_octave    = False,
        tfidf_ngram_range    = (1, 3),
        tfidf_max_features   = 5000,
        use_octave           = True,
        use_duration         = True,
    )

    fast_config = FastConfig(
        n_workers           = args.workers,
        chunk_size          = 64,
        min_jaccard_for_lcs = args.min_jaccard,
        keep_tfidf_sparse   = True,
        use_window          = True,
        use_intervals       = args.use_intervals,
    )

    if args.use_intervals:
        print("[run_experiment] Interval mode: transposition-invariant tokenisation enabled",
              flush=True)

    calibrator = None if args.no_calibrate else ThresholdCalibrator()
    ensemble   = None if args.no_ensemble  else EnsembleConfig(
        metrics     = ["jaccard_n5", "lcs_strict", "tfidf_dist"],
        require_all = True,
    )

    # -- Load corpora ----------------------------------------------------------
    pipe = AnalysisPipelineFast(config=config, fast_config=fast_config)

    pipe.add_corpus(CorpusGroup.LLM_OUTPUT,
                    args.llm,
                    limit=limit(args.n_samples, args.n_llm))

    pipe.add_corpus(CorpusGroup.HUMAN_BASELINE,
                    args.human,
                    limit=limit(args.n_samples, args.n_human))

    # Instrumentation-matched SQ baseline -- loaded as DUPED group so it
    # flows through all pair comparisons without interfering with the
    # calibrator null (which uses HUMAN_BASELINE).
    # Rename the group label in post-processing via aggregate_report.py.
    sq_dir = Path(args.human_sq)
    has_sq = sq_dir.exists() and any(sq_dir.iterdir())
    if has_sq:
        pipe.add_corpus(CorpusGroup.DUPED,
                        args.human_sq,
                        limit=limit(args.n_samples, args.n_human_sq))
        logging.info("human_baseline_sq loaded as DUPED group from %s", args.human_sq)
    else:
        logging.warning("human_baseline_sq not found at %s -- single-baseline mode",
                        args.human_sq)

    train_dir = Path(args.train)
    if train_dir.exists() and any(train_dir.iterdir()):
        pipe.add_corpus(CorpusGroup.TRAIN_CORPUS,
                        args.train,
                        limit=limit(args.n_samples, args.n_train))
    else:
        logging.warning("Train corpus not found at %s", args.train)

    if args.duped:
        duped_dir = Path(args.duped)
        if duped_dir.exists():
            pipe.add_corpus(CorpusGroup.DUPED, args.duped)

    # -- Runtime estimate -----------------------------------------------------
    n_pieces = sum(len(c) for c in pipe._groups.values())
    n_pairs  = n_pieces * (n_pieces - 1) // 2
    workers  = args.workers or __import__("os").cpu_count()
    est_s    = n_pairs * 0.05 / max(workers, 1)
    logging.info("Pieces: %d | Pairs: %d | Workers: %d | Est. LCS: %.0fs (~%.0fmin)",
                 n_pieces, n_pairs, workers, est_s, est_s / 60)

    # -- Run -------------------------------------------------------------------
    t_start = time.time()
    report  = pipe.run(calibrator=calibrator, ensemble=ensemble)

    # -- Save -----------------------------------------------------------------
    report.save(report_path)
    logging.info("Report saved -> %s", report_path)
    save_all_figures(report, output_dir=str(figures_dir))

    # -- Print summary ---------------------------------------------------------
    print(report.summary())

    if report.calibrated_thresholds:
        ct = report.calibrated_thresholds
        print(f"\n-- Calibrated thresholds ({ct.calibration_source}) --")
        print(f"  lcs_strict   : {ct.lcs_strict}")
        print(f"  jaccard_n5   : {ct.jaccard_n5}")
        print(f"  tfidf_dist   : {ct.tfidf_dist_max}  (flag if BELOW)")
        print(f"  hist_dist    : {ct.hist_dist_max}")
        print(f"  null pairs   : {ct.n_null_pairs}  "
              f"(from MAESTRO human_baseline)")
        print(f"  cand pairs   : {ct.n_candidate_pairs}")

        if ct.n_candidate_pairs == 0:
            print("\n  NOTE: n_candidate_pairs=0 means no LLM x Train pairs "
                  "found for Pareto calibration.")
            print("  This is a FINDING -- NotaGen output does not detectably "
                  "overlap with PDMX training data at the token level.")
            print("  Use 2-sigma thresholds from null instead:")
            print("  Run: python scripts/calibrate.py --n-samples 30")

        if has_sq:
            print(f"\n  NOTE: human_baseline_sq (PDMX 4-track) loaded as 'duped' group.")
            print(f"  In aggregate_report.py output, 'LLM x Duped' = LLM vs SQ human.")
            print(f"  'Human x Duped' = MAESTRO vs SQ human (cross-instrumentation).")

    print(f"\nTotal wall time: {time.time()-t_start:.0f}s "
          f"({(time.time()-t_start)/60:.1f} min)")


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    main()
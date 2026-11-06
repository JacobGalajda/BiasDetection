"""
run_full_analysis.py
--------------------
Clean canonical experiment script for the NotaGen bias study.

Key differences from previous runs:
  - lcs_cross_group_only=False  -> Human x Human LCS is computed (required for
                                   null baseline; previous runs had this True,
                                   producing LCS=0 for all human pairs)
  - ThresholdCalibrator enabled -> thresholds derived from your actual null
                                   distribution instead of hardcoded defaults
  - EnsembleConfig enabled      -> AND-gate across jaccard_n5 + lcs_strict +
                                   tfidf_dist for publication-quality flagging
  - Single report saved         -> no more merging two reports with different
                                   ngram configs (n5-only vs n3/5/8)
  - ngram_ns = [3, 5, 8]        -> full ngram set in the single output

Usage:
    python experiments/run_full_analysis.py
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

# Allow running from repo root or experiments/
sys.path.insert(0, str(Path(__file__).parent.parent))

from music_bias_analyzer.pipeline import (
    AnalysisPipeline,
    CorpusGroup,
    PipelineConfig,
    ThresholdCalibrator,
    EnsembleConfig,
)
from music_bias_analyzer.visualize import save_all_figures

# ---------------------------------------------------------------------------
# Paths -- edit these
# ---------------------------------------------------------------------------

DATA_ROOT  = Path("data")
LLM_DIR    = DATA_ROOT / "llm_output"
HUMAN_DIR  = DATA_ROOT / "human_baseline"
TRAIN_DIR  = DATA_ROOT / "train_corpus"   # leave as None if not yet available
DUPED_DIR  = DATA_ROOT / "duped"          # leave as None if not yet available

RESULTS_DIR = Path("results")
REPORT_PATH = RESULTS_DIR / "report.json"
FIGURES_DIR = RESULTS_DIR / "figures"

# Set to 0 for no limit (full corpus)
LLM_LIMIT   = 5
HUMAN_LIMIT = 5

# ---------------------------------------------------------------------------
# Config -- the canonical settings
# ---------------------------------------------------------------------------

config = PipelineConfig(
    # N-gram: full set so the report has n3/5/8 in one file
    ngram_ns            = [3, 5, 8],
    ngram_flag_threshold = 0.15,   # overridden by calibrator if provided

    # LCS: MUST be False so Human x Human pairs are computed for null baseline
    lcs_cross_group_only = False,
    lcs_flag_threshold   = 0.35,   # overridden by calibrator if provided
    lcs_max_seq_len      = 4000,
    lcs_ignore_duration  = False,
    lcs_ignore_octave    = False,

    # Embeddings
    tfidf_ngram_range  = (1, 3),
    tfidf_max_features = 5000,

    # Tokeniser
    use_octave   = True,
    use_duration = True,
)

# ---------------------------------------------------------------------------
# Calibrator -- derives thresholds from Human x Human null distribution
# Requires both LLM_OUTPUT and HUMAN_BASELINE corpora to be loaded.
# Set to None to fall back to manual thresholds (not recommended for publication).
# ---------------------------------------------------------------------------

calibrator = ThresholdCalibrator()

# Ensemble: AND-gate across three metrics -- reduces false positives
# substantially compared to any single metric.
ensemble = EnsembleConfig(
    metrics      = ["jaccard_n5", "lcs_strict", "tfidf_dist"],
    require_all  = True,
)

# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------

pipe = AnalysisPipeline(config=config)

pipe.add_corpus(CorpusGroup.LLM_OUTPUT,     LLM_DIR,   limit=LLM_LIMIT)
pipe.add_corpus(CorpusGroup.HUMAN_BASELINE, HUMAN_DIR, limit=HUMAN_LIMIT)

# Add train corpus and duped only if the directories exist
if TRAIN_DIR and TRAIN_DIR.exists():
    pipe.add_corpus(CorpusGroup.TRAIN_CORPUS, TRAIN_DIR)
else:
    logging.warning("TRAIN_CORPUS not found at %s -- skipping", TRAIN_DIR)

if DUPED_DIR and DUPED_DIR.exists():
    pipe.add_corpus(CorpusGroup.DUPED, DUPED_DIR)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

logging.info("Starting full analysis ...")

report = pipe.run(
    calibrator = calibrator,
    ensemble   = ensemble,
)

# ---------------------------------------------------------------------------
# Save report + figures
# ---------------------------------------------------------------------------

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
report.save(REPORT_PATH)
logging.info("Report saved -> %s", REPORT_PATH)

save_all_figures(report, output_dir=str(FIGURES_DIR))
logging.info("Figures saved -> %s/", FIGURES_DIR)

# ---------------------------------------------------------------------------
# Print summary + calibrated thresholds
# ---------------------------------------------------------------------------

print(report.summary())

if report.calibrated_thresholds:
    ct = report.calibrated_thresholds
    print("\n-- Calibrated thresholds (Pareto, from null distribution) --")
    print(f"  jaccard_n5   : {ct.jaccard_n5}")
    print(f"  bleu_n5      : {ct.bleu_n5}")
    print(f"  lcs_strict   : {ct.lcs_strict}")
    print(f"  tfidf_dist   : {ct.tfidf_dist_max}  (flag if BELOW this)")
    print(f"  hist_dist    : {ct.hist_dist_max}   (flag if BELOW this)")
    print(f"  null pairs   : {ct.n_null_pairs}")
    print(f"  candidate pairs: {ct.n_candidate_pairs}")
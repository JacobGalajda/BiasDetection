"""
calibrate.py
------------
Quick calibration run to determine good PipelineConfig parameters
before committing to a full analysis.

Now that train_corpus is available, the Pareto calibrator can use
LLM x Train pairs as the positive distribution -- giving real, meaningful
thresholds rather than falling back to manual defaults.

Usage:
    python scripts/calibrate.py
    python scripts/calibrate.py --n-samples 15
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")

sys.path.insert(0, str(Path(__file__).parent.parent))

from music_bias_analyzer.pipeline import (
    AnalysisPipeline, CorpusGroup, PipelineConfig, ThresholdCalibrator,
)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--llm",   default="data/llm_output")
parser.add_argument("--human", default="data/human_baseline")
parser.add_argument("--train", default="data/train_corpus")
parser.add_argument("--n-samples", type=int, default=12,
                    help="Files per group (default 12 -> ~66 pairs each)")
args = parser.parse_args()

N = args.n_samples

# ---------------------------------------------------------------------------
# Config -- fast pass
# ---------------------------------------------------------------------------

config = PipelineConfig(
    ngram_ns             = [5],
    lcs_cross_group_only = False,   # need Human x Human for null baseline
    lcs_max_seq_len      = 2000,
    use_octave           = True,
    use_duration         = True,
)

pipe = AnalysisPipeline(config=config)
pipe.add_corpus(CorpusGroup.LLM_OUTPUT,     args.llm,   limit=N)
pipe.add_corpus(CorpusGroup.HUMAN_BASELINE, args.human, limit=N)

train_dir = Path(args.train)
if train_dir.exists() and any(train_dir.iterdir()):
    pipe.add_corpus(CorpusGroup.TRAIN_CORPUS, args.train, limit=N)
    has_train = True
else:
    logging.warning("train_corpus empty or missing -- calibrator will use manual defaults")
    has_train = False

calibrator = ThresholdCalibrator()
report     = pipe.run(calibrator=calibrator)
ct         = report.calibrated_thresholds

# ---------------------------------------------------------------------------
# Extract distributions
# ---------------------------------------------------------------------------

def pairs_of(ga, gb):
    return [p for p in report.pair_results
            if p.group_a == ga and p.group_b == gb
            or p.group_a == gb and p.group_b == ga]

null_pairs  = pairs_of("human_baseline", "human_baseline")
llm_pairs   = pairs_of("llm_output",     "llm_output")
cross_pairs = pairs_of("llm_output",     "train_corpus")

def stats(vals, label):
    if not vals:
        return f"    {label:<18}  (no data)"
    a = np.array(vals)
    return (f"    {label:<18}  mean={a.mean():.3f}  std={a.std():.3f}  "
            f"p95={np.percentile(a,95):.3f}  max={a.max():.3f}")

W = 64
print("\n" + "=" * W)
print("  CALIBRATION RESULTS")
print(f"  n_samples per group : {N}")
print(f"  Train corpus loaded : {has_train}")
print(f"  Calibration source  : {ct.calibration_source}")
print(f"  Null pairs (HxH)    : {ct.n_null_pairs}")
print(f"  Candidate pairs (LxT): {ct.n_candidate_pairs}")
print("=" * W)

print("\n  Null distribution (Human x Human):")
print(stats([p.lcs_score             for p in null_pairs], "LCS"))
print(stats([p.tfidf_cosine_dist     for p in null_pairs], "TF-IDF dist"))
print(stats([p.histogram_cosine_dist for p in null_pairs], "Hist dist"))

print("\n  LLM x LLM distribution:")
print(stats([p.lcs_score for p in llm_pairs], "LCS"))

if has_train:
    print("\n  LLM x Train distribution (candidates):")
    print(stats([p.lcs_score             for p in cross_pairs], "LCS"))
    print(stats([p.tfidf_cosine_dist     for p in cross_pairs], "TF-IDF dist"))
    print(stats([p.histogram_cosine_dist for p in cross_pairs], "Hist dist"))

print("\n  Pareto-calibrated thresholds:")
print(f"    lcs_flag_threshold   = {ct.lcs_strict:<8}  (manual default: 0.35)")
print(f"    ngram_flag_threshold = {ct.jaccard_n5:<8}  (manual default: 0.15)")
print(f"    tfidf_dist_max       = {ct.tfidf_dist_max:<8}  (manual default: 0.25, flag if BELOW)")
print(f"    hist_dist_max        = {ct.hist_dist_max:<8}  (manual default: 0.10, flag if BELOW)")

if ct.calibration_source != "pareto":
    print("\n  WARNING: Calibrator returned manual defaults.")
    print("  This means n_candidate_pairs=0 -- train corpus may be empty or")
    print("  not parsed correctly. Check data/train_corpus/ has .mid or .mxl files.")

# Recommended lcs_max_seq_len
all_lens = [len(s) for corpus in pipe._groups.values() for s in corpus.values()]
p95_len  = int(np.percentile(all_lens, 95))

print(f"\n  Token length p95 : {p95_len}")
print(f"  Recommended lcs_max_seq_len = {min(p95_len + 200, 4000)}")

print(f"""
  Copy this into run_full_analysis.py:

    config = PipelineConfig(
        ngram_ns             = [3, 5, 8],
        ngram_flag_threshold = {ct.jaccard_n5},
        lcs_cross_group_only = False,
        lcs_flag_threshold   = {ct.lcs_strict},
        lcs_max_seq_len      = {min(p95_len + 200, 4000)},
    )
""")
print("=" * W)
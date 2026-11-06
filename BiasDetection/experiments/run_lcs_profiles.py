"""
experiments/run_lcs_profiles.py
---------------------------------
Loads a completed report.json, finds the top flagged pairs, and generates
sliding-window LCS profile plots for each -- showing *where* in the reference
piece the LLM output most closely matches.

Usage:
    python experiments/run_lcs_profiles.py [results/report.json] [--top N]
"""

import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

from music_bias_analyzer.tokenizer import file_to_tokens, load_corpus
from music_bias_analyzer.lcs import sliding_window_lcs
from music_bias_analyzer.visualise import plot_lcs_profile
from music_bias_analyzer import CorpusGroup

parser = argparse.ArgumentParser()
parser.add_argument("report", nargs="?", default="results/report.json")
parser.add_argument("--top", type=int, default=5)
parser.add_argument("--window", type=int, default=50)
args = parser.parse_args()

with open(args.report) as f:
    data = json.load(f)

flagged = [p for p in data["pair_results"] if p["flagged_lcs"] or p["flagged_ngram"]]
flagged.sort(key=lambda x: x["lcs_score"], reverse=True)
flagged = flagged[:args.top]

if not flagged:
    print("No flagged pairs found in report.")
    sys.exit(0)

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
corpus = {}
for group in ["llm_output", "train_corpus", "duped", "human_baseline"]:
    d = os.path.join(DATA, group)
    if os.path.isdir(d):
        corpus.update(load_corpus(d))

os.makedirs("results/figures", exist_ok=True)

for i, pair in enumerate(flagged):
    na, nb = pair["piece_a"], pair["piece_b"]
    if na not in corpus or nb not in corpus:
        print(f"Skipping {na} <-> {nb}: files not found in data/")
        continue

    profile = sliding_window_lcs(corpus[na], corpus[nb], window=args.window)
    fig = plot_lcs_profile(profile, query_name=na, reference_name=nb,
                           flag_threshold=0.35)
    out = f"results/figures/lcs_profile_{i+1:02d}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}  (lcs={pair['lcs_score']:.3f})")

    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except ImportError:
        pass

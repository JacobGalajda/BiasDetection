"""
experiments/run_single_query.py
--------------------------------
Compare one LLM-generated piece against the full training corpus.
Useful for quickly auditing a specific suspected output before running
the full pairwise analysis.

Usage:
    python experiments/run_single_query.py path/to/suspect_piece.abc
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

from music_bias_analyzer import AnalysisPipeline, CorpusGroup
from music_bias_analyzer.tokenizer import file_to_tokens

if len(sys.argv) < 2:
    print("Usage: python run_single_query.py <path_to_music_file>")
    sys.exit(1)

query_path = sys.argv[1]
DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

query_tokens = file_to_tokens(query_path)
if not query_tokens:
    print(f"ERROR: Could not tokenize {query_path}")
    sys.exit(1)

print(f"\nQuery: {query_path}  ({len(query_tokens)} tokens)\n")

pipe = AnalysisPipeline()
pipe.add_corpus(CorpusGroup.TRAIN_CORPUS,   os.path.join(DATA, "train_corpus"))
pipe.add_corpus(CorpusGroup.HUMAN_BASELINE, os.path.join(DATA, "human_baseline"))

results = pipe.query_single(query_tokens, query_name=os.path.basename(query_path))

print(f"{'Reference':<40} {'Group':<16} {'LCS':>6}  {'Jac_n5':>7}  {'Jac_n3':>7}")
print("-" * 85)
for r in results[:20]:   # top 20 matches
    flag = " <" if r["lcs_sim"] > 0.35 or r.get("jaccard_n5", 0) > 0.15 else ""
    print(f"{r['reference']:<40} {r['group']:<16} "
          f"{r['lcs_sim']:>6.3f}  {r.get('jaccard_n5',0):>7.4f}  "
          f"{r.get('jaccard_n3',0):>7.4f}{flag}")

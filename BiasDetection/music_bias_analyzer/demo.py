"""
demo.py
-------
Smoke-test and demonstration of all analysis modules using synthetic
token sequences (no music files or music21 needed).

This also serves as a template for the real experiment:
  - Swap SYNTHETIC_CORPUS entries for real tokenised sequences
  - Add CorpusGroup.TRAIN_CORPUS, DUPED, etc. entries
  - Call pipeline.run() for full pairwise analysis

Run:
    python demo.py
"""

import sys, os
# Allow running as: python demo.py (inside the package dir) or
#                   python music_bias_analyzer/demo.py (from parent)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # parent of package

import numpy as np
import logging
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s  %(name)s  %(message)s")

# -- Import library (without music21) --------------------------------------

from music_bias_analyzer.ngram      import (jaccard_overlap, bleu_precision,
                                            multi_n_overlap,
                                            pairwise_ngram_matrix,
                                            flag_high_overlap_pairs)
from music_bias_analyzer.lcs        import (lcs_similarity, lcs_length,
                                            sliding_window_lcs, peak_lcs_region)
from music_bias_analyzer.embeddings import (TFIDFMusicEmbedder,
                                            pitch_class_histogram,
                                            interval_histogram,
                                            corpus_feature_matrix,
                                            cosine_distance_matrix,
                                            style_collapse_score, pca_2d)

# -- Synthetic corpora -----------------------------------------------------

def _melody(pitches, dur="quarter"):
    return [f"{p}_{dur}" for p in pitches]

# Simulate three groups:
#   LLM output: closely mirrors Training piece A (suspected copy)
#   Training  : two distinct pieces
#   Human     : clearly different style

TRAINING = {
    "train/bach_invention_A": _melody(
        ["C4","D4","E4","F4","G4","A4","B4","C5",
         "B4","A4","G4","F4","E4","D4","C4","REST",
         "G4","A4","B4","C5","D5","E5","F5","G5",
         "C4","E4","G4","C5","G4","E4","C4","REST"]),
    "train/bach_invention_B": _melody(
        ["E4","F#4","G4","A4","B4","C5","D5","E5",
         "D5","C5","B4","A4","G4","F#4","E4","REST",
         "A4","B4","C5","D5","E5","F#5","G5","A5",
         "E4","G4","B4","E5","B4","G4","E4","REST"]),
}

LLM_OUTPUT = {
    # Near-copy of bach_invention_A with minor transposition (C->D)
    "llm/notagen_output_1": _melody(
        ["D4","E4","F#4","G4","A4","B4","C#5","D5",
         "C#5","B4","A4","G4","F#4","E4","D4","REST",
         "A4","B4","C#5","D5","E5","F#5","G5","A5",
         "D4","F#4","A4","D5","A4","F#4","D4","REST"]),
    # More divergent - stylistically similar but different material
    "llm/notagen_output_2": _melody(
        ["C4","E4","G4","C5","E5","G5","C6","G5",
         "E5","C5","G4","E4","C4","REST","REST","REST",
         "G3","B3","D4","G4","B4","D5","G5","D5",
         "B4","G4","D4","B3","G3","REST","REST","REST"]),
}

HUMAN_BASELINE = {
    "maestro/chopin_nocturne_fragment": _melody(
        ["E5","D#5","E5","D#5","E5","B4","D5","C5",
         "A4","REST","C4","E4","A4","B4","REST","E4",
         "G#4","B4","C5","REST","E4","E5","D#5","E5",
         "D#5","E5","B4","D5","C5","A4","REST","REST"]),
    "maestro/mozart_sonata_fragment": _melody(
        ["C5","B4","A4","G4","F4","E4","D4","C4",
         "G4","F4","E4","D4","C4","B3","A3","G3",
         "C4","D4","E4","F4","G4","A4","B4","C5",
         "G5","F5","E5","D5","C5","B4","A4","G4"]),
}

ALL_CORPUS = {**TRAINING, **LLM_OUTPUT, **HUMAN_BASELINE}

FLAT_GROUP_MAP = {
    **{k: "train_corpus"   for k in TRAINING},
    **{k: "llm_output"     for k in LLM_OUTPUT},
    **{k: "human_baseline" for k in HUMAN_BASELINE},
}

print("\n" + "="*60)
print("MUSIC BIAS ANALYZER -- DEMO")
print("="*60)

# -------------------------------------------------------------------------
# 1. N-GRAM ANALYSIS
# -------------------------------------------------------------------------
print("\n[1] N-GRAM OVERLAP (verbatim copy detection)")
print("-"*50)

llm1_seq    = LLM_OUTPUT["llm/notagen_output_1"]
train_a_seq = TRAINING["train/bach_invention_A"]
human_seq   = HUMAN_BASELINE["maestro/chopin_nocturne_fragment"]

for ref_name, ref_seq in [("train/bach_invention_A", train_a_seq),
                           ("maestro/chopin_nocturne_fragment", human_seq)]:
    scores = multi_n_overlap(llm1_seq, ref_seq, ns=[3, 5, 8])
    print(f"\n  llm/notagen_output_1  <->  {ref_name}")
    for k, v in scores.items():
        bar = "#" * int(v * 40)
        print(f"    {k:15s}: {v:.4f}  {bar}")

# Pairwise matrix
ngram_mat, ng_names = pairwise_ngram_matrix(ALL_CORPUS, n=5, metric="jaccard")
flagged = flag_high_overlap_pairs(ngram_mat, ng_names, threshold=0.08)
print(f"\n  Pairwise Jaccard n=5 matrix shape: {ngram_mat.shape}")
print(f"  Pairs flagged (threshold=0.08):    {len(flagged)}")
for f in flagged[:5]:
    print(f"    rank {f['rank']:2d}: {f['piece_a'][:30]:30s} <-> {f['piece_b'][:30]:30s}  score={f['score']:.4f}")

# -------------------------------------------------------------------------
# 2. LCS ANALYSIS
# -------------------------------------------------------------------------
print("\n\n[2] LCS SIMILARITY (paraphrased copy detection)")
print("-"*50)

pairs = [
    ("llm/notagen_output_1",       "train/bach_invention_A"),
    ("llm/notagen_output_1",       "maestro/chopin_nocturne_fragment"),
    ("train/bach_invention_A",     "train/bach_invention_B"),
    ("llm/notagen_output_2",       "train/bach_invention_A"),
]

for na, nb in pairs:
    sa = ALL_CORPUS[na]; sb = ALL_CORPUS[nb]
    # Strict
    s_strict = lcs_similarity(sa, sb)
    # Pitch-only (ignore duration)
    s_pitch  = lcs_similarity(sa, sb, ignore_duration=True)
    print(f"\n  {na.split('/')[-1]:35s} <-> {nb.split('/')[-1]}")
    print(f"    strict    : {s_strict:.4f}  {'#'*int(s_strict*40)}")
    print(f"    pitch-only: {s_pitch:.4f}  {'#'*int(s_pitch*40)}")

# Sliding-window profile for the most suspicious pair
print("\n  Sliding-window LCS profile (llm_output_1 vs train_A, window=8):")
profile = sliding_window_lcs(llm1_seq, train_a_seq, window=8, step=4)
for pos, score in profile:
    bar = "#" * int(score * 30)
    print(f"    pos={pos:3d}  score={score:.3f}  {bar}")

peak = peak_lcs_region(llm1_seq, train_a_seq, window=8, step=4)
if peak:
    print(f"\n  * Peak region: tokens [{peak[0]}-{peak[1]}]  score={peak[2]:.4f}")

# -------------------------------------------------------------------------
# 3. EMBEDDING / STYLE COLLAPSE ANALYSIS
# -------------------------------------------------------------------------
print("\n\n[3] EMBEDDING DISTANCE (style collapse detection)")
print("-"*50)

# TF-IDF embeddings
embedder = TFIDFMusicEmbedder(ngram_range=(1, 2), max_features=200)
tfidf_mat, emb_names = embedder.fit_transform(ALL_CORPUS)
tfidf_dist = cosine_distance_matrix(tfidf_mat)

print(f"\n  TF-IDF embedding shape: {tfidf_mat.shape}")
print("\n  Pairwise cosine DISTANCE (0=identical, 1=maximally different):")
for i, ni in enumerate(emb_names):
    for j, nj in enumerate(emb_names):
        if j <= i: continue
        d   = tfidf_dist[i, j]
        bar = "." * int(d * 30)
        print(f"    {ni.split('/')[-1]:30s} <-> {nj.split('/')[-1]:30s}  dist={d:.3f}  {bar}")

# Per-group style collapse
print("\n  Style Collapse Scores by group (mean intra-group cosine distance):")
for group_name in ["train_corpus", "llm_output", "human_baseline"]:
    g_names = [n for n, g in FLAT_GROUP_MAP.items() if g == group_name]
    g_idxs  = [emb_names.index(n) for n in g_names]
    if len(g_idxs) < 2:
        print(f"    {group_name:20s}: need >=2 pieces (skipped)")
        continue
    g_emb = tfidf_mat[g_idxs]
    sc    = style_collapse_score(g_emb)
    bar   = "#" * int(sc * 40)
    print(f"    {group_name:20s}: {sc:.4f}  {bar}  ({'collapsed' if sc < 0.3 else 'diverse'})")

# Pitch-class histograms
print("\n  Pitch-Class Histogram distances (tonal fingerprints):")
hist_mat, hist_names = corpus_feature_matrix(ALL_CORPUS)
hist_dist = cosine_distance_matrix(hist_mat)
for i, ni in enumerate(hist_names):
    for j, nj in enumerate(hist_names):
        if j <= i: continue
        d = hist_dist[i, j]
        print(f"    {ni.split('/')[-1]:30s} <-> {nj.split('/')[-1]:30s}  hist_dist={d:.3f}")

# PCA
pca_xy = pca_2d(tfidf_mat)
print(f"\n  PCA 2D coords computed (shape {pca_xy.shape})")
for name, (x, y) in zip(emb_names, pca_xy):
    print(f"    {name.split('/')[-1]:35s}  PC1={x:+.3f}  PC2={y:+.3f}")

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
print("\n" + "="*60)
print("SUMMARY / INTERPRETATION GUIDE")
print("="*60)
print("""
  Metric              High value means ...             Flag threshold
  ------------------  -----------------------------  --------------
  Jaccard n=5         Shared melodic vocabulary      > 0.15
  BLEU-precision n=5  LLM output echoes training     > 0.20
  LCS (strict)        Verbatim shared subsequence    > 0.35
  LCS (pitch-only)    Rhythmic paraphrase            > 0.40
  Cosine dist (TF-IDF) Low = style collapse          < 0.25
  Hist dist            Low = same tonal fingerprint  < 0.10

  Negative result note:
    If no pairs exceed thresholds, this is still publishable --
    it bounds the *degree* of memorisation, which is a meaningful
    finding for a model safety evaluation protocol.
""")
print("Demo complete. Install music21 and point add_corpus() at real")
print("MIDI/ABC directories to run the full experiment.\n")

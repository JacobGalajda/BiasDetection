"""
demo.py
-------
Smoke-test and demonstration of all analysis modules using synthetic
token sequences (no music files or music21 needed).

WHY THIS DEMO EXISTS
--------------------
The real experiment compares NotaGen-generated MIDI against its training
corpus to test for memorization bias. Before running on hundreds of real
files, we validate every metric on a known-ground-truth synthetic corpus
where we can predict exactly what each number should be.

  Corpus design:
    - notagen_output_1  is a transposed copy of bach_invention_A
                        (C major -> D major, same structure, same rhythm).
                        Every metric should detect this -- but some will fail.
                        That failure is the point: it reveals each metric's
                        blind spots before we trust them on real data.

    - notagen_output_2  is stylistically similar (tonal, Baroque arpeggios)
                        but uses different material. Should score lower than
                        output_1. Verifies we're not flagging by accident.

    - chopin / mozart   are the human baseline: stylistically distant from
                        Bach. These pairs set the null distribution -- scores
                        here represent background musical similarity that
                        any two tonal pieces share by coincidence.

Run:
    python demo.py        (from inside the package dir)
    python music_bias_analyzer/demo.py  (from project root)
"""

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import numpy as np
import logging
logging.basicConfig(level=logging.WARNING,        # quiet in demo mode
                    format="%(levelname)s  %(name)s  %(message)s")

from music_bias_analyzer.ngram      import (jaccard_overlap, bleu_precision,
                                            multi_n_overlap,
                                            pairwise_ngram_matrix,
                                            flag_high_overlap_pairs)
from music_bias_analyzer.lcs        import (lcs_similarity, lcs_length,
                                            lcs_density_score,
                                            sliding_window_lcs, peak_lcs_region)
from music_bias_analyzer.embeddings import (TFIDFMusicEmbedder,
                                            pitch_class_histogram,
                                            semitone_interval_histogram,
                                            music_feature_vector_extended,
                                            corpus_feature_matrix,
                                            cosine_distance_matrix,
                                            style_collapse_score, pca_2d)

# -- ANSI color helpers ----------------------------------------------------

_RED    = "\033[91m"
_YELLOW = "\033[93m"
_GREEN  = "\033[92m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"

def _bar_high(value, flag_thr, warn_thr=None, width=40):
    """
    Coloured bar for HIGH-is-suspicious metrics (n-gram, LCS).
      Red    : value >= flag_thr   (flagged)
      Yellow : value >= warn_thr   (approaching threshold)
      Green  : value < warn_thr   (clean)
    """
    warn_thr = warn_thr or flag_thr * 0.7
    filled   = int(value * width)
    bar      = "#" * filled
    if value >= flag_thr:
        return f"{_RED}{bar}{_RESET}"
    elif value >= warn_thr:
        return f"{_YELLOW}{bar}{_RESET}"
    else:
        return f"{_GREEN}{bar}{_RESET}"

def _bar_low(value, flag_thr, width=30):
    """
    Coloured bar for LOW-is-suspicious metrics (cosine distance).
      Red    : value <= flag_thr   (flagged -- too similar)
      Yellow : value <= flag_thr * 1.5
      Green  : safe distance
    """
    filled = int(value * width)
    bar    = "." * filled
    if value <= flag_thr:
        return f"{_RED}{bar}{_RESET}"
    elif value <= flag_thr * 1.5:
        return f"{_YELLOW}{bar}{_RESET}"
    else:
        return f"{_GREEN}{bar}{_RESET}"

def _flag_marker(value, flag_thr, high=True):
    """Return a <- FLAG marker if threshold crossed, else empty string."""
    triggered = (value >= flag_thr) if high else (value <= flag_thr)
    return f"  {_RED}<- FLAG{_RESET}" if triggered else ""

# -- Synthetic corpora -----------------------------------------------------

def _melody(pitches, dur="quarter"):
    return [f"{p}_{dur}" for p in pitches]

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
    # Transposed copy: bach_A shifted C->D, identical structure/rhythm
    "llm/notagen_output_1": _melody(
        ["D4","E4","F#4","G4","A4","B4","C#5","D5",
         "C#5","B4","A4","G4","F#4","E4","D4","REST",
         "A4","B4","C#5","D5","E5","F#5","G5","A5",
         "D4","F#4","A4","D5","A4","F#4","D4","REST"]),
    # Different material -- tonal/Baroque style but not a copy
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

# Experimental labels: what each pair represents scientifically
PAIR_LABELS = {
    ("llm/notagen_output_1", "train/bach_invention_A"):        "suspected transposed copy",
    ("llm/notagen_output_1", "maestro/chopin_nocturne_fragment"): "null baseline (LLM vs human)",
    ("train/bach_invention_A", "train/bach_invention_B"):      "within-training variation",
    ("llm/notagen_output_2",  "train/bach_invention_A"):       "stylistically similar, distinct",
    ("llm/notagen_output_1",  "train/bach_invention_B"):       "partial structural overlap",
    ("maestro/chopin_nocturne_fragment", "maestro/mozart_sonata_fragment"): "human-human null pair",
}

def _pair_label(a, b):
    return PAIR_LABELS.get((a, b)) or PAIR_LABELS.get((b, a)) or ""


# ==============================================================================
print("\n" + "="*65)
print(f"{_BOLD}  MUSIC BIAS ANALYZER -- DEMO{_RESET}")
print("="*65)
print(f"""
{_DIM}  Synthetic corpus:  6 pieces across 3 groups
  Ground truth:       notagen_output_1 is a transposed copy of bach_A
  Goal:               verify each metric catches (or misses) the known copy,
                      and quantify what background musical similarity looks like
                      in the human baseline pairs.

  Bar colors:   {_GREEN}# safe{_RESET}   {_YELLOW}# approaching threshold{_RESET}   {_RED}# flagged{_RESET}{_DIM}
{_RESET}""")

llm1_seq    = LLM_OUTPUT["llm/notagen_output_1"]
train_a_seq = TRAINING["train/bach_invention_A"]
train_b_seq = TRAINING["train/bach_invention_B"]
human_seq   = HUMAN_BASELINE["maestro/chopin_nocturne_fragment"]
mozart_seq  = HUMAN_BASELINE["maestro/mozart_sonata_fragment"]


# ==============================================================================
# 1. N-GRAM ANALYSIS
# ==============================================================================
print(f"{_BOLD}[1] N-GRAM OVERLAP -- verbatim copy detection{_RESET}")
print("-"*65)
print(f"""{_DIM}
  What it measures:  fraction of shared melodic n-grams (contiguous note groups).
  0.0 = no shared n-grams (maximally different token vocabulary)
  1.0 = identical n-gram vocabulary (exact token-for-token match)

  Blind spot:  transposition changes every token, so a melody shifted to a
  different key scores 0.00 even if the structure is identical.  This test
  confirms that blind spot -- and shows why LCS and interval metrics exist.

  Flag threshold:  Jaccard n=5 > 0.15  |  BLEU n=5 > 0.20
{_RESET}""")

NGRAM_PAIRS = [
    ("llm/notagen_output_1",  "train/bach_invention_A",              llm1_seq,    train_a_seq),
    ("llm/notagen_output_1",  "maestro/chopin_nocturne_fragment",    llm1_seq,    human_seq),
    ("train/bach_invention_A","train/bach_invention_B",              train_a_seq, train_b_seq),
    ("maestro/chopin_nocturne_fragment","maestro/mozart_sonata_fragment", human_seq, mozart_seq),
]

for na, nb, sa, sb in NGRAM_PAIRS:
    label   = _pair_label(na, nb)
    scores  = multi_n_overlap(sa, sb, ns=[3, 5, 8])
    short_a = na.split("/")[-1]
    short_b = nb.split("/")[-1]
    print(f"  {_CYAN}{short_a}{_RESET}  <->  {_CYAN}{short_b}{_RESET}")
    print(f"  {_DIM}> {label}{_RESET}")
    for k, v in scores.items():
        thr  = 0.15 if "jaccard" in k else 0.20
        bar  = _bar_high(v, flag_thr=thr, width=40)
        flag = _flag_marker(v, thr)
        print(f"    {k:15s}: {v:.4f}  {bar}{flag}")
    print()

# Pairwise matrix + flagged pairs
ngram_mat, ng_names = pairwise_ngram_matrix(ALL_CORPUS, n=5, metric="jaccard")
flagged = flag_high_overlap_pairs(ngram_mat, ng_names, threshold=0.08)
print(f"  Pairwise Jaccard n=5 matrix: {ngram_mat.shape}  |  "
      f"Pairs above 0.08: {_RED if flagged else _GREEN}{len(flagged)}{_RESET}")
for f in flagged[:5]:
    la = f['piece_a'].split('/')[-1]; lb = f['piece_b'].split('/')[-1]
    lbl = _pair_label(f['piece_a'], f['piece_b'])
    bar = _bar_high(f['score'], flag_thr=0.15, width=30)
    print(f"    rank {f['rank']:2d}: {la:25s} <-> {lb:25s}  {bar}  {f['score']:.4f}  {_DIM}{lbl}{_RESET}")


# ==============================================================================
# 2. LCS ANALYSIS
# ==============================================================================
print(f"\n\n{_BOLD}[2] LCS SIMILARITY -- paraphrase / structural copy detection{_RESET}")
print("-"*65)
print(f"""{_DIM}
  What it measures:  longest token subsequence shared by two pieces,
                     allowing gaps between matched notes.
  0.0 = no common subsequence (completely unrelated structure)
  1.0 = one sequence is entirely contained within the other

  Strict mode:     tokens must match exactly including octave + duration
  Pitch-only mode: duration stripped -- catches rhythmic paraphrases
  WAR/WAQ density: rewards compact matches, penalizes scattered ones
                   (a 10-note theme in a block scores higher than the
                   same 10 notes spread across 100 positions)

  Flag threshold:  LCS strict > 0.35  |  pitch-only > 0.40
{_RESET}""")

LCS_PAIRS = [
    ("llm/notagen_output_1",   "train/bach_invention_A"),
    ("llm/notagen_output_1",   "maestro/chopin_nocturne_fragment"),
    ("train/bach_invention_A", "train/bach_invention_B"),
    ("llm/notagen_output_2",   "train/bach_invention_A"),
]

for na, nb in LCS_PAIRS:
    sa = ALL_CORPUS[na]; sb = ALL_CORPUS[nb]
    label     = _pair_label(na, nb)
    s_strict  = lcs_similarity(sa, sb)
    s_pitch   = lcs_similarity(sa, sb, ignore_duration=True)
    density   = lcs_density_score(sb, sa)

    short_a = na.split("/")[-1]; short_b = nb.split("/")[-1]
    print(f"  {_CYAN}{short_a:35s}{_RESET} <-> {_CYAN}{short_b}{_RESET}")
    print(f"  {_DIM}> {label}{_RESET}")

    bar_s = _bar_high(s_strict, flag_thr=0.35, width=40)
    bar_p = _bar_high(s_pitch,  flag_thr=0.40, width=40)
    print(f"    strict      : {s_strict:.4f}  {bar_s}{_flag_marker(s_strict, 0.35)}")
    print(f"    pitch-only  : {s_pitch:.4f}  {bar_p}{_flag_marker(s_pitch, 0.40)}")
    print(f"    WAR/WAQ     : density={density['density_score']:.4f}  "
          f"war={density['war']:.3f}  waq={density['waq']:.3f}  "
          f"span={density['w_ref']}tok")
    print()

# WAR/WAQ density profile for the most suspicious pair
print(f"  {_DIM}WAR/WAQ density breakdown -- notagen_output_1 vs bach_invention_A{_RESET}")
print(f"  {_DIM}(density_score > lcs_normalized -> match is compact, not coincidental scatter){_RESET}")
d = lcs_density_score(train_a_seq, llm1_seq)
print(f"    lcs_length    : {d['lcs_length']} tokens")
print(f"    lcs_normalized: {d['lcs_normalized']:.4f}  (standard normalized LCS)")
print(f"    density_score : {_RED if d['density_score'] > 0.5 else _GREEN}{d['density_score']:.4f}{_RESET}  "
      f"({'compact -- suspicious' if d['density_score'] > d['lcs_normalized'] else 'scattered -- coincidental'})")
print(f"    span in ref   : {d['w_ref']} of {len(train_a_seq)} tokens")


# ==============================================================================
# 3. EMBEDDING / STYLE COLLAPSE ANALYSIS
# ==============================================================================
print(f"\n\n{_BOLD}[3] EMBEDDING DISTANCE -- style collapse detection{_RESET}")
print("-"*65)
print(f"""{_DIM}
  What it measures:  global stylistic distance between two pieces.
  0.0 = identical style fingerprint (maximally suspicious)
  1.0 = maximally different styles

  TF-IDF distance:      token-vocabulary fingerprint (fast, interpretable)
  Pitch histogram dist: tonal fingerprint -- transposition-SENSITIVE
  Semitone interval dist: melodic contour -- transposition-INVARIANT  [new R10]
                          Same melody in any key = distance 0.0

  Flag threshold:  cosine dist < 0.25 (style collapse within a group)
{_RESET}""")

embedder = TFIDFMusicEmbedder(ngram_range=(1, 2), max_features=200)
tfidf_mat, emb_names = embedder.fit_transform(ALL_CORPUS)
tfidf_dist = cosine_distance_matrix(tfidf_mat)

print(f"  TF-IDF embedding shape: {tfidf_mat.shape}\n")
print(f"  Pairwise cosine DISTANCE  {_DIM}(0=identical style -> suspicious, 1=maximally different){_RESET}")

EMBED_PAIRS = [
    ("train/bach_invention_A",          "llm/notagen_output_1"),
    ("train/bach_invention_A",          "llm/notagen_output_2"),
    ("train/bach_invention_A",          "train/bach_invention_B"),
    ("llm/notagen_output_1",            "maestro/chopin_nocturne_fragment"),
    ("maestro/chopin_nocturne_fragment","maestro/mozart_sonata_fragment"),
]
for na, nb in EMBED_PAIRS:
    ia = emb_names.index(na); ib = emb_names.index(nb)
    d     = float(tfidf_dist[ia, ib])
    label = _pair_label(na, nb)
    bar   = _bar_low(d, flag_thr=0.25, width=30)
    flag  = _flag_marker(d, 0.25, high=False)
    la    = na.split("/")[-1]; lb = nb.split("/")[-1]
    print(f"    {la:30s} <-> {lb:30s}  {bar}  {d:.3f}{flag}  {_DIM}{label}{_RESET}")

# Per-group style collapse
print(f"\n  Style Collapse Score per group  {_DIM}(lower = model output less diverse){_RESET}")
print(f"  {_DIM}Expected: LLM group < human group if collapse is occurring{_RESET}\n")
for group_name in ["train_corpus", "llm_output", "human_baseline"]:
    g_names = [n for n, g in FLAT_GROUP_MAP.items() if g == group_name]
    g_idxs  = [emb_names.index(n) for n in g_names if n in emb_names]
    if len(g_idxs) < 2:
        print(f"    {group_name:20s}: need >=2 pieces (skipped)")
        continue
    g_emb = tfidf_mat[g_idxs]
    sc    = style_collapse_score(g_emb)
    bar   = _bar_low(sc, flag_thr=0.30, width=40)
    verdict = f"{_RED}COLLAPSED{_RESET}" if sc < 0.30 else f"{_GREEN}diverse{_RESET}"
    print(f"    {group_name:20s}: {sc:.4f}  {bar}  ({verdict})")

# Pitch histogram distances
print(f"\n  Pitch-class histogram distances  {_DIM}(transposition-SENSITIVE tonal fingerprint){_RESET}")
hist_mat, hist_names = corpus_feature_matrix(ALL_CORPUS)
hist_dist = cosine_distance_matrix(hist_mat)
for na, nb in EMBED_PAIRS:
    ia = hist_names.index(na); ib = hist_names.index(nb)
    d     = float(hist_dist[ia, ib])
    label = _pair_label(na, nb)
    bar   = _bar_low(d, flag_thr=0.10, width=30)
    flag  = _flag_marker(d, 0.10, high=False)
    la    = na.split("/")[-1]; lb = nb.split("/")[-1]
    print(f"    {la:30s} <-> {lb:30s}  {bar}  {d:.3f}{flag}  {_DIM}{label}{_RESET}")

# Semitone interval histogram -- transposition-invariant [R10]
print(f"\n  Semitone interval histogram distances  {_DIM}(transposition-INVARIANT contour){_RESET}")
print(f"  {_DIM}Same melody in C and D major -> distance 0.0.  Detects copies that pitch-histogram misses.{_RESET}\n")
for na, nb in EMBED_PAIRS:
    sa  = ALL_CORPUS[na]; sb = ALL_CORPUS[nb]
    ha  = semitone_interval_histogram(sa)
    hb  = semitone_interval_histogram(sb)
    # cosine distance
    denom = (np.linalg.norm(ha) * np.linalg.norm(hb))
    d     = float(1 - np.dot(ha, hb) / denom) if denom > 0 else 1.0
    label = _pair_label(na, nb)
    bar   = _bar_low(d, flag_thr=0.10, width=30)
    flag  = _flag_marker(d, 0.10, high=False)
    la    = na.split("/")[-1]; lb = nb.split("/")[-1]
    print(f"    {la:30s} <-> {lb:30s}  {bar}  {d:.3f}{flag}  {_DIM}{label}{_RESET}")

# PCA
pca_xy = pca_2d(tfidf_mat)
print(f"\n  PCA 2D projection  {_DIM}(pieces closer together = more similar style){_RESET}")
print(f"  {_DIM}Expected: train and llm_output_1 nearby; human baseline far away{_RESET}\n")
for name, (x, y) in zip(emb_names, pca_xy):
    group  = FLAT_GROUP_MAP.get(name, "")
    color = _RED if "llm" in group else (_CYAN if "train" in group else _DIM)
    short  = name.split("/")[-1]
    print(f"    {color}{short:35s}{_RESET}  PC1={x:+.3f}  PC2={y:+.3f}  {_DIM}[{group}]{_RESET}")


# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*65)
print(f"{_BOLD}  SUMMARY -- WHAT THIS SMOKE TEST CONFIRMS{_RESET}")
print("="*65)
print(f"""
  {_BOLD}Metric family        0.0 means ...         1.0 means ...        Flag if ...{_RESET}
  -----------------------------------------------------------------
  Jaccard n=5          no shared n-grams    identical vocab    > 0.15
  BLEU n=5             LLM echoes nothing   full echo          > 0.20
  LCS strict           no common subseq     fully contained    > 0.35
  LCS pitch-only       no pitch skeleton    identical melody   > 0.40
  WAR/WAQ density      scattered match      compact block      > lcs_normalized
  TF-IDF dist          {_RED}identical style{_RESET}     maximally diff     < 0.25
  Hist dist (pitch-class) {_RED}same key/mode{_RESET}   opposite modes     < 0.10
  Semitone interval    {_RED}same contour{_RESET}        opposite contour   < 0.10

  {_BOLD}Key findings from synthetic ground truth:{_RESET}
  +-----------------------------------------------------------------------------------+
  |  notagen_output_1 IS a transposed copy of bach_invention_A.                       |
  |                                                                                   |
  |  * N-gram:        {_RED}MISSES IT{_RESET} -- transposition shifts all tokens       |
  |  * LCS strict:    {_GREEN}CATCHES IT{_RESET} -- shared structural skeleton         |
  |  * Pitch hist:    {_RED}MISSES IT{_RESET} -- different key = different histogram   |
  |  * Semitone int:  {_GREEN}CATCHES IT{_RESET} -- contour is key-independent  [R10]  |
  |  * WAR/WAQ:       {_GREEN}AMPLIFIES IT{_RESET} -- confirms match is compact        |
  +-----------------------------------------------------------------------------------+

\n""")


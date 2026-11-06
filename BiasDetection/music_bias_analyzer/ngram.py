"""
ngram.py
--------
N-gram overlap metrics for detecting verbatim (or near-verbatim) copying
between symbolic music token sequences.

Why n-grams for music?
  A melodic "quote" is structurally identical to a textual quote -- a contiguous
  subsequence of tokens that appears in two works.  Jaccard overlap on n-gram
  *sets* catches exact repetition regardless of absolute position; BLEU-style
  precision catches repetition weighted by frequency (covers both copy-paste
  and looped motifs).

Recommended n values
  n=3  : short motif (3-note figure) -- high recall, lower precision
  n=5  : phrase-level -- good balance for melody detection
  n=8  : full phrase / theme detection -- high precision, lower recall
"""

from __future__ import annotations
from collections import Counter
from typing import List, Tuple, Dict
import numpy as np


# ---------------------------------------------------------------------------
# Core n-gram utilities
# ---------------------------------------------------------------------------

def make_ngrams(sequence: List[str], n: int) -> List[Tuple[str, ...]]:
    """Return all contiguous n-grams from a token sequence."""
    if len(sequence) < n:
        return []
    return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]


def ngram_counts(sequence: List[str], n: int) -> Counter:
    return Counter(make_ngrams(sequence, n))


def ngram_set(sequence: List[str], n: int) -> set:
    return set(make_ngrams(sequence, n))


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------

def jaccard_overlap(seq1: List[str], seq2: List[str], n: int = 5) -> float:
    """
    Jaccard similarity of n-gram sets.

    J(A,B) = |A & B| / |A | B|

    Value in [0, 1].  1 = identical n-gram vocabulary, 0 = no shared n-grams.
    Robust to length differences; does NOT penalise repeated motifs.
    Best for: detecting whether two pieces share the same musical material
    regardless of how many times it appears.
    """
    s1 = ngram_set(seq1, n)
    s2 = ngram_set(seq2, n)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def bleu_precision(hypothesis: List[str], reference: List[str],
                   n: int = 5) -> float:
    """
    Modified n-gram precision (BLEU component), clipped to reference counts.

    Measures what fraction of the hypothesis n-grams appear in the reference,
    capped at the reference frequency (prevents gaming by repetition).
    Best for: quantifying how much of an LLM output could have been "copied"
    from a reference corpus piece.
    """
    hyp_counts = ngram_counts(hypothesis, n)
    ref_counts = ngram_counts(reference, n)

    clipped_matches = sum(
        min(count, ref_counts[gram])
        for gram, count in hyp_counts.items()
    )
    total_hyp = sum(hyp_counts.values())
    if total_hyp == 0:
        return 0.0
    return clipped_matches / total_hyp


def multi_n_overlap(seq1: List[str], seq2: List[str],
                    ns: List[int] = (3, 5, 8)) -> Dict[str, float]:
    """
    Compute Jaccard overlap at multiple n values.
    Returns dict keyed by "jaccard_n<N>" and "bleu_n<N>".
    Useful for building a feature vector per pair.
    """
    result: Dict[str, float] = {}
    for n in ns:
        result[f"jaccard_n{n}"] = jaccard_overlap(seq1, seq2, n)
        result[f"bleu_n{n}"]    = bleu_precision(seq1, seq2, n)
    return result


# ---------------------------------------------------------------------------
# Corpus-level pairwise matrix
# ---------------------------------------------------------------------------

def pairwise_ngram_matrix(corpus: Dict[str, List[str]],
                          n: int = 5,
                          metric: str = "jaccard") -> Tuple[np.ndarray, List[str]]:
    """
    Compute an all-pairs n-gram similarity matrix for a corpus.

    Parameters
    ----------
    corpus  : {name: token_list}
    n       : gram size
    metric  : "jaccard" | "bleu"

    Returns
    -------
    (matrix, names)  -- symmetric float32 matrix + ordered name list
    """
    names = list(corpus.keys())
    seqs  = [corpus[k] for k in names]
    m     = len(names)
    mat   = np.zeros((m, m), dtype=np.float32)

    fn = jaccard_overlap if metric == "jaccard" else bleu_precision

    for i in range(m):
        mat[i, i] = 1.0
        for j in range(i + 1, m):
            score      = fn(seqs[i], seqs[j], n)
            mat[i, j]  = score
            mat[j, i]  = score

    return mat, names


# ---------------------------------------------------------------------------
# Flagging: identify suspiciously high-overlap pairs
# ---------------------------------------------------------------------------

def flag_high_overlap_pairs(matrix: np.ndarray, names: List[str],
                             threshold: float = 0.15) -> List[Dict]:
    """
    Return all off-diagonal pairs whose score exceeds `threshold`.

    threshold=0.15 is a conservative starting point for n=5 Jaccard;
    lower it to 0.10 for stricter flagging of LLM outputs vs. training data.

    Returns list of dicts: {piece_a, piece_b, score, rank}
    sorted by score descending.
    """
    flagged = []
    m = len(names)
    for i in range(m):
        for j in range(i + 1, m):
            score = float(matrix[i, j])
            if score >= threshold:
                flagged.append({
                    "piece_a": names[i],
                    "piece_b": names[j],
                    "score":   round(score, 4),
                })

    flagged.sort(key=lambda x: x["score"], reverse=True)
    for rank, entry in enumerate(flagged, 1):
        entry["rank"] = rank

    return flagged


# ---------------------------------------------------------------------------
# LCS pre-filter: candidate pairs worth deeper inspection  [R7]
# ---------------------------------------------------------------------------

def candidate_pairs(corpus: Dict[str, List[str]],
                    min_jaccard: float = 0.05,
                    n: int = 3) -> List[Tuple[str, str]]:
    """
    Return pairs whose Jaccard n-gram overlap exceeds `min_jaccard`.

    Motivation (Lin et al. 2011 -- Tarhio-Ukkonen filtering principle)
    ------------------------------------------------------------------
    Full pairwise LCS is O(m*n) per pair.  For a corpus of C pieces the
    naive approach runs C^2/2 LCS computations.  N-gram Jaccard at n=3
    is O(L) per pair (just set intersection) and serves as a cheap
    first-pass filter: if two pieces share no 3-note n-grams above a
    minimum threshold they cannot have a meaningful LCS either.

    Only pairs that survive this filter need to be passed to
    `pairwise_lcs_matrix()`, dramatically reducing the LCS workload
    for large corpora where most pairs are unrelated.

    Parameters
    ----------
    corpus      : {name: token_list}
    min_jaccard : minimum Jaccard n=`n` score to keep a pair (default 0.05)
    n           : gram size for the filter pass (default 3 -- short motifs)

    Returns
    -------
    List of (name_a, name_b) tuples, sorted by score descending.

    Notes
    -----
    - Setting min_jaccard=0.0 returns all pairs (no filtering).
    - The filter is conservative by design: a false negative here means
      a pair is permanently excluded from LCS analysis.  Keep min_jaccard
      low (0.03-0.07) rather than high.
    - For corpora < 50 pieces the filtering overhead may exceed the saving;
      benchmark before applying to small corpora.
    """
    names = list(corpus.keys())
    seqs  = [corpus[k] for k in names]
    m     = len(names)

    # Pre-compute n-gram sets once per piece
    gram_sets = [ngram_set(seq, n) for seq in seqs]

    candidates: List[Tuple[str, str, float]] = []

    for i in range(m):
        for j in range(i + 1, m):
            s1, s2 = gram_sets[i], gram_sets[j]
            if not s1 or not s2:
                continue
            score = len(s1 & s2) / len(s1 | s2)
            if score >= min_jaccard:
                candidates.append((names[i], names[j], score))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return [(a, b) for a, b, _ in candidates]

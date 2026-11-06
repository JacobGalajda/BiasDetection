"""
lcs_fast.py
-----------
Drop-in replacement for the slow parts of lcs.py.

Three key optimisations:

1. Numpy anti-diagonal LCS  (replaces pure-Python double for-loop)
   ~30-60x faster at 2000 tokens.

2. Best-window LCS for asymmetric-length pairs  (Option B)
   Slides the shorter sequence across the longer one, taking the max
   LCS score over all windows. Normalises by len(query) -- the right
   formulation for memorisation detection.

3. Multiprocessing pairwise matrix
   Each (i,j) pair is independent -- maps across all CPU cores via Pool.

Usage -- monkey-patch or import directly from pipeline_fast.py.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token normalisation
# ---------------------------------------------------------------------------

def _normalise(tokens: List[str],
               ignore_duration: bool,
               ignore_octave: bool) -> List[str]:
    out = []
    for t in tokens:
        if ignore_duration:
            t = t.split("_")[0]
        if ignore_octave:
            t = t.rstrip("0123456789")
        out.append(t)
    return out


def _encode(seq1: List[str],
            seq2: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Map string tokens to integer IDs for fast numpy comparison."""
    vocab: Dict[str, int] = {}
    idx = 0
    a, b = [], []
    for t in seq1:
        if t not in vocab:
            vocab[t] = idx; idx += 1
        a.append(vocab[t])
    for t in seq2:
        if t not in vocab:
            vocab[t] = idx; idx += 1
        b.append(vocab[t])
    return np.array(a, dtype=np.int32), np.array(b, dtype=np.int32)


# ---------------------------------------------------------------------------
# Core: numpy anti-diagonal LCS
# ---------------------------------------------------------------------------

def lcs_length_numpy(seq1: List[str], seq2: List[str]) -> int:
    """
    LCS length via numpy anti-diagonal vectorisation.
    ~30-60x faster than pure Python at 2000 tokens.
    """
    if not seq1 or not seq2:
        return 0

    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1

    m, n = len(seq1), len(seq2)
    a, b = _encode(seq1, seq2)
    dp   = np.zeros((m + 1, n + 1), dtype=np.int32)

    for k in range(1, m + n + 1):
        i0 = max(1, k - n)
        i1 = min(m, k - 1)
        if i0 > i1:
            continue
        iv       = np.arange(i0, i1 + 1, dtype=np.int32)
        jv       = k - iv
        match    = a[iv - 1] == b[jv - 1]
        from_diag = dp[iv - 1, jv - 1] + 1
        from_side = np.maximum(dp[iv - 1, jv], dp[iv, jv - 1])
        dp[iv, jv] = np.where(match, from_diag, from_side)

    return int(dp[m, n])


# ---------------------------------------------------------------------------
# Similarity functions
# ---------------------------------------------------------------------------

def lcs_similarity_numpy(seq1: List[str],
                         seq2: List[str],
                         ignore_duration: bool = False,
                         ignore_octave:   bool = False) -> float:
    """Global LCS normalised by max length. Drop-in for lcs.lcs_similarity."""
    if not seq1 or not seq2:
        return 0.0
    s1    = _normalise(seq1, ignore_duration, ignore_octave)
    s2    = _normalise(seq2, ignore_duration, ignore_octave)
    denom = max(len(s1), len(s2))
    return 0.0 if denom == 0 else lcs_length_numpy(s1, s2) / denom


def lcs_best_window(seq1: List[str],
                    seq2: List[str],
                    ignore_duration: bool = False,
                    ignore_octave:   bool = False,
                    step:            int  = None) -> float:
    """
    Best-window LCS for asymmetric-length pairs  (Option B).

    Treats the shorter sequence as the query and slides it across the
    longer reference, returning the maximum LCS / len(query) over all
    windows. This asks: "what fraction of the shorter piece appears in
    the best-matching passage of the longer piece?"

    This is the correct normalisation for memorisation detection:
      - A 300-token training fragment fully reproduced in a 1400-token
        LLM output scores 1.0 instead of 0.21.
      - Human self-copy rates remain the null baseline.

    step : window stride (default = query_len // 4, 75% overlap).
           Smaller step = more thorough, slower.
    """
    if not seq1 or not seq2:
        return 0.0

    s1 = _normalise(seq1, ignore_duration, ignore_octave)
    s2 = _normalise(seq2, ignore_duration, ignore_octave)

    # Always query with the shorter sequence
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    qlen = len(s1)
    rlen = len(s2)

    if qlen == 0:
        return 0.0

    # Sequences close in length -- full LCS is fine, no windowing needed
    if rlen <= int(qlen * 1.5):
        return lcs_length_numpy(s1, s2) / qlen

    _step = step if step is not None else max(1, qlen // 4)
    best  = 0.0

    positions = list(range(0, rlen - qlen + 1, _step))
    # Always include the final window
    last = rlen - qlen
    if positions[-1] != last:
        positions.append(last)

    for start in positions:
        window = s2[start: start + qlen]
        score  = lcs_length_numpy(s1, window) / qlen
        if score > best:
            best = score
        if best >= 1.0:
            break

    return best


# ---------------------------------------------------------------------------
# Multiprocessing pairwise matrix
# ---------------------------------------------------------------------------

_WORKER_SEQS:     List[List[str]] = []
_WORKER_IGN_DUR:  bool = False
_WORKER_IGN_OCT:  bool = False
_WORKER_USE_WIN:  bool = False


def _worker_init(seqs: List[List[str]],
                 ignore_duration: bool,
                 ignore_octave:   bool,
                 use_window:      bool) -> None:
    global _WORKER_SEQS, _WORKER_IGN_DUR, _WORKER_IGN_OCT, _WORKER_USE_WIN
    _WORKER_SEQS    = seqs
    _WORKER_IGN_DUR = ignore_duration
    _WORKER_IGN_OCT = ignore_octave
    _WORKER_USE_WIN = use_window


def _compute_pair(args: Tuple[int, int]) -> Tuple[int, int, float]:
    i, j   = args
    s1, s2 = _WORKER_SEQS[i], _WORKER_SEQS[j]
    if _WORKER_USE_WIN:
        score = lcs_best_window(s1, s2, _WORKER_IGN_DUR, _WORKER_IGN_OCT)
    else:
        score = lcs_similarity_numpy(s1, s2, _WORKER_IGN_DUR, _WORKER_IGN_OCT)
    return i, j, score


def pairwise_lcs_matrix_parallel(
        corpus:          Dict[str, List[str]],
        ignore_duration: bool          = False,
        ignore_octave:   bool          = False,
        max_seq_len:     int           = 3000,
        pairs_mask:      Optional[set] = None,
        n_workers:       int           = 0,
        chunk_size:      int           = 64,
        use_window:      bool          = True,   # Option B on by default
) -> Tuple[np.ndarray, List[str]]:
    """
    All-pairs LCS matrix using multiprocessing + numpy LCS.

    use_window : True  -> best-window LCS (recommended for cross-group pairs
                         with mismatched lengths, e.g. LLM x Train)
                 False -> global LCS (faster, use for intra-group same-length pairs)
    """
    names = list(corpus.keys())
    seqs  = [
        _normalise(corpus[k][:max_seq_len], ignore_duration, ignore_octave)
        for k in names
    ]

    m   = len(names)
    mat = np.zeros((m, m), dtype=np.float32)
    np.fill_diagonal(mat, 1.0)

    all_pairs = [
        (i, j)
        for i in range(m)
        for j in range(i + 1, m)
        if pairs_mask is None or (i, j) in pairs_mask
    ]
    total = len(all_pairs)
    if total == 0:
        return mat, names

    n_workers = n_workers or mp.cpu_count()
    log.info("LCS pairwise: %d pairs, %d workers, window=%s, max_seq_len=%d",
             total, n_workers, use_window, max_seq_len)

    t0        = time.time()
    completed = 0
    log_every = max(1, total // 20)

    with mp.Pool(
        processes   = n_workers,
        initializer = _worker_init,
        initargs    = (seqs, ignore_duration, ignore_octave, use_window),
    ) as pool:
        for i, j, score in pool.imap_unordered(
                _compute_pair, all_pairs, chunksize=chunk_size):
            mat[i, j] = score
            mat[j, i] = score
            completed += 1
            if completed % log_every == 0:
                elapsed = time.time() - t0
                rate    = completed / max(elapsed, 0.001)
                eta     = (total - completed) / max(rate, 0.001)
                log.info("  LCS %d/%d (%.0f%%)  %.1f pairs/s  ETA %.0fs",
                         completed, total,
                         100 * completed / total, rate, eta)

    log.info("LCS pairwise done in %.1fs", time.time() - t0)
    return mat, names

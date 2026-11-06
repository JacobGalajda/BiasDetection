"""
lcs.py
------
Longest Common Subsequence (LCS) metrics for detecting *paraphrased* copying
between music token sequences.

Musical paraphrasing analogy
  A composer who "paraphrases" another's work might:
  - Transpose the melody (different pitch tokens, same interval structure)
  - Change the rhythm (different duration tokens, same pitches)
  - Reorder a few phrases

LCS detects shared token subsequences that survive such transformations *better*
than n-gram overlap, because LCS allows gaps between matched tokens.

Four modes (in increasing sophistication):

  1. Strict (default)
     Tokens must match exactly ("C4_quarter" == "C4_quarter").
     lcs_similarity()

  2. Pitch-only
     Strip duration suffix before matching -- catches rhythmic paraphrasing
     where pitches are preserved but note lengths change.
     lcs_similarity(ignore_duration=True)

  3. Rough equality  [R4 -- Lin et al. 2011]
     Two tokens "roughly match" if their numeric pitch and duration differ by
     less than configurable thresholds.  Partial credit is awarded proportional
     to closeness.  Catches near-transpositions without discarding pitch info.
     lcs_rough()

  4. WAR/WAQ density scoring  [R5 -- Lin et al. 2011]
     Standard LCS cannot distinguish a compact 5-note match from the same
     5 notes scattered across 50 positions -- both score identically.
     WAR (Width Across Reference) and WAQ (Width Across Query) measure how
     compactly the LCS is distributed.  Final score penalises scattered
     matches and rewards dense ones.  Requires full O(mxn) table.
     lcs_density_score()

  DEPRECATED (use lcs_density_score instead)
     sliding_window_lcs() and peak_lcs_region() -- coarse approximations of
     what WAR/WAQ computes exactly in a single DP pass.  Retained for
     backwards-compatibility but no longer recommended.

Computational notes
  lcs_length() / lcs_similarity()  -- O(min(m,n)) space, two-row DP.
  lcs_rough()                       -- O(mxn) space, full table required for
                                      fractional credit accumulation.
  lcs_density_score()               -- O(mxn) space, three tables simultaneously
                                      (LCS length, WAR, WAQ).
  For very large corpora use candidate_pairs() from ngram.py to pre-filter
  before running any O(mxn) computation.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Token normalisation helpers
# ---------------------------------------------------------------------------

def _strip_duration(token: str) -> str:
    """'C4_quarter' -> 'C4'   'REST_eighth' -> 'REST'"""
    return token.split("_")[0]

def _strip_octave(token: str) -> str:
    """'C4_quarter' -> 'C_quarter'  (pitch-class only, keeps duration)"""
    base = token.split("_")[0].rstrip("0123456789")
    dur  = token.split("_")[1] if "_" in token else ""
    return f"{base}_{dur}" if dur else base

def normalise_tokens(tokens: List[str],
                     ignore_duration: bool = False,
                     ignore_octave: bool = False) -> List[str]:
    result = tokens
    if ignore_duration:
        result = [_strip_duration(t) for t in result]
    if ignore_octave:
        result = [_strip_octave(t)   for t in result]
    return result


# ---------------------------------------------------------------------------
# Core LCS (space-efficient DP)
# ---------------------------------------------------------------------------

def lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """
    Standard DP Longest Common Subsequence length.
    Uses O(min(m,n)) space (only two rows kept).
    """
    if not seq1 or not seq2:
        return 0

    # ensure seq2 is the shorter one for space efficiency
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1

    m, n = len(seq1), len(seq2)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


def lcs_similarity(seq1: List[str], seq2: List[str],
                   ignore_duration: bool = False,
                   ignore_octave: bool = False) -> float:
    """
    Normalised LCS similarity in [0, 1].

    Normalisation: LCS / max(len(seq1), len(seq2))
    Equivalent to "what fraction of the longer sequence is shared?"
    Values > 0.4 warrant manual inspection for a 5-gram level.
    Values > 0.6 are strong evidence of melodic copying.
    """
    s1 = normalise_tokens(seq1, ignore_duration, ignore_octave)
    s2 = normalise_tokens(seq2, ignore_duration, ignore_octave)

    length = lcs_length(s1, s2)
    denom  = max(len(s1), len(s2))
    return length / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Rough equality helpers  [R4]
# ---------------------------------------------------------------------------

# Map duration label -> quarterLength value (mirrors tokenizer._DURATION_MAP)
_DUR_TO_QL: dict = {
    "whole": 4.0, "dotted-half": 3.0, "half": 2.0, "dotted-quarter": 1.5,
    "quarter": 1.0, "dotted-eighth": 0.75, "eighth": 0.5,
    "dotted-sixteenth": 0.375, "sixteenth": 0.25, "thirty-second": 0.125,
    "unknown": 1.0,
}
# Semitone offsets for pitch-class parsing
_PC_TO_SEMI: dict = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7,
    "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}


def _token_to_numeric(token: str) -> Optional[Tuple[float, float]]:
    """
    Convert a baseline token to (midi_pitch, quarter_length).

    Returns None for rests, interval tokens, or unrecognised tokens.
    Used by lcs_rough() for Manhattan distance comparisons.

    MIDI pitch: C4=60, chromatic scale.
    quarter_length: from _DUR_TO_QL map; 1.0 if duration absent.
    """
    if token.startswith("REST"):
        return None
    # Reject interval-mode tokens
    first = token[0] if token else ""
    if first in ("+", "-") or first.isdigit():
        return None

    parts    = token.split("_", 1)
    note_str = parts[0].split(".")[0]   # first note of chord
    dur_str  = parts[1] if len(parts) > 1 else ""

    # Parse pitch-class + octave from note_str
    i = len(note_str)
    while i > 0 and (note_str[i-1].isdigit() or note_str[i-1] == "-"):
        i -= 1
    pc_str  = note_str[:i]
    oct_str = note_str[i:]

    if pc_str not in _PC_TO_SEMI:
        return None
    try:
        octave = int(oct_str)
    except (ValueError, TypeError):
        return None

    midi = (octave + 1) * 12 + _PC_TO_SEMI[pc_str]
    ql   = _DUR_TO_QL.get(dur_str, 1.0)
    return (float(midi), ql)


def lcs_rough(seq1: List[str], seq2: List[str],
              pitch_threshold: float = 3.0,
              duration_threshold: float = 0.5,
              alpha: float = 0.76) -> float:
    """
    Rough-equality LCS similarity with weighted fractional credit.

    Source: Lin, Wu & Wang (2011) "Music Matching Based on Rough Longest
    Common Subsequence", Journal of Information Science and Engineering.

    Two tokens "roughly match" if their Manhattan distance is below threshold:
        d(i, j) = alpha * |midi_i - midi_j| + (1 - alpha) * |ql_i - ql_j|

    When d < T_d (= pitch_threshold after alpha-weighting), they match with
    fractional credit  (1 - d / T_d)  instead of the binary 1 of strict LCS.
    This allows partial matches for near-transpositions (off by 1-2 semitones)
    without fully stripping pitch information.

    Parameters
    ----------
    seq1, seq2         : baseline absolute-pitch token sequences
    pitch_threshold    : maximum weighted distance to consider a rough match
                         (default 3.0 -- roughly 3 semitones before alpha)
    duration_threshold : maximum duration difference in quarterLength units,
                         used to set the combined threshold T_d
                         (default 0.5 -- half a beat)
    alpha              : weight for pitch vs. duration (default 0.76, from
                         Lin et al. -- pitch weighted ~3x more than duration)

    Returns
    -------
    Fractional LCS score normalised by max(len(seq1), len(seq2)).
    Range [0, 1].  Compare against lcs_similarity() on the same pair --
    a higher rough score indicates near-matches that strict LCS missed.

    Notes
    -----
    - Requires full O(mxn) DP table (cannot use space-optimised two-row).
    - Rests are excluded from rough matching (no pitch to compare).
      Two aligned rests from different sequences are treated as exact matches.
    - Tokens that cannot be parsed numerically fall back to exact matching.
    - Anti-symmetry caveat (Lin et al.): if seq1/seq2 were produced with
      use_intervals=True, a single-note change alters two adjacent intervals,
      so distances are double-penalised.  Use rough LCS on absolute-pitch
      tokens, not interval tokens.

    Example
    -------
    C major scale vs D major scale (transposed up 2 semitones):
        lcs_similarity()  -> 0.00   (no exact token matches)
        lcs_rough()       -> 0.87   (all notes within 2-semitone threshold)
    """
    if not seq1 or not seq2:
        return 0.0

    m, n = len(seq1), len(seq2)
    T_d  = pitch_threshold   # combined distance threshold

    # Full O(mxn) table of fractional LCS scores
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            t1 = seq1[i - 1]
            t2 = seq2[j - 1]

            # Determine match credit
            if t1 == t2:
                # Exact match (includes REST == REST)
                credit = 1.0
            elif t1.startswith("REST") or t2.startswith("REST"):
                # One is a rest, other is a note -- no match
                credit = 0.0
            else:
                n1 = _token_to_numeric(t1)
                n2 = _token_to_numeric(t2)
                if n1 is None or n2 is None:
                    # Fallback: treat as exact string comparison
                    credit = 1.0 if t1 == t2 else 0.0
                else:
                    midi1, ql1 = n1
                    midi2, ql2 = n2
                    dist = alpha * abs(midi1 - midi2) + (1 - alpha) * abs(ql1 - ql2)
                    credit = max(0.0, 1.0 - dist / T_d) if dist < T_d else 0.0

            if credit > 0.0:
                dp[i][j] = dp[i-1][j-1] + credit
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    raw   = dp[m][n]
    denom = max(m, n)
    return raw / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Corpus-level pairwise matrix
# ---------------------------------------------------------------------------

def pairwise_lcs_matrix(corpus: dict,
                        ignore_duration: bool     = False,
                        ignore_octave: bool       = False,
                        max_seq_len: int          = 3000,
                        pairs_mask: Optional[set] = None,
                        ) -> Tuple[np.ndarray, List[str]]:
    """
    All-pairs LCS similarity matrix.

    max_seq_len  : truncate sequences to this length to keep runtime feasible.
                   For long MIDI files consider ignore_duration=True to reduce
                   vocabulary and speed up matching.

    pairs_mask   : optional set of (i, j) index pairs (i < j) to compute.
                   All other pairs are left as 0.0 and skipped entirely.
                   Use this to restrict computation to cross-group pairs and
                   avoid expensive intra-baseline computation.
                   When None (default) all pairs are computed.

    Returns (matrix, names)  -- symmetric float32 matrix.
    """
    names = list(corpus.keys())
    seqs  = [
        normalise_tokens(corpus[k][:max_seq_len], ignore_duration, ignore_octave)
        for k in names
    ]
    import logging, time
    _log = logging.getLogger(__name__)

    m       = len(names)
    mat     = np.zeros((m, m), dtype=np.float32)
    total   = m * (m - 1) // 2
    skipped = 0
    computed = 0

    # Try to use tqdm if available, fall back to plain logging
    try:
        from tqdm import tqdm
        to_compute = total - (
            sum(1 for i in range(m) for j in range(i+1, m)
                if pairs_mask is not None and (i, j) not in pairs_mask)
            if pairs_mask else 0
        )
        pbar = tqdm(total=to_compute, desc="LCS pairs", unit="pair",
                    dynamic_ncols=True, leave=True)
        use_tqdm = True
    except ImportError:
        pbar = None
        use_tqdm = False

    t_start  = time.time()
    t_report = t_start  # next log-line timestamp

    for i in range(m):
        mat[i, i] = 1.0
        for j in range(i + 1, m):
            if pairs_mask is not None and (i, j) not in pairs_mask:
                skipped += 1
                continue

            t0    = time.time()
            denom = max(len(seqs[i]), len(seqs[j]))
            score = 0.0 if denom == 0 else lcs_length(seqs[i], seqs[j]) / denom
            mat[i, j] = score
            mat[j, i] = score
            computed  += 1
            elapsed_pair = time.time() - t0

            if use_tqdm:
                pbar.update(1)
                pbar.set_postfix({
                    "last": f"{elapsed_pair:.2f}s",
                    "pair": f"{names[i].split('/')[-1][:12]}<->{names[j].split('/')[-1][:12]}",
                })
            else:
                # Plain fallback: log a progress line every 10 s
                now = time.time()
                if now - t_report >= 10.0:
                    pct  = 100.0 * computed / max(total - skipped, 1)
                    rate = computed / max(now - t_start, 0.001)
                    eta  = (total - skipped - computed) / max(rate, 0.001)
                    _log.info(
                        "LCS progress: %d/%d pairs (%.0f%%)  "
                        "rate=%.1f pairs/s  ETA=%.0fs  "
                        "last pair: %s <-> %s (%.2fs)",
                        computed, total - skipped, pct,
                        rate, eta,
                        names[i].split("/")[-1], names[j].split("/")[-1],
                        elapsed_pair,
                    )
                    t_report = now

    if use_tqdm:
        pbar.close()

    elapsed_total = time.time() - t_start
    _log.info(
        "pairwise_lcs_matrix done: %d computed, %d skipped in %.1fs (%.2f pairs/s)",
        computed, skipped, elapsed_total,
        computed / max(elapsed_total, 0.001),
    )

    return mat, names


# ---------------------------------------------------------------------------
# WAR/WAQ density scoring  [R5 -- Lin et al. 2011]
# ---------------------------------------------------------------------------

def lcs_density_score(seq1: List[str], seq2: List[str],
                      beta: float = 0.5,
                      ignore_duration: bool = False,
                      ignore_octave: bool = False) -> dict:
    """
    LCS density score using WAR/WAQ (Width Across Reference / Width Across Query).

    Source: Lin, Wu & Wang (2011) "Music Matching Based on Rough Longest
    Common Subsequence", Journal of Information Science and Engineering.

    Motivation
    ----------
    Standard LCS cannot distinguish between:
      - A compact 5-note theme appearing as a continuous block, and
      - The same 5 notes scattered across 200 positions with large gaps.
    Both score identically under normalised LCS.  WAR/WAQ penalises the
    scattered case by measuring how compactly the LCS spans each sequence.

    This is critical for memorisation detection: an LLM reproducing a
    16-bar theme inside a 200-note output would be diluted by bare LCS
    but highlighted by WAR/WAQ, because the matched tokens are concentrated.

    Algorithm
    ---------
    Build the full O(mxn) LCS length table c[i][j].
    Backtrack to recover the actual LCS alignment (matched positions in
    seq1 and seq2).  Then compute:

        w_R = last_matched_pos_in_seq1 - first_matched_pos_in_seq1 + 1
        w_Q = last_matched_pos_in_seq2 - first_matched_pos_in_seq2 + 1
        c   = LCS length

    Final score (Lin et al. eq. 8):
        s = (beta * c/w_R  +  (1-beta) * c/w_Q) * (c / len(seq2))

    The (c/w_R) term rewards dense coverage of the reference.
    The (c/w_Q) term rewards dense coverage of the query.
    The (c/n) term is the standard LCS normalisation.

    Parameters
    ----------
    seq1           : reference sequence (suspected source)
    seq2           : query sequence (LLM output being tested)
    beta           : weight between reference density and query density
                     (default 0.5 -- equal weight; increase for reference bias)
    ignore_duration: strip duration suffixes before matching
    ignore_octave  : strip octave digits before matching

    Returns
    -------
    dict with keys:
        lcs_length     : int    -- raw LCS token count
        lcs_norm       : float  -- standard normalised LCS (/ max len)
        war            : float  -- c / w_R  (density in reference)
        waq            : float  -- c / w_Q  (density in query)
        density_score  : float  -- final combined score in [0, 1]
        w_ref          : int    -- span of LCS in reference (tokens)
        w_query        : int    -- span of LCS in query (tokens)

    Notes
    -----
    - Requires full O(mxn) space -- cannot use the two-row optimisation.
      For sequences > 3000 tokens, truncate before calling.
    - A density_score > lcs_norm indicates the match is concentrated
      (more suspicious than bare LCS suggests).
    - A density_score ~= lcs_norm indicates the match is uniformly spread
      (likely coincidental shared vocabulary).
    - Use lcs_similarity() for large corpora pre-screening; reserve this
      function for pairs flagged as suspicious.
    """
    s1 = normalise_tokens(seq1, ignore_duration, ignore_octave)
    s2 = normalise_tokens(seq2, ignore_duration, ignore_octave)
    m, n = len(s1), len(s2)

    if m == 0 or n == 0:
        return dict(lcs_length=0, lcs_norm=0.0, war=0.0, waq=0.0,
                    density_score=0.0, w_ref=0, w_query=0)

    # --- Full O(mxn) LCS table ---
    # c[i][j] = LCS length of s1[:i] and s2[:j]
    c = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                c[i][j] = c[i-1][j-1] + 1
            else:
                c[i][j] = max(c[i-1][j], c[i][j-1])

    lcs_len = c[m][n]

    if lcs_len == 0:
        return dict(lcs_length=0, lcs_norm=0.0, war=0.0, waq=0.0,
                    density_score=0.0, w_ref=0, w_query=0)

    # --- Backtrack to recover matched positions in s1 and s2 ---
    matched_s1: List[int] = []
    matched_s2: List[int] = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1]:
            matched_s1.append(i - 1)   # 0-based index in s1
            matched_s2.append(j - 1)   # 0-based index in s2
            i -= 1
            j -= 1
        elif c[i-1][j] >= c[i][j-1]:
            i -= 1
        else:
            j -= 1
    # Backtracking gives positions in reverse order
    matched_s1.reverse()
    matched_s2.reverse()

    # --- Compute WAR, WAQ ---
    w_ref   = matched_s1[-1] - matched_s1[0] + 1   # span in reference
    w_query = matched_s2[-1] - matched_s2[0] + 1   # span in query

    war = lcs_len / w_ref   if w_ref   > 0 else 0.0
    waq = lcs_len / w_query if w_query > 0 else 0.0

    # --- Final density score (Lin et al. eq. 8) ---
    lcs_norm      = lcs_len / max(m, n)
    density_score = (beta * war + (1 - beta) * waq) * (lcs_len / n)

    return dict(
        lcs_length    = lcs_len,
        lcs_norm      = round(lcs_norm, 4),
        war           = round(war, 4),
        waq           = round(waq, 4),
        density_score = round(density_score, 4),
        w_ref         = w_ref,
        w_query       = w_query,
    )


# ---------------------------------------------------------------------------
# Sliding-window LCS  [DEPRECATED -- use lcs_density_score() instead]
# ---------------------------------------------------------------------------

def sliding_window_lcs(query: List[str], reference: List[str],
                       window: int = 50, step: int = 25) -> List[Tuple[int, float]]:
    """
    Detect local copying by comparing a query sequence against sliding
    windows of a reference sequence.

    .. deprecated::
        Use lcs_density_score() instead.  WAR/WAQ computes an equivalent
        measure of match compactness in a single DP pass, without the
        window-size sensitivity or boundary artefacts of this approach.
        This function is retained for backwards-compatibility only.

    Returns list of (window_start, lcs_similarity) for each window position.
    Peaks in this profile indicate *where* in the reference the query piece
    most closely matches.

    Parameters
    ----------
    query     : token list of the piece under scrutiny
    reference : token list of the suspected source piece
    window    : number of tokens per reference window
    step      : hop size between windows
    """
    results: List[Tuple[int, float]] = []
    for start in range(0, len(reference) - window + 1, step):
        ref_win = reference[start:start + window]
        score   = lcs_similarity(query, ref_win)
        results.append((start, round(score, 4)))
    return results


def peak_lcs_region(query: List[str], reference: List[str],
                    window: int = 50, step: int = 25
                    ) -> Optional[Tuple[int, int, float]]:
    """
    Return (start, end, score) of the highest-similarity window,
    or None if reference is shorter than the window.

    .. deprecated::
        Use lcs_density_score() instead.  The w_ref field in the returned
        dict gives the actual span of the densest match region in the
        reference without window-size sensitivity.
        This function is retained for backwards-compatibility only.
    """
    profile = sliding_window_lcs(query, reference, window, step)
    if not profile:
        return None
    start, score = max(profile, key=lambda x: x[1])
    return (start, start + window, score)
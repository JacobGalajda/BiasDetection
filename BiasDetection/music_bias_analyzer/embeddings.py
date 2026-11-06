"""
embeddings.py
-------------
Embedding-based distance metrics for detecting *style collapse* --
the tendency of a generative model to produce outputs that cluster
tightly around a narrow region of musical style-space, regardless of
prompt diversity.

Three complementary embedding strategies are provided:

1. TF-IDF token embeddings  (sklearn)
   Fast, interpretable, captures token-frequency fingerprints.
   Good for spotting that a model always uses the same set of
   pitch/duration tokens even when asked for different styles.

2. Pitch-class histogram  (12-dim, hand-crafted)
   Pitch-class occurrence frequencies (C, C#, D ... B).
   Transposition-SENSITIVE -- different keys produce different vectors.
   Directly comparable to musicology literature on tonal fingerprints.
   Best for: identifying the key/mode signature of a piece.

3. Semitone interval histogram  (25-dim, hand-crafted)  [R10]
   True semitone intervals between consecutive notes (-12 ... +12),
   computed from full MIDI pitch (preserving octave information).
   Transposition-INVARIANT -- the same melody in any key produces the
   same vector.  Complements the pitch-class histogram.

   Contrast with interval_histogram() (retained for backwards-
   compatibility), which uses pitch-CLASS differences (mod 12) and
   loses octave information: a perfect fifth up (C4->G4, +7) and a
   minor seventh up (C4->Bb4, +10) are distinct, but C4->G5 (+19) and
   C4->G4 (+7) would be conflated at mod-12.  The semitone version
   avoids this by operating on full MIDI pitch numbers.

   Source: Malandrino et al. (2022) PINL transposition-invariance
   principle applied to the embedding domain.

Feature vectors
   music_feature_vector()           -> 37-dim  (pitch-class 12d + interval_pc 25d)
                                       [unchanged -- backwards-compatible]
   music_feature_vector_extended()  -> 62-dim  (pitch-class 12d + interval_pc 25d
                                               + semitone_interval 25d)
                                       [new -- recommended for new analyses]
Distances returned are cosine distances (1 - cosine_similarity)
so that 0 = identical style, 1 = maximally different.
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# 1. TF-IDF token embeddings
# ---------------------------------------------------------------------------

class TFIDFMusicEmbedder:
    """
    Treat each piece's token sequence as a "document" and embed via TF-IDF.

    The vectoriser is fit on the whole corpus so IDF weights are shared --
    rare musical events (e.g. a tritone leap in sixteenths) get higher weight
    than ubiquitous ones (e.g. C4_quarter).
    """

    def __init__(self, ngram_range: Tuple[int, int] = (1, 3),
                 max_features: int = 5000):
        self.ngram_range  = ngram_range
        self.max_features = max_features
        self._vectorizer: TfidfVectorizer | None = None

    def fit_transform(self, corpus: Dict[str, List[str]]) -> Tuple[np.ndarray, List[str]]:
        """
        Fit on corpus and return (embedding_matrix, ordered_names).
        embedding_matrix shape: (n_pieces, n_features)
        """
        names = list(corpus.keys())
        docs  = [" ".join(corpus[k]) for k in names]   # tokens -> string

        self._vectorizer = TfidfVectorizer(
            ngram_range  = self.ngram_range,
            max_features = self.max_features,
            sublinear_tf = True,       # log(1+tf) -- dampens repeated motifs
        )
        matrix = self._vectorizer.fit_transform(docs).toarray()
        return matrix.astype(np.float32), names

    def transform(self, sequences: List[List[str]]) -> np.ndarray:
        """Embed new sequences using already-fit vectoriser (inference)."""
        if self._vectorizer is None:
            raise RuntimeError("Call fit_transform first.")
        docs = [" ".join(seq) for seq in sequences]
        return self._vectorizer.transform(docs).toarray().astype(np.float32)

    @property
    def feature_names(self) -> List[str]:
        if self._vectorizer is None:
            return []
        return list(self._vectorizer.get_feature_names_out())


# ---------------------------------------------------------------------------
# 2. Pitch-class histogram embedding
# ---------------------------------------------------------------------------

_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F",
                  "F#", "G", "G#", "A", "A#", "B"]
_PC_INDEX      = {pc: i for i, pc in enumerate(_PITCH_CLASSES)}
# Also map enharmonic spellings
_ENHARMONIC    = {"Db": 1, "Eb": 3, "Fb": 4, "Gb": 6, "Ab": 8, "Bb": 10, "Cb": 11}


def _token_to_pitch_class(token: str) -> int | None:
    """
    Extract the pitch-class index (0-11) from a token like 'C#4_quarter'.
    Returns None for rests or unrecognized tokens.
    """
    if token.startswith("REST"):
        return None
    # Strip duration suffix
    name = token.split("_")[0]
    # Strip chord (take first note of chord)
    name = name.split(".")[0]
    # Strip octave digits
    pc_str = name.rstrip("0123456789")
    if pc_str in _PC_INDEX:
        return _PC_INDEX[pc_str]
    if pc_str in _ENHARMONIC:
        return _ENHARMONIC[pc_str]
    return None


# Semitone offsets within an octave for each pitch class
_PC_SEMITONE = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
                "E": 4, "Fb": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7,
                "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11}

def _token_to_midi(token: str) -> int | None:
    """
    Extract full MIDI pitch number from a baseline token like 'C#4_quarter'.

    Preserves octave information -- C4 -> 60, G5 -> 79, etc.
    Returns None for rests, interval tokens (+N / -N), or unrecognized tokens.

    Only operates on baseline (absolute pitch) tokens.  Interval tokens
    produced by tokenizer.use_intervals=True start with '+'/'-'/digits and
    are intentionally rejected here; use semitone_interval_histogram() on
    those sequences directly.
    """
    if token.startswith("REST"):
        return None
    # Reject interval-mode tokens (start with sign or digit)
    first = token.lstrip()[0] if token.lstrip() else ""
    if first in ("+", "-") or first.isdigit():
        return None

    # Strip duration suffix
    name = token.split("_")[0]
    # Take first note of chord
    name = name.split(".")[0]

    # Split into pitch-class string and octave digits
    # e.g. "C#4" -> pc="C#", octave=4
    i = len(name)
    while i > 0 and (name[i-1].isdigit() or name[i-1] == "-"):
        i -= 1
    pc_str  = name[:i]
    oct_str = name[i:]

    if pc_str not in _PC_SEMITONE:
        return None
    try:
        octave = int(oct_str)
    except (ValueError, TypeError):
        return None

    # MIDI convention: C4 = 60 -> (octave+1)*12 + semitone_within_octave
    return (octave + 1) * 12 + _PC_SEMITONE[pc_str]


def pitch_class_histogram(tokens: List[str]) -> np.ndarray:
    """
    Return a L1-normalized 12-dim pitch-class frequency vector.
    Zeros for all-rest sequences (no pitches).
    """
    hist = np.zeros(12, dtype=np.float32)
    for tok in tokens:
        pc = _token_to_pitch_class(tok)
        if pc is not None:
            hist[pc] += 1.0
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def interval_histogram(tokens: List[str]) -> np.ndarray:
    """
    25-dim histogram of melodic intervals in semitones (-12 ... +12).
    Captures directional contour; bin 12 = 0 (unison/same pitch).
    """
    hist = np.zeros(25, dtype=np.float32)
    prev_pc = None
    for tok in tokens:
        pc = _token_to_pitch_class(tok)
        if pc is None:
            continue
        if prev_pc is not None:
            interval = pc - prev_pc          # in [-11, +11]
            hist[interval + 12] += 1.0
        prev_pc = pc
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def music_feature_vector(tokens: List[str]) -> np.ndarray:
    """
    Concatenate pitch-class (12-d) + interval (25-d) histograms -> 37-d vector.
    This compact representation is human-interpretable and directly maps to
    tonal fingerprint analysis in music-information-retrieval literature.
    """
    return np.concatenate([pitch_class_histogram(tokens),
                           interval_histogram(tokens)])


# ---------------------------------------------------------------------------
# Semitone interval histogram -- transposition-invariant  [R10]
# ---------------------------------------------------------------------------

def semitone_interval_histogram(tokens: List[str]) -> np.ndarray:
    """
    25-dim histogram of melodic intervals in true semitones (-12 ... +12),
    computed from full MIDI pitch numbers (octave-preserving).

    Transposition-invariant: the same melody transposed to any key produces
    an identical histogram.

    Bin layout:  index 0 = -12 (octave down),
                 index 12 = 0  (unison / repeated note),
                 index 24 = +12 (octave up).
    Intervals outside [-12, +12] are clamped to the nearest boundary bin.

    Contrast with interval_histogram() which uses pitch-CLASS differences
    (mod 12) and conflates leaps that differ by an octave.

    Parameters
    ----------
    tokens : baseline absolute-pitch token list (e.g. ['C4_quarter', ...])
             Interval-mode tokens ('+N'/'-N') are silently skipped.

    Returns
    -------
    L1-normalized 25-dim float32 array.  All-zeros for sequences with
    fewer than two pitched notes.
    """
    hist      = np.zeros(25, dtype=np.float32)
    prev_midi: int | None = None

    for tok in tokens:
        midi = _token_to_midi(tok)
        if midi is None:
            continue   # rest or unrecognized -- don't update prev_midi
        if prev_midi is not None:
            delta     = midi - prev_midi
            delta_clamped = max(-12, min(12, delta))
            hist[delta_clamped + 12] += 1.0
        prev_midi = midi

    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def music_feature_vector_extended(tokens: List[str]) -> np.ndarray:
    """
    62-dim feature vector combining all three histogram types:
      - pitch-class histogram       (12-dim, transposition-sensitive)
      - pitch-class interval hist   (25-dim, mod-12 intervals)
      - semitone interval histogram (25-dim, true semitone intervals)  [R10]

    The first component captures *what key* the piece is in.
    The second and third capture *how the melody moves*, with the third
    being fully transposition-invariant.

    Recommended for new analyses where transposition-invariance matters.
    Use music_feature_vector() (37-dim) if backwards-compatibility is needed.
    """
    return np.concatenate([
        pitch_class_histogram(tokens),       # 12-dim
        interval_histogram(tokens),          # 25-dim (pitch-class intervals)
        semitone_interval_histogram(tokens), # 25-dim (true semitone intervals)
    ])


def corpus_feature_matrix(corpus: Dict[str, List[str]]
                          ) -> Tuple[np.ndarray, List[str]]:
    """Return (matrix, names) using the 37-d feature vector."""
    names  = list(corpus.keys())
    matrix = np.stack([music_feature_vector(corpus[k]) for k in names])
    return matrix.astype(np.float32), names


# ---------------------------------------------------------------------------
# Distance/similarity utilities
# ---------------------------------------------------------------------------

def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine DISTANCE matrix (1 - similarity).
    Shape: (n, n).  Diagonal = 0.
    """
    sim = sklearn_cosine(embeddings)
    return (1.0 - sim).astype(np.float32)


def style_collapse_score(embeddings: np.ndarray) -> float:
    """
    A single scalar measuring how tightly a set of embeddings clusters.

    Defined as the mean pairwise cosine *distance* -- lower means the
    model's outputs are more similar to each other (style collapse).

    Usage: compare LLM-generated outputs vs. held-out human corpus.
    If LLM score << human score, the model is collapsing to a narrow style.
    """
    dist_mat = cosine_distance_matrix(embeddings)
    n = len(embeddings)
    if n < 2:
        return 0.0
    # Upper triangle only (no diagonal)
    upper_idx = np.triu_indices(n, k=1)
    return float(dist_mat[upper_idx].mean())


# ---------------------------------------------------------------------------
# PCA visualization helper
# ---------------------------------------------------------------------------

def pca_2d(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project embeddings to 2D for scatter-plot visualization."""
    pca = PCA(n_components=n_components)
    return pca.fit_transform(embeddings).astype(np.float32)

"""
tokenizer.py
------------
Converts symbolic music files (MIDI, ABC) into NLP-style token sequences
using music21 for parsing. Each note/chord becomes a discrete "word" that
downstream NLP metrics (n-gram, TF-IDF, LCS) can operate on directly.

--- BASELINE token format (default) ---
  Single note : "<pitch><octave>_<duration>"   e.g. "C4_quarter", "G#5_eighth"
  Chord       : "<p1>.<p2>.<p3>_<duration>"    e.g. "C4.E4.G4_half"
  Rest        : "REST_<duration>"               e.g. "REST_quarter"

  Duration is quantized to the nearest standard value to absorb trivial
  timing differences that would otherwise inflate distance scores.

--- INTERVAL token format (use_intervals=True) ---
  Encodes semitone deltas between consecutive non-rest notes rather than
  absolute pitch names.  Makes n-gram and LCS metrics transposition-invariant:
  a melody in C major and the same melody in D major produce identical tokens.

  Interval note : "<+/-N>_<duration>"   e.g. "+4_quarter", "-2_half", "0_eighth"
  Rest          : "REST_<duration>"     (unchanged -- rests have no pitch delta)

  The first note of each piece is skipped (no previous note to diff against).
  The interval is always relative to the last non-rest note (rests do not
  reset the pitch reference).

  Source: Malandrino et al. (2022) PINL representation; interval encoding
  principle also used in Mongeau & Sankoff (1990), Lemstrom & Ukkonen (2000).

  *** Anti-symmetry warning (Lin et al. 2011) ***
  When use_intervals=True, altering a SINGLE note changes TWO consecutive
  intervals (the one before and the one after that note).  A one-note
  substitution therefore registers as a distance-2 perturbation in any
  downstream n-gram or LCS metric, not distance-1.  This is a known
  limitation of all interval-based representations.  Keep it in mind when
  interpreting scores: interval-space distances can be unexpectedly large
  for near-copies that differ by isolated note changes.

--- DYNAMIC DURATION format (use_dynamic_duration=True) ---
  Replaces fixed duration labels ("quarter", "eighth") with integer multiples
  of the piece's own base note length.  Makes pieces in different time
  signatures or tempos directly comparable.

  Duration token : integer string  e.g. "1", "2", "4"  (multiples of base unit)

  Algorithm (Malandrino et al. 2022, PINL normalisation):
    1. Find d_min = shortest duration in the piece.
    2. If d_min appears in fewer than 3% of notes, discard those notes
       and repeat until d_min is sufficiently common.
    3. Express each note duration as round(ql / d_min), capped at 16.

  Can be combined with use_intervals=True for fully relative encoding.
  Overrides use_duration if both are set (use_dynamic_duration takes priority).
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional
import logging

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Duration quantisation map (music21 quarterLength -> label)
# ---------------------------------------------------------------------------
_DURATION_MAP = [
    (4.0,  "whole"),
    (3.0,  "dotted-half"),
    (2.0,  "half"),
    (1.5,  "dotted-quarter"),
    (1.0,  "quarter"),
    (0.75, "dotted-eighth"),
    (0.5,  "eighth"),
    (0.375,"dotted-sixteenth"),
    (0.25, "sixteenth"),
    (0.125,"thirty-second"),
]

def _quantize_duration(ql: float) -> str:
    """Snap a quarterLength float to the nearest named duration bucket."""
    best_label = "unknown"
    best_dist  = float("inf")
    for val, label in _DURATION_MAP:
        d = abs(ql - val)
        if d < best_dist:
            best_dist  = d
            best_label = label
    return best_label


def _compute_base_duration(quarter_lengths: List[float],
                           min_freq: float = 0.03) -> float:
    """
    Find the base note duration for dynamic normalisation (R2).

    Implements the PINL algorithm from Malandrino et al. (2022):
      1. Find d_min = minimum quarterLength in the piece.
      2. If d_min appears in fewer than `min_freq` fraction of all notes,
         discard notes of that duration and repeat.
      3. Return the surviving d_min as the base unit.

    Parameters
    ----------
    quarter_lengths : list of quarterLength floats (one per note/rest)
    min_freq        : minimum frequency fraction to accept d_min (default 0.03)

    Returns
    -------
    Base unit quarterLength (float).  Falls back to 0.25 (sixteenth) if
    the list is empty or all durations are filtered out.
    """
    if not quarter_lengths:
        return 0.25

    durations = [ql for ql in quarter_lengths if ql > 0]
    if not durations:
        return 0.25

    while durations:
        d_min  = min(durations)
        f_min  = durations.count(d_min) / len(durations)
        if f_min >= min_freq:
            return d_min
        # d_min is too rare -- filter it out and try the next shortest
        durations = [ql for ql in durations if ql != d_min]

    return 0.25  # fallback


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def file_to_tokens(filepath: str | Path, *, use_octave: bool = True,
                   use_duration: bool = True,
                   use_intervals: bool = False,
                   use_dynamic_duration: bool = False) -> List[str]:
    """
    Parse a MIDI or ABC file and return a flat list of string tokens.

    Parameters
    ----------
    filepath             : path to .mid / .midi / .abc file
    use_octave           : include octave in pitch token (default True)
                           Set False for pitch-class-only comparison.
                           Ignored when use_intervals=True.
    use_duration         : include duration in token (default True)
                           Set False for purely melodic contour comparison.
    use_intervals        : encode semitone intervals between consecutive notes
                           rather than absolute pitch names (default False).
                           Makes tokens transposition-invariant.
                           See anti-symmetry warning in module docstring.
    use_dynamic_duration : normalise durations relative to the piece's own
                           base note length rather than fixed labels (default
                           False).  Overrides use_duration when True.
                           Improves comparability across time signatures.

    Returns
    -------
    List of string tokens, one per note-event or rest.
    Returns [] on parse failure (logged as warning).
    """
    try:
        from music21 import converter
        score = converter.parse(str(filepath))
    except Exception as exc:
        log.warning("Could not parse %s: %s", filepath, exc)
        return []

    return _score_to_tokens(score, use_octave=use_octave,
                            use_duration=use_duration,
                            use_intervals=use_intervals,
                            use_dynamic_duration=use_dynamic_duration)


def tokens_from_string(abc_text: str, *, use_octave: bool = True,
                       use_duration: bool = True,
                       use_intervals: bool = False,
                       use_dynamic_duration: bool = False) -> List[str]:
    """
    Parse ABC notation directly from a string (handy for LLM-generated ABC
    that hasn't been saved to disk yet).

    Parameters mirror file_to_tokens() -- see that docstring for details.
    """
    try:
        from music21 import converter
        score = converter.parse(abc_text, format="abc")
    except Exception as exc:
        log.warning("Could not parse ABC string: %s", exc)
        return []

    return _score_to_tokens(score, use_octave=use_octave,
                            use_duration=use_duration,
                            use_intervals=use_intervals,
                            use_dynamic_duration=use_dynamic_duration)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_to_tokens(score, *, use_octave: bool, use_duration: bool,
                     use_intervals: bool = False,
                     use_dynamic_duration: bool = False) -> List[str]:
    """
    Core tokenisation logic.  Four modes depending on flag combinations:

      use_intervals=False, use_dynamic_duration=False  -> baseline (absolute pitch, fixed duration labels)
      use_intervals=True,  use_dynamic_duration=False  -> interval + fixed duration labels
      use_intervals=False, use_dynamic_duration=True   -> absolute pitch + dynamic duration
      use_intervals=True,  use_dynamic_duration=True   -> fully relative encoding
    """
    from music21 import note as m21note, chord as m21chord

    flat = score.flatten().notesAndRests

    # --- Pre-compute dynamic base duration if needed (R2) ---
    if use_dynamic_duration:
        all_qls = [el.quarterLength for el in flat]
        base_ql = _compute_base_duration(all_qls)
    else:
        base_ql = None

    def _dur_suffix(ql: float) -> str:
        """Return duration suffix string given the active duration mode."""
        if use_dynamic_duration:
            # Express as integer multiple of base unit, capped at 16
            multiple = max(1, min(16, round(ql / base_ql)))
            return f"_{multiple}"
        elif use_duration:
            return f"_{_quantize_duration(ql)}"
        else:
            return ""

    tokens: List[str] = []

    if use_intervals:
        # --- INTERVAL MODE (R1) ---
        # Track last non-rest MIDI pitch to compute semitone delta.
        # The first note is skipped (no previous pitch to diff against).
        prev_midi: Optional[int] = None

        for element in flat:
            suffix = _dur_suffix(element.quarterLength)

            if isinstance(element, m21note.Rest):
                # Rests are kept as-is; they do not update prev_midi
                tokens.append(f"REST{suffix}")

            elif isinstance(element, m21note.Note):
                curr_midi = element.pitch.midi
                if prev_midi is None:
                    # Skip the first note -- no interval to compute
                    prev_midi = curr_midi
                    continue
                delta = curr_midi - prev_midi
                sign  = "+" if delta >= 0 else ""
                tokens.append(f"{sign}{delta}{suffix}")
                prev_midi = curr_midi

            elif isinstance(element, m21chord.Chord):
                # For chords, use the lowest pitch as the reference pitch
                pitches   = sorted(element.pitches, key=lambda p: p.midi)
                curr_midi = pitches[0].midi
                if prev_midi is None:
                    prev_midi = curr_midi
                    continue
                delta = curr_midi - prev_midi
                sign  = "+" if delta >= 0 else ""
                # Append interval of bass note; upper voices as offsets
                upper = ".".join(
                    f"+{p.midi - curr_midi}" for p in pitches[1:]
                )
                chord_str = f"{sign}{delta}"
                if upper:
                    chord_str += f".{upper}"
                tokens.append(f"{chord_str}{suffix}")
                prev_midi = curr_midi

    else:
        # --- BASELINE / ABSOLUTE PITCH MODE ---
        for element in flat:
            suffix = _dur_suffix(element.quarterLength)

            if isinstance(element, m21note.Rest):
                tokens.append(f"REST{suffix}")

            elif isinstance(element, m21note.Note):
                pitch_str = _pitch_str(element.pitch, use_octave)
                tokens.append(f"{pitch_str}{suffix}")

            elif isinstance(element, m21chord.Chord):
                pitches   = sorted(element.pitches, key=lambda p: p.midi)
                pitch_str = ".".join(_pitch_str(p, use_octave) for p in pitches)
                tokens.append(f"{pitch_str}{suffix}")

    return tokens

def _pitch_str(pitch, use_octave: bool) -> str:
    """Return e.g. 'C#4' or 'C#' depending on flag."""
    name = pitch.name.replace("-", "b")   # normalise flats: B- -> Bb
    return f"{name}{pitch.octave}" if use_octave else name


# ---------------------------------------------------------------------------
# Convenience: build a "corpus" dict from a directory
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".mid", ".midi", ".abc", ".mxl", ".xml", ".musicxml"}

def load_corpus(directory: str | Path, limit: int = 0,
                **token_kwargs) -> dict[str, List[str]]:
    """
    Walk a directory tree, parse every supported file, and return
    {relative_path: token_list}.  Failures are skipped with a warning.

    limit        : stop after successfully parsing this many files (0 = no limit).
                   Applied early -- files beyond the limit are never read from disk.

    Per-file parse time is logged at DEBUG level -- run with
    logging.DEBUG to diagnose slow files (common on spinning disk with
    large MIDI files that music21 parses slowly).
    """
    import time
    corpus: dict[str, List[str]] = {}
    root   = Path(directory)
    files  = sorted(fp for fp in root.rglob("*")
                    if fp.suffix.lower() in SUPPORTED_EXTENSIONS
                    and not any(p.startswith("_dedup") for p in fp.parts))

    log.info("Found %d files to parse in %s", len(files), directory)
    if limit:
        log.info("  --limit %d: will stop after %d successful parses", limit, limit)
    t_start = time.time()

    for n, fp in enumerate(files, 1):
        # Early exit once we have enough successfully parsed files
        if limit and len(corpus) >= limit:
            log.info("  --limit %d reached, stopping early (%d files examined)",
                     limit, n - 1)
            break

        t0   = time.time()
        rel  = str(fp.relative_to(root))
        toks = file_to_tokens(fp, **token_kwargs)
        elapsed = time.time() - t0

        if toks:
            corpus[rel] = toks
            log.debug(
                "  [%d/%d] parsed %s -> %d tokens in %.2fs",
                n, len(files), fp.name, len(toks), elapsed,
            )
        else:
            log.warning("  [%d/%d] skipping empty/unparseable: %s (%.2fs)",
                        n, len(files), rel, elapsed)

        if elapsed > 5.0:
            log.warning(
                "  SLOW FILE: %s took %.1fs to parse -- "
                "consider truncating or pre-converting to a lighter format",
                fp.name, elapsed,
            )

    log.info(
        "Loaded %d/%d pieces from %s in %.1fs",
        len(corpus), len(files), directory, time.time() - t_start,
    )
    return corpus
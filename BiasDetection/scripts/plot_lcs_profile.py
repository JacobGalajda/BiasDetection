"""
plot_lcs_profile.py
-------------------
Generate the Appendix C sliding-window LCS profile plot for the
R18_C93 x R6_C34 transposed pair.

Shows:
  - Absolute-pitch LCS profile (score at each window position)
  - Interval LCS profile overlaid
  - Peak annotations
  - Key metadata (A major vs E major, lift=66%)

Usage
-----
    python scripts/plot_lcs_profile.py \
        --piece_a "data/llm_output/our_model_R18_C93_4_tracks_3_4_sig_A_key_8218.mid" \
        --piece_b "data/llm_output/our_model_R6_C34_4_tracks_2_3_sig_E_key_4110.mid" \
        --out     results/figures/appendix_c_lcs_profile.png \
        [--window 200] [--step 10]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from music_bias_analyzer.tokenizer import file_to_tokens
    from music_bias_analyzer.lcs import lcs_length
except ImportError as e:
    print(f"[error] {e}"); sys.exit(1)


# ---------------------------------------------------------------------------
# Interval tokenization
# ---------------------------------------------------------------------------
def to_interval_tokens(abs_tokens):
    import re as _re
    NOTE_TO_MIDI = {
        "C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,
        "F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11,
    }
    def tok_to_midi(tok):
        m = _re.match(r'^([A-Ga-g][#b]?)(-?\d+)', tok.split("_")[0])
        if not m: return None
        note, oct_ = m.group(1).capitalize(), int(m.group(2))
        s = NOTE_TO_MIDI.get(note)
        return None if s is None else s + (oct_+1)*12
    out, prev = [], None
    for tok in abs_tokens:
        parts = tok.split("_",1)
        dur = parts[1] if len(parts)>1 else "unknown"
        midi = tok_to_midi(tok)
        out.append(f"0_{dur}" if (midi is None or prev is None)
                   else f"{midi-prev}_{dur}")
        if midi is not None: prev = midi
    return out


# ---------------------------------------------------------------------------
# Sliding window LCS profile
# ---------------------------------------------------------------------------
def sliding_profile(query, reference, window, step):
    """
    Slide a window of size `window` over `reference` in steps of `step`.
    At each position, compute best-window LCS normalized by window size --
    matching the original audit metric (lcs_length / window_size).

    Also slides a window of same size over the query to find the best
    query fragment, then scores query_window vs reference_window.
    Returns (positions, scores).
    """
    positions = []
    scores    = []
    ref_len   = len(reference)
    qry_len   = len(query)

    # Clamp window to sequence lengths
    w = min(window, ref_len, qry_len)

    for i in range(0, max(1, ref_len - w + 1), step):
        ref_win = reference[i: i + w]
        # Find best matching query window for this reference position
        best = 0.0
        for j in range(0, max(1, qry_len - w + 1), max(1, step * 5)):
            qry_win = query[j: j + w]
            raw = lcs_length(qry_win, ref_win)
            score = raw / w  # normalize by window size (matches audit metric)
            if score > best:
                best = score
        positions.append(i)
        scores.append(best)

    return np.array(positions), np.array(scores)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--piece_a", required=True,
                    help="Query piece (R18_C93, A major)")
    ap.add_argument("--piece_b", required=True,
                    help="Reference piece (R6_C34, E major)")
    ap.add_argument("--out",     default="results/figures/appendix_c_lcs_profile.png")
    ap.add_argument("--window",  type=int, default=200,
                    help="Sliding window size in tokens (default: 200)")
    ap.add_argument("--step",    type=int, default=5,
                    help="Window step size in tokens (default: 5)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -- Load tokens --------------------------------------------------------
    print(f"Loading piece A: {Path(args.piece_a).name}")
    toks_a_abs = file_to_tokens(args.piece_a)
    print(f"  {len(toks_a_abs)} absolute tokens")

    print(f"Loading piece B: {Path(args.piece_b).name}")
    toks_b_abs = file_to_tokens(args.piece_b)
    print(f"  {len(toks_b_abs)} absolute tokens")

    toks_a_int = to_interval_tokens(toks_a_abs)
    toks_b_int = to_interval_tokens(toks_b_abs)

    # -- Compute profiles ---------------------------------------------------
    print(f"\nComputing absolute-pitch LCS profile "
          f"(window={args.window}, step={args.step}) ...")
    pos_abs, scores_abs = sliding_profile(
        toks_a_abs, toks_b_abs, args.window, args.step)

    print(f"Computing interval LCS profile ...")
    pos_int, scores_int = sliding_profile(
        toks_a_int, toks_b_int, args.window, args.step)

    # -- Summary stats ------------------------------------------------------
    peak_abs = float(scores_abs.max()) if len(scores_abs) else 0
    peak_int = float(scores_int.max()) if len(scores_int) else 0
    peak_abs_pos = int(pos_abs[scores_abs.argmax()]) if len(scores_abs) else 0
    peak_int_pos = int(pos_int[scores_int.argmax()]) if len(scores_int) else 0
    lift = (peak_int - peak_abs) / peak_abs * 100 if peak_abs > 0 else float("inf")

    print(f"\n  Peak abs LCS: {peak_abs:.4f} at token {peak_abs_pos}")
    print(f"  Peak int LCS: {peak_int:.4f} at token {peak_int_pos}")
    print(f"  Lift: {lift:.1f}%")

    # -- Plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor("#0f0f0f")
    fig.patch.set_facecolor("#0f0f0f")

    # Profiles
    ax.plot(pos_abs, scores_abs, color="#457B9D", linewidth=1.5,
            label=f"Absolute-pitch LCS (peak={peak_abs:.3f})", alpha=0.85)
    ax.plot(pos_int, scores_int, color="#E63946", linewidth=1.5,
            label=f"Interval LCS (peak={peak_int:.3f})", alpha=0.85)

    # Shaded area between curves (lift region)
    common_len = min(len(pos_abs), len(pos_int))
    ax.fill_between(
        pos_int[:common_len],
        scores_abs[:common_len],
        scores_int[:common_len],
        where=(scores_int[:common_len] > scores_abs[:common_len]),
        alpha=0.15, color="#E63946",
        label=f"Interval lift (+{lift:.0f}%)"
    )

    # Peak annotations
    ax.axvline(peak_int_pos, color="#E63946", linestyle="--",
               linewidth=0.8, alpha=0.6)
    # Place int peak label as fixed text to avoid edge clipping
    ax.text(0.98, 0.97,
            f"Peak int @ token {peak_int_pos}: {peak_int:.3f}",
            transform=ax.transAxes,
            fontsize=8, color="#E63946", ha="right", va="top",
            bbox=dict(facecolor="#1a1a1a", edgecolor="#333333",
                      boxstyle="round,pad=0.3"))

    if abs(peak_abs_pos - peak_int_pos) > args.step * 2:
        ax.axvline(peak_abs_pos, color="#457B9D", linestyle="--",
                   linewidth=0.8, alpha=0.6)
        ax.annotate(
            f"Peak abs\n@ token {peak_abs_pos}\n{peak_abs:.3f}",
            xy=(peak_abs_pos, peak_abs),
            xytext=(peak_abs_pos + max(10, len(pos_abs)//10), peak_abs + 0.02),
            fontsize=8, color="#457B9D",
            arrowprops=dict(arrowstyle="->", color="#457B9D", lw=0.8),
        )

    # Flag threshold line
    ax.axhline(0.387, color="#E9C46A", linestyle=":", linewidth=1.0,
               alpha=0.7, label="Flag threshold (T=0.387)")

    # Labels
    ax.set_xlabel("Reference token position (R6_C34, E major)",
                  color="#aaaaaa", fontsize=10)
    ax.set_ylabel("LCS similarity", color="#aaaaaa", fontsize=10)
    ax.set_title(
        f"Appendix C -- Sliding-window LCS Profile\n"
        f"R18_C93 (A major, 3/4) vs R6_C34 (E major, 2/3)  "
        f"[window={args.window}, step={args.step}]",
        color="white", fontsize=11, pad=10
    )

    ax.tick_params(colors="#777777", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    legend = ax.legend(facecolor="#1a1a1a", labelcolor="white",
                       fontsize=9, framealpha=0.8, loc="upper right")

    # Metadata box
    meta = (
        f"abs LCS = {peak_abs:.3f}   "
        f"int LCS = {peak_int:.3f}   "
        f"lift = +{lift:.0f}%\n"
        f"A major (3/4) -> E major (2/3)  |  "
        f"Perfect fourth transposition"
    )
    ax.text(0.02, 0.04, meta, transform=ax.transAxes,
            fontsize=8, color="#aaaaaa",
            bbox=dict(facecolor="#1a1a1a", edgecolor="#333333",
                      boxstyle="round,pad=0.4"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
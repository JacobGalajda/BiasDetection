"""
visualize_composer_space.py
---------------------------
Generate composer style space visualizations from a trained composer model.

Produces:
  1. composer_space_umap.png  -- UMAP scatter, colored by composer,
                                 with centroids and Voronoi boundaries
  2. composer_space_pca.png   -- PCA version (faster, deterministic)
  3. composer_attribution.png -- Bar chart: similarity profile for a query piece

Usage
-----
    python scripts/visualize_composer_space.py \
        --model    results/composer/composer_model.json \
        --out_dir  results/composer/figures \
        [--query   path/to/unknown.mid] \
        [--n_pieces 20]   # pieces per composer to plot (default: 20)
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from music_bias_analyzer.tokenizer import file_to_tokens
except ImportError as e:
    print(f"[error] {e}"); sys.exit(1)

# ---------------------------------------------------------------------------
# Colour palette -- distinct enough for 12 composers
# ---------------------------------------------------------------------------
PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#6A4C93", "#1982C4", "#8AC926", "#FF595E", "#6A994E",
    "#A8DADC", "#C77DFF",
]

# ---------------------------------------------------------------------------
# TF-IDF vector from tokens using saved IDF
# ---------------------------------------------------------------------------
def tfidf_vec(tokens, global_idf):
    tf = Counter(tokens)
    total = max(len(tokens), 1)
    vec = {tok: (tf[tok]/total) * global_idf.get(tok, 1.0) for tok in tf}
    norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
    return {tok: v/norm for tok, v in vec.items()}

def cosine(a, b):
    shared = set(a) & set(b)
    dot = sum(a[t]*b[t] for t in shared)
    na = math.sqrt(sum(v*v for v in a.values())) or 1e-9
    nb = math.sqrt(sum(v*v for v in b.values())) or 1e-9
    return dot / (na * nb)

def vec_to_array(vec, vocab):
    return np.array([vec.get(tok, 0.0) for tok in vocab])

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
# Build piece matrix for dimensionality reduction
# ---------------------------------------------------------------------------
def build_matrix(model, n_per_composer, cache_dir):
    """
    Sample n_per_composer train pieces per composer, build TF-IDF matrix.
    Returns (matrix, labels, piece_ids, vocab).
    """
    import random; random.seed(42)
    global_idf = model["global_idf"]
    composers  = model["composers"]
    piece_meta = model["piece_meta"]
    train_ids  = set(model["train_ids"])
    token_mode = model["token_mode"]

    # Group train pieces by composer
    by_comp = defaultdict(list)
    for pid, info in piece_meta.items():
        if pid in train_ids:
            by_comp[info["composer"]].append(pid)

    selected = []
    for comp in composers:
        items = by_comp[comp]
        random.shuffle(items)
        selected.extend([(pid, comp) for pid in items[:n_per_composer]])

    # Build shared vocab from centroids
    vocab = sorted(set(
        tok for cen in model["centroids"].values() for tok in cen
    ))
    vocab_idx = {tok: i for i, tok in enumerate(vocab)}

    # Load token cache and build vectors
    abs_cache = cache_dir / "abs"
    int_cache  = cache_dir / "int"
    used_cache = int_cache if token_mode == "interval" else abs_cache

    matrix = []
    labels = []
    pids   = []
    for pid, comp in selected:
        cache_path = used_cache / f"{pid}.json"
        if not cache_path.exists():
            continue
        toks = json.load(open(cache_path))
        if not toks:
            continue
        vec  = tfidf_vec(toks, global_idf)
        arr  = vec_to_array(vec, vocab)
        matrix.append(arr)
        labels.append(comp)
        pids.append(pid)

    return np.array(matrix), labels, pids, vocab

# ---------------------------------------------------------------------------
# Voronoi overlay on scatter
# ---------------------------------------------------------------------------
def add_voronoi(ax, centroids_2d, colors, xlim, ylim, alpha=0.08):
    """Shade Voronoi regions by painting nearest-centroid color per pixel."""
    try:
        res = 200
        xs = np.linspace(xlim[0], xlim[1], res)
        ys = np.linspace(ylim[0], ylim[1], res)
        xx, yy = np.meshgrid(xs, ys)
        grid   = np.stack([xx.ravel(), yy.ravel()], axis=1)

        # Distance to each centroid
        cents = np.array(centroids_2d)
        dists = np.linalg.norm(grid[:, None, :] - cents[None, :, :], axis=2)
        owners = np.argmin(dists, axis=1).reshape(res, res)

        img = np.zeros((res, res, 4))
        for i, col in enumerate(colors):
            rgba = np.array(to_rgba(col))
            mask = owners == i
            img[mask] = rgba
        img[:, :, 3] = alpha

        ax.imshow(img, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                  origin="lower", aspect="auto", zorder=0)
    except Exception:
        pass  # graceful degradation if imshow fails

# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------
def make_style_graph(matrix, labels, centroids_2d, cent_labels,
                     composer_colors, title, out_path,
                     query_2d=None, query_label=None,
                     method="UMAP"):
    composers = cent_labels
    color_map = {c: composer_colors[i % len(PALETTE)]
                 for i, c in enumerate(composers)}

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#0f0f0f")
    fig.patch.set_facecolor("#0f0f0f")

    # Voronoi background
    xlim = (matrix[:, 0].min()-0.5, matrix[:, 0].max()+0.5)
    ylim = (matrix[:, 1].min()-0.5, matrix[:, 1].max()+0.5)
    cent_colors = [color_map[c] for c in cent_labels]
    add_voronoi(ax, centroids_2d, cent_colors, xlim, ylim, alpha=0.12)

    # Piece scatter
    for comp in composers:
        idx = [i for i, l in enumerate(labels) if l == comp]
        if not idx: continue
        pts = matrix[idx]
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=color_map[comp], alpha=0.55, s=18,
                   linewidths=0, zorder=2)

    # Composer centroids
    for i, (comp, cent_pt) in enumerate(zip(cent_labels, centroids_2d)):
        ax.scatter(*cent_pt, c=color_map[comp],
                   s=220, marker="*", edgecolors="white",
                   linewidths=0.8, zorder=5)
        # Short composer label (last name)
        last = comp.split()[-1]
        ax.annotate(last, cent_pt,
                    fontsize=7.5, color="white", fontweight="bold",
                    ha="center", va="bottom",
                    xytext=(0, 7), textcoords="offset points",
                    zorder=6)

    # Query piece
    if query_2d is not None:
        ax.scatter(*query_2d, c="white", s=300, marker="X",
                   edgecolors="#FF595E", linewidths=1.5,
                   zorder=7, label=f"Query: {query_label or 'unknown'}")
        ax.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=9)

    # Legend
    handles = [mpatches.Patch(color=color_map[c],
                              label=c.split()[-1]) for c in composers]
    ax.legend(handles=handles, loc="lower right", ncol=2,
              facecolor="#1a1a1a", labelcolor="white",
              fontsize=8, framealpha=0.7)

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title(title, color="white", fontsize=13, pad=12)
    ax.set_xlabel(f"{method} 1", color="#aaaaaa", fontsize=9)
    ax.set_ylabel(f"{method} 2", color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#555555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")

# ---------------------------------------------------------------------------
# Attribution bar chart
# ---------------------------------------------------------------------------
def make_attribution_chart(scores, composer_colors, query_label, out_path):
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    composers = [r[0].split()[-1] for r in ranked]
    values    = [r[1] for r in ranked]
    colors    = [composer_colors[i] for i in range(len(ranked))]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#0f0f0f")
    fig.patch.set_facecolor("#0f0f0f")

    bars = ax.barh(composers[::-1], values[::-1],
                   color=colors[::-1], edgecolor="none", height=0.65)

    ax.set_xlabel("Cosine similarity to composer centroid",
                  color="#aaaaaa", fontsize=9)
    ax.set_title(f"Composer attribution: {query_label or 'query piece'}",
                 color="white", fontsize=11, pad=10)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    # Value labels
    for bar, val in zip(bars[::-1], values[::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", color="white", fontsize=7.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",      default="results/composer/composer_model.json")
    ap.add_argument("--out_dir",    default="results/composer/figures")
    ap.add_argument("--query",      default=None,
                    help="Path to unknown MIDI file for attribution")
    ap.add_argument("--query_label",default=None,
                    help="Display label for query piece")
    ap.add_argument("--n_pieces",   type=int, default=20,
                    help="Pieces per composer to plot (default 20)")
    ap.add_argument("--no_umap",    action="store_true",
                    help="Skip UMAP (use PCA only -- faster)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model} ...")
    model      = json.load(open(args.model))
    composers  = model["composers"]
    centroids  = model["centroids"]
    global_idf = model["global_idf"]
    token_mode = model["token_mode"]
    cache_dir  = Path(args.model).parent / "tokens"

    comp_colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(composers)}

    # -- Build piece matrix -------------------------------------------------
    print(f"Building piece matrix ({args.n_pieces} per composer) ...")
    matrix, labels, pids, vocab = build_matrix(
        model, args.n_pieces, cache_dir)
    print(f"  Matrix shape: {matrix.shape}")

    # -- Add centroids to matrix for reduction -----------------------------
    cent_matrix = np.array([
        vec_to_array(centroids[c], vocab) for c in composers
    ])
    full_matrix = np.vstack([matrix, cent_matrix])

    # -- PCA ---------------------------------------------------------------
    print("Running PCA ...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    full_pca = pca.fit_transform(full_matrix)
    piece_pca = full_pca[:len(matrix)]
    cent_pca  = full_pca[len(matrix):]

    make_style_graph(
        piece_pca, labels, cent_pca, composers,
        [comp_colors[c] for c in composers],
        f"Composer Style Space -- PCA ({token_mode} tokens)",
        out_dir / "composer_space_pca.png",
        method="PCA"
    )

    # -- UMAP --------------------------------------------------------------
    if not args.no_umap:
        try:
            import umap
            print("Running UMAP ...")
            reducer = umap.UMAP(n_components=2, random_state=42,
                                n_neighbors=15, min_dist=0.1)
            full_umap = reducer.fit_transform(full_matrix)
            piece_umap = full_umap[:len(matrix)]
            cent_umap  = full_umap[len(matrix):]

            make_style_graph(
                piece_umap, labels, cent_umap, composers,
                [comp_colors[c] for c in composers],
                f"Composer Style Space -- UMAP ({token_mode} tokens)",
                out_dir / "composer_space_umap.png",
                method="UMAP"
            )
        except ImportError:
            print("  umap-learn not installed -- skipping UMAP.")
            print("  Install with: pip install umap-learn")

    # -- Query attribution -------------------------------------------------
    if args.query:
        print(f"\nComputing attribution for: {args.query}")
        try:
            toks = file_to_tokens(args.query)
            if token_mode == "interval":
                toks = to_interval_tokens(toks)
            q_vec = tfidf_vec(toks, global_idf)
            scores = {c: cosine(q_vec, centroids[c]) for c in composers}
            ranked = sorted(scores.items(), key=lambda x: -x[1])

            print("\n  Composer attribution ranking:")
            for comp, score in ranked:
                print(f"    {score:.4f}  {comp}")

            make_attribution_chart(
                scores,
                [comp_colors[c] for c in composers],
                args.query_label or Path(args.query).stem,
                out_dir / "composer_attribution.png"
            )
        except Exception as e:
            print(f"  [error] {e}")

    print(f"\nAll figures saved to {out_dir}/")
    print(f"Classification accuracy: "
          f"Top-1={model['results']['top1_accuracy']:.3f}  "
          f"Top-3={model['results']['top3_accuracy']:.3f}")

if __name__ == "__main__":
    main()

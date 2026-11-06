"""
visualise.py
------------
Generates matplotlib figures from AnalysisReport objects.

Outputs:
  1. UMAP/PCA scatter plot -- coloured by corpus group, markers by flag status
  2. Heat-map grid -- pairwise similarity matrix (LCS or n-gram)
  3. Style-collapse bar chart -- intra-group mean cosine distance
  4. Interval histogram overlay -- compare tonal fingerprints across groups
  5. Sliding-window LCS profile -- for a specific suspect pair

All functions return matplotlib Figure objects so callers can .savefig() or
.show() as needed.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

# Lazy imports so the module doesn't hard-fail if matplotlib unavailable
def _import_plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("pip install matplotlib to use visualise.py")


# ---------------------------------------------------------------------------
# Group display config -- canonical labels and colours
# ---------------------------------------------------------------------------

GROUP_COLORS = {
    "llm_output":        "#e74c3c",   # red
    "train_corpus":      "#3498db",   # blue
    "duped":             "#f39c12",   # orange  (displayed as human_baseline_sq)
    "human_baseline":    "#2ecc71",   # green
    "human_baseline_sq": "#f39c12",   # orange  (alias -- same as duped)
    "other":             "#95a5a6",   # grey
}

# Human-readable labels for legend and axis tick labels
GROUP_LABELS = {
    "llm_output":        "NotaGen output",
    "train_corpus":      "Train corpus (PDMX)",
    "duped":             "Human baseline SQ (PDMX 4-track)",
    "human_baseline":    "Human baseline (MAESTRO)",
    "human_baseline_sq": "Human baseline SQ (PDMX 4-track)",
    "other":             "Other",
}

# Display order in legends
GROUP_ORDER = [
    "llm_output",
    "train_corpus",
    "duped",
    "human_baseline_sq",
    "human_baseline",
    "other",
]


def _label(group: str) -> str:
    return GROUP_LABELS.get(group, group)


def _color(group: str) -> str:
    return GROUP_COLORS.get(group, "#555555")


# ---------------------------------------------------------------------------
# 1. UMAP / PCA scatter
# ---------------------------------------------------------------------------

def plot_pca(pca_coords: Dict[str, List[float]],
             name_to_group: Dict[str, str],
             flagged_names: Optional[set] = None,
             title: str = "Style Space (UMAP projection of TF-IDF tokens)"
             ):
    """
    2D scatter of piece embeddings (UMAP or PCA).
    * Colour = corpus group
    * Star marker (*) for flagged pieces
    * Hollow circle (o) for unflagged pieces
    """
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(11, 7))

    # Group points by group so legend order is deterministic
    from collections import defaultdict
    group_points: Dict[str, list] = defaultdict(list)
    for name, (x, y) in pca_coords.items():
        group = name_to_group.get(name, "other")
        is_flagged = bool(flagged_names and name in flagged_names)
        group_points[group].append((x, y, is_flagged, name))

    # Plot in canonical order
    seen_labels = set()
    for group in GROUP_ORDER + sorted(set(group_points) - set(GROUP_ORDER)):
        if group not in group_points:
            continue
        pts = group_points[group]
        color  = _color(group)
        legend = _label(group)

        # Unflagged
        unflagged = [(x, y) for x, y, fl, _ in pts if not fl]
        if unflagged:
            xs, ys = zip(*unflagged)
            lbl = legend if legend not in seen_labels else None
            ax.scatter(xs, ys, c=color, marker="o", s=40,
                       alpha=0.75, label=lbl,
                       edgecolors="white", linewidths=0.4)
            seen_labels.add(legend)

        # Flagged -- star, larger
        flagged_pts = [(x, y) for x, y, fl, _ in pts if fl]
        if flagged_pts:
            xs, ys = zip(*flagged_pts)
            lbl = f"{legend} (flagged)" if f"{legend} (flagged)" not in seen_labels else None
            ax.scatter(xs, ys, c=color, marker="*", s=180,
                       alpha=0.95, label=lbl,
                       edgecolors="white", linewidths=0.5)
            seen_labels.add(f"{legend} (flagged)")

    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Dimension 1", fontsize=10)
    ax.set_ylabel("Dimension 2", fontsize=10)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Pairwise heat-map
# ---------------------------------------------------------------------------

def plot_heatmap(matrix: np.ndarray, names: List[str],
                 title: str = "Pairwise Similarity Matrix",
                 vmin: float = 0.0, vmax: float = 1.0,
                 cmap: str = "YlOrRd"):
    """Heat-map of a pairwise similarity/distance matrix."""
    plt = _import_plt()
    n   = len(names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.5), max(5, n * 0.45)))

    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    short_names = [nm[:25] for nm in names]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Style-collapse bar chart
# ---------------------------------------------------------------------------

def plot_style_collapse(group_stats: dict, metric: str = "tfidf"):
    """
    Bar chart of intra-group mean cosine distance.
    Lower bars = more style-collapsed.
    metric: 'tfidf' or 'histogram'
    """
    plt = _import_plt()
    groups = list(group_stats.keys())
    field  = f"style_collapse_{metric}"
    values = [getattr(group_stats[g], field, 0) for g in groups]
    colors = [_color(g) for g in groups]
    labels = [_label(g) for g in groups]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, values, color=colors,
                  edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
    ax.set_ylabel("Mean pairwise cosine distance\n(lower = more style-collapsed)",
                  fontsize=9)
    ax.set_title(f"Style Collapse Score -- {metric.upper()} embeddings",
                 fontsize=11)
    ax.set_ylim(0, max(values) * 1.3 if values else 1.0)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Tonal fingerprint overlay
# ---------------------------------------------------------------------------

def plot_tonal_fingerprints(group_histograms: Dict[str, np.ndarray]):
    """
    Bar overlay of mean pitch-class histograms per group.
    group_histograms: {group_name: 12-dim pitch-class vector}
    """
    plt = _import_plt()
    PC_LABELS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    x = np.arange(12)
    n_groups = len(group_histograms)

    fig, ax = plt.subplots(figsize=(11, 4))
    width = 0.8 / n_groups

    for i, (group, hist) in enumerate(group_histograms.items()):
        offset = (i - n_groups / 2 + 0.5) * width
        ax.bar(x + offset, hist, width=width,
               label=_label(group), color=_color(group),
               alpha=0.8, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(PC_LABELS)
    ax.set_ylabel("Normalised frequency")
    ax.set_title("Tonal Fingerprint -- Mean Pitch-Class Histogram by Group")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Sliding-window LCS profile
# ---------------------------------------------------------------------------

def plot_lcs_profile(profile: List[Tuple[int, float]],
                     query_name: str = "query",
                     reference_name: str = "reference",
                     flag_threshold: float = 0.35):
    """
    Line plot of sliding-window LCS scores across a reference piece.
    Dashed horizontal line shows the flag threshold.
    """
    plt = _import_plt()
    positions = [p[0] for p in profile]
    scores    = [p[1] for p in profile]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(positions, scores, linewidth=1.2, color="#3498db", label="LCS sim")
    ax.fill_between(positions, scores, alpha=0.15, color="#3498db")
    ax.axhline(flag_threshold, linestyle="--", color="#e74c3c",
               linewidth=1, label=f"Flag threshold ({flag_threshold})")

    if scores:
        peak_pos = positions[np.argmax(scores)]
        peak_val = max(scores)
        ax.annotate(f"peak={peak_val:.3f}",
                    xy=(peak_pos, peak_val),
                    xytext=(peak_pos, peak_val + 0.03),
                    fontsize=8, color="#e74c3c",
                    arrowprops=dict(arrowstyle="->", color="#e74c3c"))

    ax.set_xlabel("Reference token position (window start)")
    ax.set_ylabel("LCS similarity")
    ax.set_title(f"Sliding-window LCS: '{query_name}'  <->  '{reference_name}'")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience: save all standard figures from a report
# ---------------------------------------------------------------------------

def save_all_figures(report, output_dir: str = "results/figures/",
                     dpi: int = 150) -> List[str]:
    """
    Generate and save all standard report figures.
    Returns list of saved file paths.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    plt = _import_plt()
    saved: List[str] = []

    name_to_group = {}
    for pr in report.pair_results:
        name_to_group[pr.piece_a] = pr.group_a
        name_to_group[pr.piece_b] = pr.group_b

    # UMAP / PCA scatter
    if report.pca_coords:
        flagged = ({p.piece_a for p in report.flagged_pairs} |
                   {p.piece_b for p in report.flagged_pairs})
        fig  = plot_pca(report.pca_coords, name_to_group, flagged)
        path = os.path.join(output_dir, "pca_scatter.png")
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        saved.append(path)

    # Style collapse bars
    fig  = plot_style_collapse(report.group_stats)
    path = os.path.join(output_dir, "style_collapse.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    saved.append(path)

    return saved
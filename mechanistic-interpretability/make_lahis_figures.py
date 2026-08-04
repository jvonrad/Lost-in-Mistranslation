"""Regenerate the LAHIS paper figures from the vendored head-importance tensors.

Replaces the Colab-only plotting cells of LAHIS_pipeline_olmo2.ipynb and
`src/heatmap_viz.plot_top_head_distribution`.

WHY THE REWRITE (reviewer request): the original per-language layer plot drew
every language with the SAME marker --

    ax.plot(range(num_layers), pct, "-o", markersize=4, label=lan.upper())

-- so language identity was carried by colour alone. With 12 languages and
matplotlib's default 10-colour cycle that is not merely an accessibility
problem, it is wrong: the cycle wraps, so EN/JA are both C0 blue and DE/ZH are
both C1 orange. Two pairs of languages were literally indistinguishable, in the
plot and in the legend.

Each language now gets a unique (marker, colour, linestyle) triple, so it stays
identifiable in greyscale, under any colour-vision deficiency, and after the
lossy downscaling a two-column PDF gets. Linestyle additionally encodes the
paper's High/Low split -- solid for high-resource, dashed for the three
low-resource languages -- which is free information the old plot discarded.

Outputs (PDF for LaTeX + PNG for quick viewing) into results/lahis_figures/.

Usage:
  python mechanistic-interpretability/make_lahis_figures.py
  python mechanistic-interpretability/make_lahis_figures.py --outdir results/lahis_figures
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lahis_analysis import (  # noqa: E402
    LANGUAGES, MODELS, TOPK_P, layer_profile, load_all, n_top, overlap_matrix,
)

LOW_RESOURCE = {"id", "bn", "sw"}

# One unique marker per language -- the reviewer's request, and the thing that
# makes the plot readable without colour. Chosen to stay distinguishable at
# small size: no two are rotations of each other.
MARKERS = {
    "en": "o", "de": "s", "pt": "^", "ar": "v", "es": "D", "ru": "P",
    "fr": "X", "ja": "*", "zh": "<", "id": ">", "bn": "h", "sw": "p",
}
# Paul Tol qualitative palette (colour-vision-deficiency safe) extended to 12.
# Colour is now REDUNDANT with the marker rather than load-bearing.
COLORS = {
    "en": "#4477AA", "de": "#EE6677", "pt": "#228833", "ar": "#CCBB44",
    "es": "#66CCEE", "ru": "#AA3377", "fr": "#BBBBBB", "ja": "#EE7733",
    "zh": "#0077BB", "id": "#009988", "bn": "#CC3311", "sw": "#332288",
}


def style(lang):
    return dict(
        marker=MARKERS[lang],
        color=COLORS[lang],
        # Low-resource languages dashed: survives greyscale, and encodes the
        # High/Low split the results tables use.
        linestyle="--" if lang in LOW_RESOURCE else "-",
        linewidth=1.3,
        markersize=5.5,
        markeredgecolor="black",
        markeredgewidth=0.4,
        alpha=0.9,
        label=lang.upper(),
    )


def fig_layer_distribution(model, outdir, repo_root=".", topk_p=TOPK_P,
                           plot_style="line"):
    """Per-language layer profile of the top-k% heads -- one series per language.

    plot_style="line"  faithful recreation of the original figure, with the
                       marker/colour/linestyle fix. Connecting lines are kept.
    plot_style="dots"  markers only, zero-valued layers omitted. Only 20 heads
                       are spread over 32 layers, so ~75% of the (language,
                       layer) cells are exactly 0 and the connecting lines are
                       almost entirely travel between zeros -- ink that encodes
                       nothing. Dropping them turns the figure into a clean
                       categorical scatter where each language's marker is
                       actually legible. Absence of a marker means 0%.
    """
    mats = load_all(model, LANGUAGES, repo_root)
    if not mats:
        print(f"[skip] no tensors for {model}")
        return None
    _, _, label = MODELS[model]

    if plot_style == "dots":
        return _fig_dots(model, mats, label, outdir, topk_p)

    fig, ax = plt.subplots(figsize=(11, 5.0))
    langs = list(mats)
    # DODGE. Only n_top=20 heads are distributed over 32 layers, so every value
    # is a multiple of 5% and languages coincide constantly -- at layer 2 all
    # twelve sit exactly on 0. Drawn at integer x, the last-plotted language
    # simply hides the other eleven, which is what made distinct markers alone
    # insufficient. Each language is offset to its own sub-slot within the layer
    # so co-located points fan out instead of stacking. This is a rendering
    # offset ONLY: the y-values and the layer each point belongs to are exact.
    span = 0.72
    offs = np.linspace(-span / 2, span / 2, len(langs))
    for off, lang in zip(offs, langs):
        pct = layer_profile(mats[lang], topk_p)
        ax.plot(np.arange(len(pct)) + off, pct, **style(lang))

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel(f"% of top-{topk_p*100:.0f}% heads in layer", fontsize=11)
    ax.set_title(f"Where language-important heads cluster — {label}", fontsize=12)
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.set_xlim(-0.5, len(pct) - 0.5)
    ax.margins(y=0.08)
    # Legend outside: 12 entries would otherwise cover the layer-0 spike, which
    # is the feature the figure exists to show.
    ax.legend(ncol=6, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              frameon=False, handlelength=2.6, columnspacing=1.2)

    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f"head_distribution_by_layer_{model}")
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {stem}.pdf / .png")
    return stem


def _fig_dots(model, mats, label, outdir, topk_p):
    """Marker-only variant: one row per language, marker size = % of top heads."""
    langs = list(mats)
    profiles = {l: layer_profile(m, topk_p) for l, m in mats.items()}
    n_layers = len(next(iter(profiles.values())))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    for row, lang in enumerate(langs):
        pct = profiles[lang]
        nz = np.nonzero(pct)[0]
        if len(nz) == 0:
            continue
        s = style(lang)
        # Marker area proportional to the percentage, so the layer-0 spike still
        # reads as a spike without needing a second axis.
        ax.scatter(nz, np.full(len(nz), row), s=pct[nz] * 9.0,
                   marker=s["marker"], color=s["color"],
                   edgecolor="black", linewidth=0.4, alpha=0.85, zorder=3)
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels([f"{MARKERS[l]}  {l.upper()}" for l in langs], fontsize=10)
    for row, lang in enumerate(langs):
        if lang in LOW_RESOURCE:
            ax.get_yticklabels()[row].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Language", fontsize=11)
    ax.set_title(f"Where language-important heads cluster — {label}\n"
                 f"(marker area = % of that language's top-{topk_p*100:.0f}% heads; "
                 f"no marker = 0%; bold = low-resource)", fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_xlim(-0.8, n_layers - 0.2)
    ax.set_ylim(len(langs) - 0.4, -0.6)

    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f"head_distribution_dots_{model}")
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {stem}.pdf / .png")
    return stem


def fig_pairwise_overlap(model, outdir, repo_root=".", topk_p=TOPK_P):
    """12x12 % of shared top-k heads between every language pair."""
    mats = load_all(model, LANGUAGES, repo_root)
    if not mats:
        return None
    _, _, label = MODELS[model]
    mat, langs = overlap_matrix(mats, topk_p)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    # cividis, not YlOrRd: perceptually uniform and CVD-safe, so the cell
    # annotations are not the only way to read a value.
    im = ax.imshow(mat, cmap="cividis", vmin=0, vmax=100)
    ax.set_xticks(range(len(langs)))
    ax.set_yticks(range(len(langs)))
    # Prepend each language's marker glyph to its tick label so the two figures
    # share one visual key for language identity.
    labels = [f"{MARKERS[l]} {l.upper()}" for l in langs]
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(langs)):
        for j in range(len(langs)):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if mat[i, j] < 55 else "black")
    ax.set_title(f"Top-{topk_p*100:.0f}% head overlap (%) — {label}", fontsize=12)
    fig.colorbar(im, ax=ax, label="% shared heads", fraction=0.046, pad=0.04)

    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f"pairwise_overlap_{model}")
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {stem}.pdf / .png")
    return stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--outdir", default="results/lahis_figures")
    ap.add_argument("--models", nargs="*", default=["olmo2", "olmo2_ft"])
    a = ap.parse_args()

    for m in a.models:
        fig_layer_distribution(m, a.outdir, a.repo_root, plot_style="line")
        fig_layer_distribution(m, a.outdir, a.repo_root, plot_style="dots")
        fig_pairwise_overlap(m, a.outdir, a.repo_root)

    # numbers worth quoting alongside the figures
    print("\nmean off-diagonal overlap:")
    for m in a.models:
        mats = load_all(m, LANGUAGES, a.repo_root)
        if not mats:
            continue
        mat, langs = overlap_matrix(mats)
        off = mat[~np.eye(len(langs), dtype=bool)]
        print(f"  {MODELS[m][2]:<20} {off.mean():.1f}%  "
              f"(n_top={n_top(next(iter(mats.values())))}, {len(langs)} langs)")


if __name__ == "__main__":
    main()

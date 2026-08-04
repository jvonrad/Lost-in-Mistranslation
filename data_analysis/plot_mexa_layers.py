#!/usr/bin/env python
"""Cross-lingual alignment among NON-ENGLISH pairs, layer by layer.

X-Y is the mean top-1 retrieval over the 55 language pairs that exclude English
-- how close non-English languages sit to each other. It is the half of the MEXA
measurement that separates "builds a shared multilingual space" from "pulls
everything toward English": a method that centralises on English raises EN-X
without raising X-Y.

Plotted as a CHANGE from the base model, because the absolute curves for five
models sit within ~0.05 of each other and overlap illegibly; the deltas are the
finding. Base is therefore the zero line by construction.

Both variants are shown side by side:
  raw       embeddings as-is; dominated by the per-language centroid, which is
            what a centralisation effect would move
  centered  each language's sentence-mean removed first; isolates the finer
            translation structure from that centroid

Usage:
  python data_analysis/plot_mexa_layers.py --family olmo
  python data_analysis/plot_mexa_layers.py --family olmo --metric en_x
"""
from __future__ import annotations

import argparse
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"SFT": "#4477AA", "DCO": "#228833", "CM-Align": "#CCBB44",
          "GRPO": "#EE6677", "CPT+GRPO": "#AA3377"}
MARKERS = {"SFT": "s", "DCO": "^", "CM-Align": "D", "GRPO": "o", "CPT+GRPO": "v"}

FAMILIES = {
    "olmo": ("OLMo-2-1124-7B", "olmo-base", [
        ("SFT", "olmo-sft-10k"), ("DCO", "olmo-dco-10k"),
        ("CM-Align", "olmo-cmalign-10k"), ("GRPO", "olmo-grpo-attmlp"),
        ("CPT+GRPO", "olmo-cpt-grpo")]),
    "qwen": ("Qwen-2.5-7B", "qwen-base", [
        ("SFT", "qwen-sft-10k"), ("DCO", "qwen-dco-10k"),
        ("CM-Align", "qwen-cmalign-10k"), ("GRPO", "qwen-grpo")]),
}


def load(tag):
    p = f"results/mexa/{tag}_mexa.json"
    return json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="olmo", choices=sorted(FAMILIES))
    ap.add_argument("--metric", default="x_y", choices=["x_y", "en_x"])
    ap.add_argument("--width", type=float, default=7.0,
                    help="inches; 7.0 = ACL \\textwidth for a two-column figure*")
    ap.add_argument("--height", type=float, default=2.6)
    ap.add_argument("--variants", nargs="+", default=["raw"],
                    choices=["raw", "centered"],
                    help="raw is the default: it retains the per-language "
                         "centroid, which is what a centralisation effect moves")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    title, base_tag, spec = FAMILIES[a.family]
    b = load(base_tag)
    if b is None:
        raise SystemExit(f"missing MEXA result for {base_tag}")

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, len(a.variants), figsize=(a.width, a.height),
                             sharey=True, squeeze=False)
    axes = axes[0]

    label = {"x_y": "non-English pairs (X–Y)", "en_x": "English pairs (EN–X)"}[a.metric]
    for ax, variant in zip(axes, a.variants):
        bc = np.array(b["alignment"][variant]["per_layer"][a.metric])
        nl = len(bc)
        ax.axhline(0, color="#666666", linewidth=0.8, zorder=2)
        for n, t in spec:
            v = load(t)
            if v is None:
                continue
            d = np.array(v["alignment"][variant]["per_layer"][a.metric]) - bc
            ax.plot(np.arange(nl), d, color=COLORS[n], marker=MARKERS[n],
                    markersize=2.6, linewidth=1.1, label=n, zorder=3,
                    markeredgewidth=0)
        ax.set_xlabel("Layer", fontsize=9)
        if len(a.variants) > 1:
            ax.set_title(variant, fontsize=9, pad=3)
        ax.grid(alpha=0.25, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(-0.5, nl - 0.5)
    axes[0].set_ylabel(f"$\\Delta$ alignment vs base\n{label}", fontsize=9)
    axes[0].legend(ncol=5, fontsize=8.5, frameon=False, loc="upper center",
                   bbox_to_anchor=(0.5 if len(a.variants) == 1 else 1.03, 1.22),
                   handlelength=1.6, columnspacing=1.6, handletextpad=0.4)

    fig.tight_layout()
    out = a.out or f"results/mexa/fig_mexa_layers_{a.metric}_{a.family}.pdf"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.rsplit(".", 1)[0] + "." + ext, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.rsplit('.',1)[0]}.pdf / .png")
    for variant in a.variants:
        bc = np.array(b["alignment"][variant]["per_layer"][a.metric])
        print(f"  {variant}: base peaks {bc.max():.3f} @L{int(bc.argmax())}")
        for n, t in spec:
            v = load(t)
            if v is None:
                continue
            d = np.array(v["alignment"][variant]["per_layer"][a.metric]) - bc
            print(f"    {n:<10} max {d.max():+.3f} @L{int(d.argmax()):<3} min {d.min():+.3f} @L{int(d.argmin())}")


if __name__ == "__main__":
    main()

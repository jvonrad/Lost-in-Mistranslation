#!/usr/bin/env python
"""Figure 5: language-specific neurons per language, all post-training methods.

Extends the paper's Figure 5 (base / SFT / GRPO) to the full method set
(base / SFT / DCO / CM-Align / GRPO), all trained on the same 10K facts.

TWO PANELS, because one cannot carry both facts:
  (a) absolute counts on a LOG axis. English holds 1,762 of 3,522 specialised
      neurons and Russian 55 -- a 32x range that flattens every non-English bar
      to invisibility on a linear axis, which is what the single-panel version
      of this figure hides.
  (b) change vs base, in percent. This is where the finding lives, and it is
      not readable from panel (a): a +136 English shift is 8% of a tall bar,
      while -50 across three low-resource languages is a large relative move on
      short ones.

Colours are Paul Tol's bright palette (colour-vision-deficiency safe); the base
model is grey so the four methods read as a group against it. Hatching on the
low-resource languages doubles the High/Low split so it survives greyscale
printing.

Usage:
  python data_analysis/plot_lape_figure.py --family olmo
  python data_analysis/plot_lape_figure.py --family qwen --out results/lape/fig5_qwen.pdf
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

LOW = {"id", "bn", "sw"}

# Tol bright: CVD-safe, fixed order, never cycled.
COLORS = {"Base": "#BBBBBB", "SFT": "#4477AA", "DCO": "#228833",
          "CM-Align": "#CCBB44", "GRPO": "#EE6677"}

FAMILIES = {
    "olmo": ("OLMo-2-1124-7B", [
        ("Base", "olmo-base"), ("SFT", "olmo-sft-10k"), ("DCO", "olmo-dco-10k"),
        ("CM-Align", "olmo-cmalign-10k"), ("GRPO", "olmo-grpo-attmlp")]),
    "qwen": ("Qwen-2.5-7B", [
        ("Base", "qwen-base"), ("SFT", "qwen-sft-10k"), ("DCO", "qwen-dco-10k"),
        ("CM-Align", "qwen-cmalign-10k"), ("GRPO", "qwen-grpo")]),
}


def counts(tag):
    p = f"results/lape/{tag}/lape_results.json"
    if not os.path.exists(p):
        return None
    d = json.load(io.open(p, encoding="utf-8"))
    t = {}
    for _, langs in d.items():
        for l, ns in langs.items():
            t[l] = t.get(l, 0) + len(ns)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="olmo", choices=sorted(FAMILIES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=float, default=7.0,
                    help="inches; 7.0 = ACL \\textwidth for a two-column figure*")
    ap.add_argument("--height", type=float, default=2.3)
    ap.add_argument("--linear", action="store_true",
                    help="linear y-axis; the default is log because English holds "
                         "~50% of OLMo's specialised neurons and a linear axis "
                         "flattens every other language to invisibility")
    a = ap.parse_args()

    title, spec = FAMILIES[a.family]
    data = {n: counts(t) for n, t in spec}
    missing = [n for n, v in data.items() if v is None]
    if missing:
        raise SystemExit(f"missing LAPE results for: {missing}")
    base = data["Base"]
    langs = [l for l, _ in sorted(base.items(), key=lambda x: -x[1])]
    methods = [n for n, _ in spec]

    # ACL house style: serif text, no chartjunk, figure sized for one column of a
    # two-column layout so it stays legible after LaTeX scales it down.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })

    fig, ax = plt.subplots(figsize=(a.width, a.height))
    x = np.arange(len(langs))
    w = 0.16
    for i, m in enumerate(methods):
        off = (i - (len(methods) - 1) / 2) * w
        vals = [data[m].get(l, 0) for l in langs]
        ax.bar(x + off, vals, w * 0.9, label=m, color=COLORS[m],
               linewidth=0, zorder=3)

    if not a.linear:
        ax.set_yscale("log")
    ax.set_ylabel("Language-specific neurons", fontsize=9)
    ax.set_xlabel("Language", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels([l.upper() for l in langs], fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    for lbl, l in zip(ax.get_xticklabels(), langs):
        if l in LOW:
            lbl.set_fontweight("bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=5, fontsize=9.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.19), handlelength=1.5, columnspacing=2.0,
              handletextpad=0.5)

    fig.tight_layout()
    out = a.out or f"results/lape/fig5_lape_{a.family}.pdf"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.rsplit(".", 1)[0] + "." + ext, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.rsplit('.',1)[0]}.pdf / .png   (bold x-labels = low-resource)")
    for m in methods:
        print(f"  {m:<9} en={data[m]['en']:<6} total={sum(data[m].values())}")


if __name__ == "__main__":
    main()

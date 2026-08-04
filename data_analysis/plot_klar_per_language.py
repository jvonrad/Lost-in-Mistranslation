#!/usr/bin/env python
"""Figure 4(b): KLAR accuracy per language, base vs GRPO.

Companion to 4(a), which gives the two aggregates. This shows whether the gain
is broad or carried by a few languages -- the question a mean cannot answer.

Languages are grouped trained-first, then held-out, each block sorted by base
accuracy, with a divider between them. Sorting inside a block (rather than
alphabetically) puts the languages the model is worst at on the right of each
group, where the low-resource behaviour is easiest to read.

The paired bars share a language, so the useful reading is vertical distance,
not bar height: English is 62.9 at base and dwarfs everything, but its +10.9
gain is smaller in relative terms than Arabic's +12.8 on a 7.9 base.

Usage:
  python data_analysis/plot_klar_per_language.py
  python data_analysis/plot_klar_per_language.py --grpo attmlp
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

TRAINED = ["en", "es", "fr", "ru", "zh", "ja", "ar"]
# Tol bright blue / orange: distinguishable under all common colour-vision
# deficiencies, unlike the red/green pairing.
C_BASE, C_GRPO = "#4477AA", "#EE7733"

GRPO_TAGS = {"ladder": "olmo-ladder10k-s6500", "attmlp": "olmo-grpo-attmlp"}


def per_lang(tag):
    p = f"results/klar/{tag}_klar_alllangs.json"
    if not os.path.exists(p):
        return None
    pl = json.load(io.open(p, encoding="utf-8"))["per_lang"]
    return {k: 100 * v[0] / v[1] for k, v in pl.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grpo", default="ladder", choices=sorted(GRPO_TAGS))
    ap.add_argument("--base", default="olmo-base")
    ap.add_argument("--width", type=float, default=7.0)
    ap.add_argument("--height", type=float, default=2.7)
    ap.add_argument("--out", default="results/fig4b_klar_per_language.pdf")
    a = ap.parse_args()

    b = per_lang(a.base)
    g = per_lang(GRPO_TAGS[a.grpo])
    if b is None or g is None:
        raise SystemExit("missing KLAR results")

    tr = sorted([l for l in b if l in TRAINED], key=lambda l: -b[l])
    ood = sorted([l for l in b if l not in TRAINED], key=lambda l: -b[l])
    langs = tr + ood

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(a.width, a.height))
    x = np.arange(len(langs))
    w = 0.4
    ax.bar(x - w / 2, [b[l] for l in langs], w, label="Base", color=C_BASE,
           edgecolor="black", linewidth=0.4, zorder=3)
    ax.bar(x + w / 2, [g[l] for l in langs], w, label="GRPO", color=C_GRPO,
           edgecolor="black", linewidth=0.4, zorder=3)

    # Divider between the trained block and the held-out block.
    ax.axvline(len(tr) - 0.5, color="#444444", linewidth=0.8, linestyle=":", zorder=4)
    top = max(max(b.values()), max(g.values()))
    ax.text((len(tr) - 1) / 2, top * 1.02, "trained", fontsize=8.5,
            ha="center", style="italic")
    ax.text(len(tr) + (len(ood) - 1) / 2, top * 1.02, "held-out", fontsize=8.5,
            ha="center", style="italic")

    ax.set_ylabel("KLAR accuracy (%)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([l.upper() for l in langs], fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlim(-0.6, len(langs) - 0.4)
    ax.set_ylim(0, top * 1.12)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=2, fontsize=9, frameon=False, loc="upper right",
              handlelength=1.4, columnspacing=1.6)

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(a.out.rsplit(".", 1)[0] + "." + ext, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {a.out.rsplit('.',1)[0]}.pdf / .png   (GRPO = {GRPO_TAGS[a.grpo]})")
    up = sum(1 for l in langs if g[l] > b[l])
    print(f"  improved in {up}/{len(langs)} languages")
    print("  " + "  ".join(f"{l}{g[l]-b[l]:+.1f}" for l in langs))


if __name__ == "__main__":
    main()

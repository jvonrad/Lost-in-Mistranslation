#!/usr/bin/env python
"""Figure 4(a): KLAR accuracy on trained vs held-out languages, across models.

Train  = the seven languages present in training (en, es, fr, ru, zh, ja, ar)
OOD    = the ten KLAR languages never trained on
Both are the MEAN of per-language accuracy, the aggregation that reproduces the
published baseline row (24.56 / 13.30 against the paper's 24.6 / 13.2).

Two corrections relative to the earlier version of this figure:
  SFT   now the fact-matched 10K pure-cross-entropy model (25.8 / 13.0). The
        released checkpoint used 40K facts and a consistency-weighted loss, and
        scored 21.9 / 8.8 -- i.e. it looked like SFT *regressed* on KLAR, which
        the matched model shows it does not.
  GRPO  olmo-2-7b-grpo-att-mlp-full, the checkpoint the results table reports.

The baseline is drawn as a horizontal reference line rather than a bar pair, so
"above or below base" is readable without comparing bar heights across the axis.

Usage:
  python data_analysis/plot_klar_figure.py
  python data_analysis/plot_klar_figure.py --grpo ladder   # alternate GRPO ckpt
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
C_TRAIN, C_OOD = "#4477AA", "#EE6677"

# GRPO checkpoint choice. `ladder` is the strongest OLMo GRPO on KLAR
# (33.0/19.2); `attmlp` is the one the results table reports (20.5/10.6) and has
# the consistency gain. They disagree by 12.5pp on KLAR-Train, so which one the
# figure shows must match what the text claims.
GRPO_TAGS = {"ladder": ("olmo-ladder10k-s6500", "GRPO"),
             "attmlp": ("olmo-grpo-attmlp", "GRPO")}


def klar(tag):
    p = f"results/klar/{tag}_klar_alllangs.json"
    if not os.path.exists(p):
        return None
    pl = json.load(io.open(p, encoding="utf-8"))["per_lang"]
    m = lambda L: 100 * sum(pl[l][0] / pl[l][1] for l in L) / len(L)
    return m([l for l in TRAINED if l in pl]), m([l for l in pl if l not in TRAINED])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grpo", default="ladder", choices=sorted(GRPO_TAGS))
    ap.add_argument("--full", action="store_true",
                    help="all nine models; default is Base / SFT / GRPO only")
    ap.add_argument("--width", type=float, default=7.0)
    ap.add_argument("--height", type=float, default=2.7)
    ap.add_argument("--out", default="results/fig4a_klar_olmo.pdf")
    a = ap.parse_args()

    grpo_tag, grpo_lbl = GRPO_TAGS[a.grpo]
    if a.full:
        spec = [("Base", "olmo-base"), ("SFT", "olmo-sft-10k"),
                ("DCO", "olmo-dco-10k"), ("CM-Align", "olmo-cmalign-10k"),
                (grpo_lbl, grpo_tag), ("CPT", "olmo-finetranslations"),
                ("CPT+SFT", "olmo-cpt-sft-10k"), ("CPT+DCO", "olmo-cpt-dco-10k"),
                ("CPT+GRPO",
                 "olmo-2-7b-finetranslation-wikifact-grpo-att-mlp-checkpoint")]
    else:
        spec = [("Base", "olmo-base"), ("SFT", "olmo-sft-10k"), (grpo_lbl, grpo_tag)]

    base = klar("olmo-base")
    if base is None:
        raise SystemExit("missing KLAR result for olmo-base")
    rows = [(n, klar(t)) for n, t in spec]
    missing = [n for n, v in rows if v is None]
    if missing:
        raise SystemExit(f"missing KLAR results for: {missing}")

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(a.width, a.height))
    x = np.arange(len(rows))
    w = 0.38

    ax.bar(x - w / 2, [v[0] for _, v in rows], w, label="Trained (7 lang.)",
           color=C_TRAIN, edgecolor="black", linewidth=0.4, zorder=3)
    ax.bar(x + w / 2, [v[1] for _, v in rows], w, label="Held-out (10 lang.)",
           color=C_OOD, edgecolor="black", linewidth=0.4, zorder=3)
    # Base is a bar here, so value labels replace the reference lines.
    for xi, (_, v) in zip(x, rows):
        ax.text(xi - w / 2, v[0] + 0.5, f"{v[0]:.1f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, v[1] + 0.5, f"{v[1]:.1f}", ha="center", fontsize=8)

    ax.set_ylabel("KLAR accuracy (%)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in rows], fontsize=10,
                       rotation=20 if len(rows) > 4 else 0,
                       ha="right" if len(rows) > 4 else "center")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=2, fontsize=8.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.20), handlelength=1.4, columnspacing=2.0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(a.out.rsplit(".", 1)[0] + "." + ext, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {a.out.rsplit('.',1)[0]}.pdf / .png")
    print(f"  {'model':<12}{'Train':>8}{'OOD':>8}   (vs base {base[0]:.1f}/{base[1]:.1f})")
    for n, v in rows:
        print(f"  {n:<12}{v[0]:>8.1f}{v[1]:>8.1f}   {v[0]-base[0]:>+6.1f}/{v[1]-base[1]:>+6.1f}")


if __name__ == "__main__":
    main()

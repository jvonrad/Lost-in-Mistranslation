"""LAHIS analysis, ported out of LAHIS_pipeline_olmo2.ipynb into a plain module.

The notebook was Colab-only: it mounted Drive, `git clone`d the
`eleftheria-lahis-analysis` branch, `os.chdir`'d into `src/`, and depended on
notebook globals defined several cells earlier. None of that runs headless, and
the figures it produced could not be regenerated from a fresh checkout. This
module is the pure-analysis half of that pipeline -- everything downstream of
the head-importance tensors, which is all the paper figures need.

The tensors themselves (`results/olmo2/olmo2_{lang}.pth`,
`results/olmo2_finetuned/olmo2_ft_{lang}.pth`, one [32 x 32] matrix per
language) live on the `eleftheria-lahis-analysis` branch and were vendored into
this checkout; recomputing them needs the model and TED corpus and is the one
step still in `src/attn_matrix_ted.py`.

Metric definitions are copied verbatim from the notebook so the ported figures
are the same numbers:

  top-k heads      the n = round(L*H*topk_p) largest entries of the [L, H]
                   importance matrix, flattened -- n = 20 at L=H=32, p=0.02
  layer profile    % of a language's top-k heads that sit in each layer
                   (`plot_top_head_distribution`, notebook cell via heatmap_viz)
  pairwise overlap |top-k(l1) & top-k(l2)| / n * 100 (notebook cell 24)

Usage:
  python mechanistic-interpretability/lahis_analysis.py --list
  python mechanistic-interpretability/lahis_analysis.py --model olmo2_ft
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence

import numpy as np
import torch

# Paper language order (not alphabetical): high-resource first, then the three
# low-resource ones, matching the High/Low split used in the results tables.
LANGUAGES: List[str] = ["en", "de", "pt", "ar", "es", "ru", "fr", "ja", "zh",
                        "id", "bn", "sw"]

# model key -> (results dir, filename prefix, human label)
MODELS = {
    "olmo2":    ("results/olmo2", "olmo2_", "OLMo-2-7B (base)"),
    "olmo2_ft": ("results/olmo2_finetuned", "olmo2_ft_", "OLMo-2-7B (SFT)"),
}

TOPK_P = 0.02


def load_matrix(model: str, lang: str, repo_root: str = ".") -> torch.Tensor:
    d, prefix, _ = MODELS[model]
    p = os.path.join(repo_root, d, f"{prefix}{lang}.pth")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return torch.load(p, map_location="cpu").float()


def load_all(model: str, languages: Sequence[str] = LANGUAGES,
             repo_root: str = ".") -> Dict[str, torch.Tensor]:
    out = {}
    for l in languages:
        try:
            out[l] = load_matrix(model, l, repo_root)
        except FileNotFoundError:
            pass
    return out


def n_top(matrix: torch.Tensor, topk_p: float = TOPK_P) -> int:
    L, H = matrix.shape
    return max(1, int(L * H * topk_p))


def top_head_indices(matrix: torch.Tensor, topk_p: float = TOPK_P) -> set:
    """Flat indices of the top-k% heads. Flat index = layer * num_heads + head."""
    _, idx = torch.topk(matrix.view(-1), k=n_top(matrix, topk_p))
    return set(idx.tolist())


def layer_profile(matrix: torch.Tensor, topk_p: float = TOPK_P) -> np.ndarray:
    """% of this language's top-k heads falling in each layer. Sums to 100."""
    L, H = matrix.shape
    counts = torch.zeros(L)
    for flat in top_head_indices(matrix, topk_p):
        counts[flat // H] += 1
    return (counts / counts.sum() * 100).numpy()


def overlap_matrix(mats: Dict[str, torch.Tensor], topk_p: float = TOPK_P):
    """Pairwise % of shared top-k heads. Returns (matrix, langs) in dict order."""
    langs = list(mats.keys())
    heads = {l: top_head_indices(m, topk_p) for l, m in mats.items()}
    n = n_top(next(iter(mats.values())), topk_p)
    out = np.zeros((len(langs), len(langs)))
    for i, a in enumerate(langs):
        for j, b in enumerate(langs):
            out[i, j] = len(heads[a] & heads[b]) / n * 100
    return out, langs


def _main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="olmo2", choices=sorted(MODELS))
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--list", action="store_true", help="print available tensors")
    a = ap.parse_args()

    if a.list:
        for m in sorted(MODELS):
            got = load_all(m, repo_root=a.repo_root)
            print(f"{m:<10} {len(got):>2}/{len(LANGUAGES)} languages: {' '.join(got)}")
        return

    mats = load_all(a.model, repo_root=a.repo_root)
    print(f"{a.model}: {len(mats)} languages, n_top = {n_top(next(iter(mats.values())))}")
    mat, langs = overlap_matrix(mats)
    off = mat[~np.eye(len(langs), dtype=bool)]
    print(f"mean off-diagonal overlap: {off.mean():.1f}%  (min {off.min():.0f}, max {off.max():.0f})")


if __name__ == "__main__":
    _main()

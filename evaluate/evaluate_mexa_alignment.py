#!/usr/bin/env python
"""MEXA-style cross-lingual representation alignment for the post-trained models.

Third view on the same question, alongside behaviour (PolyFact/KLAR) and neurons
(LAPE): does a post-training method change where the model puts non-English
sentences in representation space?

Method (MEXA): embed FLORES+ parallel sentences by mean-pooling each layer's
hidden states, L2-normalise, then measure how well English retrieves its
translations in language X and vice versa. Parallel sentences are the ground
truth, so retrieval accuracy is a direct read on whether translations land near
each other.

The METRIC functions are imported from XScript-Pretraining's
`xscript.eval.alignment` rather than reimplemented, so the definitions of
retrieval / d' / CKA are byte-identical to that project's. Only the plumbing is
new here: HF tokenizer + HF model with `output_hidden_states=True`, instead of
xscript's own `Tok` and `model.layer_reps`, and 12 languages instead of the 5 in
xscript's LANGS.

RAW, NOT CENTERED, by default. xscript's module defaults to reporting both and
warns that a per-language centroid ("language identity" direction) dominates
cross-lingual spaces, so raw numbers can look flat. Here raw is exactly what we
want: the LAPE result says consistency methods shift neuron specialisation
toward English, and the prediction that follows is that non-English sentences
get mapped closer to the English manifold -- i.e. a change in the centroid
itself. Centering would subtract the very effect under test. `--centered` adds
the centered variant alongside, as the control that says how much of any shift
is centroid movement rather than finer-grained alignment.

Usage:
  python evaluate/evaluate_mexa_alignment.py --model <path> --tag <name>
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

import numpy as np
import torch

XSCRIPT = os.path.expanduser("~/XScript-Pretraining/src")
if os.path.isdir(XSCRIPT):
    sys.path.insert(0, XSCRIPT)
from xscript.eval.alignment import _center, _retrieval, _retrieval_sim  # noqa: E402

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
LOW = {"id", "bn", "sw"}
FLORES = {"en": "eng_Latn", "de": "deu_Latn", "id": "ind_Latn", "pt": "por_Latn",
          "ar": "arb_Arab", "bn": "ben_Beng", "sw": "swh_Latn", "es": "spa_Latn",
          "ru": "rus_Cyrl", "fr": "fra_Latn", "ja": "jpn_Jpan",
          # FLORES+ renamed Mandarin: the ISO-639-3 macrolanguage code zho_Hans
          # does not exist as a config, only the individual code cmn_Hans.
          "zh": "cmn_Hans"}


def load_parallel(langs, split, limit=None):
    """FLORES+ sentences aligned by id, so row i is the same sentence everywhere."""
    from datasets import load_dataset
    per = {}
    for l in langs:
        d = load_dataset("openlanguagedata/flores_plus", FLORES[l], split=split)
        per[l] = {int(r["id"]): r["text"] for r in d}
    common = sorted(set.intersection(*(set(v) for v in per.values())))
    if limit:
        common = common[:limit]
    return {l: [per[l][i] for i in common] for l in langs}, len(common)


def embed(model, tok, sentences, device, batch=16, max_tokens=128):
    """(n_layers+1, N, dim) L2-normalised mean-pooled hidden states."""
    out = None
    for s0 in range(0, len(sentences), batch):
        chunk = sentences[s0:s0 + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_tokens).to(device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"][None, :, :, None].float()
        reps = torch.stack(hs).float()                       # (L, B, T, d)
        pooled = (reps * m).sum(2) / m.sum(2).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy()
        if out is None:
            out = np.zeros((pooled.shape[0], len(sentences), pooled.shape[-1]),
                           dtype=np.float32)
        out[:, s0:s0 + len(chunk)] = pooled
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--centered", action="store_true",
                    help="also report the centroid-removed variant as a control")
    ap.add_argument("--out_root", default="results/mexa")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    dev = next(model.parameters()).device

    par, n = load_parallel(LANGS, a.split, a.limit)
    print(f"{a.tag}: {n} parallel sentences x {len(LANGS)} languages", flush=True)

    emb = {}
    for l in LANGS:
        emb[l] = embed(model, tok, par[l], dev, a.batch, a.max_tokens)
        print(f"  [{l}] embedded", flush=True)

    n_layers = emb["en"].shape[0]
    variants = ["raw"] + (["centered"] if a.centered else [])
    mid_L = n_layers // 2

    # ALL 66 unordered pairs, not only EN<->X. The two sub-populations answer
    # different questions and the LAPE result predicts they diverge:
    #   EN<->X   how close non-English sits to English   (English-centricity)
    #   X<->Y    how close non-English languages sit to
    #            EACH OTHER, English excluded            (mutual alignment)
    # A method that centralises on English raises the first without raising the
    # second; a method that builds a genuinely shared space raises both. Summing
    # only over EN<->X, as the first version of this script did, cannot tell
    # those apart.
    from itertools import combinations
    pairs = list(combinations(LANGS, 2))
    res = {v: {"pairs": {}, "per_layer": {}} for v in variants}
    for v in variants:
        E_ = {l: ([_center(emb[l][L]) for L in range(n_layers)] if v == "centered"
                  else [emb[l][L] for L in range(n_layers)]) for l in LANGS}
        # per-layer curves, averaged over each pair population
        # The per-layer curves need only top-1, so go through _retrieval_sim on
        # the (N,N) similarity matrix and SKIP _cka. CKA forms 4096x4096
        # products and dominates the cost: with it, 4,488 calls per model take
        # ~108 min and overrun the walltime; without it the same sweep is ~5x
        # cheaper. Full metrics (CKA, d', margin) are still computed for every
        # pair at the mid layer below, which is what the table reports.
        curves = {"en_x": [], "x_y": []}
        for L in range(n_layers):
            enx, xy = [], []
            for x, b in pairs:
                t1 = _retrieval_sim(E_[x][L] @ E_[b][L].T)["top1_a2b"]
                (enx if x == "en" else xy).append(t1)
            curves["en_x"].append(float(np.mean(enx)))
            curves["x_y"].append(float(np.mean(xy)))
        res[v]["per_layer"] = curves
        for x, b in pairs:                       # full metrics at the mid layer
            res[v]["pairs"][f"{x}-{b}"] = _retrieval(E_[x][mid_L], E_[b][mid_L])

    def summary(v):
        P = res[v]["pairs"]
        enx = [P[k]["top1_a2b"] for k in P if k.startswith("en-")]
        xy = [P[k]["top1_a2b"] for k in P if not k.startswith("en-")]
        lo = [P[k]["top1_a2b"] for k in P
              if any(p in LOW for p in k.split("-"))]
        return (float(np.mean(enx + xy)), float(np.mean(enx)),
                float(np.mean(xy)), float(np.mean(lo)))

    out = {"config": {"model": a.model, "tag": a.tag, "split": a.split,
                      "n_sentences": n, "langs": LANGS, "n_layers": n_layers,
                      "max_tokens": a.max_tokens},
           "alignment": res, "mid_layer": mid_L,
           "summary": {v: dict(zip(("all_pairs", "en_x", "x_y", "pairs_with_low"),
                                   summary(v))) for v in variants}}
    os.makedirs(a.out_root, exist_ok=True)
    p = os.path.join(a.out_root, f"{a.tag}_mexa.json")
    json.dump(out, io.open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for v in variants:
        allp, enx, xy, lo = summary(v)
        print(f"\n{v:9s} mid-layer top1  all-pairs {allp:.3f}  EN-X {enx:.3f}  "
              f"X-Y {xy:.3f}  pairs-with-low-res {lo:.3f}  (EN-X minus X-Y: {enx-xy:+.3f})")
    print("\nEN-X per language (raw, mid layer): " +
          "  ".join(f"{l} {res['raw']['pairs']['en-'+l]['top1_a2b']:.2f}"
                    for l in LANGS if l != "en"))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()

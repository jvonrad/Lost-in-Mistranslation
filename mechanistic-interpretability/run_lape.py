#!/usr/bin/env python
"""LAPE: language-specialised MLP neurons, headless.

Ported from LAPE_neuron_counting.ipynb (Colab-only: Drive mount, hardcoded
paths, notebook globals). Method unchanged:

  1. tokenise up to --max_tokens of TED text per language
  2. forward-hook every layer's mlp.gate_proj and count, per neuron, how many
     tokens give silu(gate) > 0
  3. normalise counts by tokens seen -> per-language activation probability
  4. entropy of that distribution across languages; LOW entropy = specialised
  5. take the --top_frac lowest-entropy neurons and assign each to the language
     that activates it most

Token files are shared across models (they depend only on the tokenizer), so
same-family models reuse them.

Outputs results/lape/<tag>/: activations.<lang>.pt, lape_results.json
(per-layer language -> neuron indices, matching the committed
results/lape/*.json shape), and meta.json.

Usage:
  python mechanistic-interpretability/run_lape.py --model <id> --tag <name>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
DEFAULT_DATA_DIR = os.path.join(HERE, "data", "ted")


def tokenize_lang(tok, data_dir, lang, max_tokens, cache_dir):
    """Flat token stream for one language, cached per tokenizer family."""
    import torch
    p = os.path.join(cache_dir, f"tokens.{lang}.pt")
    if os.path.exists(p):
        return torch.load(p)
    src = os.path.join(data_dir, f"ted_{lang}.json")
    if not os.path.exists(src):
        print(f"  [{lang}] no TED file at {src} -- skipping", flush=True)
        return None
    with open(src, encoding="utf-8") as f:
        texts = [d["text"] for d in json.load(f)]
    ids = []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False).input_ids)
        if len(ids) >= max_tokens:
            break
    ids = torch.tensor(ids[:max_tokens], dtype=torch.long)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(ids, p)
    print(f"  [{lang}] {len(ids):,} tokens -> {p}", flush=True)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--langs", nargs="+", default=LANGS)
    ap.add_argument("--max_tokens", type=int, default=2_000_000,
                    help="per language; the notebook's MAX_TOKENS_PER_LANG")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch_seqs", type=int, default=8,
                    help="sequences per forward; the notebook did one at a time")
    ap.add_argument("--top_frac", type=float, default=0.01)
    ap.add_argument("--token_cache", default=None,
                    help="defaults to results/lape/_tokens_<family>")
    ap.add_argument("--out_root", default="results/lape")
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = os.path.join(a.out_root, a.tag)
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    fam = "olmo" if "olmo" in a.model.lower() else "qwen"
    cache = a.token_cache or os.path.join(a.out_root, f"_tokens_{fam}")

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    nl, inter = model.config.num_hidden_layers, model.config.intermediate_size
    print(f"{a.tag}: {nl} layers x {inter} neurons", flush=True)

    counts, ntoks, found = [], [], []
    for lang in a.langs:
        ids = tokenize_lang(tok, a.data_dir, lang, a.max_tokens, cache)
        if ids is None:
            continue
        act_p = os.path.join(out_dir, f"activations.{lang}.pt")
        if os.path.exists(act_p):
            d = torch.load(act_p)
            counts.append(d["over_zero"]); ntoks.append(d["n"]); found.append(lang)
            print(f"  [{lang}] cached", flush=True)
            continue

        over = torch.zeros(nl, inter, dtype=torch.int32, device="cuda")
        hooks = []
        for li in range(nl):
            def mk(i):
                def hook(mod, inp, out):
                    g = out[0] if isinstance(out, tuple) else out
                    over[i] += (F.silu(g.float()) > 0).sum(dim=(0, 1)).to(torch.int32)
                return hook
            hooks.append(model.model.layers[li].mlp.gate_proj.register_forward_hook(mk(li)))
        try:
            n = (ids.size(0) // a.seq_len) * a.seq_len
            seqs = ids[:n].reshape(-1, a.seq_len)
            with torch.no_grad():
                for i in range(0, seqs.size(0), a.batch_seqs):
                    model(seqs[i:i + a.batch_seqs].to("cuda"))
            torch.save({"n": n, "over_zero": over.cpu()}, act_p)
            counts.append(over.cpu()); ntoks.append(n); found.append(lang)
            print(f"  [{lang}] {n:,} tokens done", flush=True)
        finally:
            for h in hooks:
                h.remove()
            torch.cuda.empty_cache()

    if not found:
        raise SystemExit("no activation data collected")

    stack = torch.stack(counts, dim=-1).double()          # (layers, neurons, langs)
    probs = stack / torch.tensor(ntoks).double()
    normed = probs / (probs.sum(-1, keepdim=True) + 1e-9)
    entropy = -(normed * torch.log(normed + 1e-9)).sum(-1)
    # Neurons that never fire anywhere are not "specialised", they are dead.
    entropy[stack.sum(-1) == 0] = float("inf")

    k = max(1, int(nl * inter * a.top_frac))
    flat = entropy.flatten()
    idx = torch.topk(flat, k, largest=False).indices
    per_layer = {}
    for f in idx.tolist():
        l, nrn = divmod(f, inter)
        lang = found[int(normed[l, nrn].argmax())]
        per_layer.setdefault(str(l + 1), {}).setdefault(lang, []).append(nrn)

    json.dump(per_layer, open(os.path.join(out_dir, "lape_results.json"), "w"), indent=1)
    json.dump({"model": a.model, "langs": found, "max_tokens": a.max_tokens,
               "seq_len": a.seq_len, "top_frac": a.top_frac, "n_selected": k,
               "num_layers": nl, "intermediate_size": inter,
               "tokens_per_lang": dict(zip(found, ntoks))},
              open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    tot = {l: sum(len(v.get(l, [])) for v in per_layer.values()) for l in found}
    print(f"\n{k} specialised neurons ({a.top_frac:.0%} of {nl*inter:,})")
    print("  " + "  ".join(f"{l}:{tot[l]}" for l in found))
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()

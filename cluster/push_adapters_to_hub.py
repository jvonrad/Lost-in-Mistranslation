#!/usr/bin/env python
"""Publish the trained LoRA adapters to the Hugging Face hub.

ADAPTERS, not merged weights: every adapter_config.json already records the
correct `base_model_name_or_path` (including the CPT variants, which point at
jvonrad/olmo-2-7b-finetranslations and jvonrad/Qwen-2.5-7B-TED rather than the
public base), so a 0.15-1.2 GB upload is reproducible where a merged copy would
be 13-14 GB. Total here is ~8 GB instead of ~150 GB.

Each repo gets a generated model card carrying the evaluation numbers from
results/, so the published artefact and the paper's tables cannot drift apart.

Usage:
  python cluster/push_adapters_to_hub.py --dry-run     # print plan, touch nothing
  python cluster/push_adapters_to_hub.py               # create + upload
  python cluster/push_adapters_to_hub.py --only olmo-dco-10k
"""
from __future__ import annotations

import argparse
import io
import json
import os

PROJ = "/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation"
M = f"{PROJ}/models"

KLAR_SEEN = ["en", "es", "fr", "ru", "zh", "ja", "ar"]
KLAR_OOD = ["ca", "el", "fa", "he", "hu", "ko", "nl", "tr", "uk", "vi"]

# result_tag | repo name | local dir | one-line method description
ENTRIES = [
    ("olmo-sft-10k", "OLMo-2-7B-SFT-10k", f"{M}/olmo-sft-n10000",
     "Supervised fine-tuning on 10,000 PolyFact-Clean facts x 12 languages, "
     "pure cross-entropy (`--consistency_weight 0.0`)."),
    ("olmo-cpt-sft-10k", "OLMo-2-7B-CPT-SFT-10k", f"{M}/olmo-cpt-sft-n10000",
     "The same SFT recipe, applied on top of the translation-CPT checkpoint."),
    ("olmo-dco-10k", "OLMo-2-7B-DCO-10k", f"{M}/olmo-dco-n10000-r128",
     "DCO (Liu et al., ICML 2026): label-free cross-lingual consistency "
     "preference optimisation on 10,000 facts."),
    ("olmo-cpt-dco-10k", "OLMo-2-7B-CPT-DCO-10k", f"{M}/olmo-cpt-dco-n10000-r128",
     "The same DCO recipe, applied on top of the translation-CPT checkpoint."),
    ("olmo-cmalign-10k", "OLMo-2-7B-CM-Align-10k", f"{M}/olmo-cmalign-n10000",
     "CM-Align (Zhang et al., EMNLP 2025 Findings): English-pivot self-supervised "
     "DPO on 10,000 facts."),
    ("olmo-ladder10k-s6500", "OLMo-2-7B-GRPO-ladder-s6500",
     f"{M}/final/_preserved/olmo-ladder10k-s6500",
     "Consistency-driven GRPO with a laddered all-correct bonus "
     "(`--bonus_shape ladder`), checkpoint 6,500 of a 10,000-step run."),
    ("qwen-sft-10k", "Qwen-2.5-7B-SFT-10k", f"{M}/qwen-sft-n10000",
     "Supervised fine-tuning on 10,000 PolyFact-Clean facts x 12 languages, "
     "pure cross-entropy (`--consistency_weight 0.0`)."),
    ("qwen-cpt-sft-10k", "Qwen-2.5-7B-CPT-SFT-10k", f"{M}/qwen-cpt-sft-n10000",
     "The same SFT recipe, applied on top of the TED translation-CPT checkpoint."),
    ("qwen-dco-10k", "Qwen-2.5-7B-DCO-10k", f"{M}/qwen-dco-n10000-r128",
     "DCO (Liu et al., ICML 2026): label-free cross-lingual consistency "
     "preference optimisation on 10,000 facts."),
    ("qwen-cpt-dco-10k", "Qwen-2.5-7B-CPT-DCO-10k", f"{M}/qwen-cpt-dco-n10000-r128",
     "The same DCO recipe, applied on top of the TED translation-CPT checkpoint."),
    ("qwen-cmalign-10k", "Qwen-2.5-7B-CM-Align-10k", f"{M}/qwen-cmalign-n10000",
     "CM-Align (Zhang et al., EMNLP 2025 Findings): English-pivot self-supervised "
     "DPO on 10,000 facts."),
]

UPLOAD = ["adapter_config.json", "adapter_model.safetensors", "tokenizer.json",
          "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"]

# Repos that exist already and only need their visibility changed.
UNPRIVATE = ["jvonrad/Qwen-2.5-7B-grpo-clip5-1500"]

# Which evaluated tag corresponds to each adapter's OWN base. The CPT variants
# sit on the CPT checkpoint, so that -- not the family base -- is the reference
# a delta should be read against.
BASE_TAG = {"allenai/OLMo-2-1124-7B": "olmo-base",
            "Qwen/Qwen2.5-7B": "qwen-base",
            "jvonrad/olmo-2-7b-finetranslations": "olmo-finetranslations",
            "jvonrad/Qwen-2.5-7B-TED": "Qwen-2.5-7B-TED"}


def mcq(tag, bench):
    p = f"results/{tag}_{bench}_consistency.json"
    if not os.path.exists(p):
        return None
    return json.load(io.open(p, encoding="utf-8"))


def klar(tag):
    p = f"results/klar/{tag}_klar_alllangs.json"
    if not os.path.exists(p):
        return None
    d = json.load(io.open(p, encoding="utf-8"))["per_lang"]
    return {k: 100 * v[0] / v[1] for k, v in d.items()}


def metrics(tag):
    """The six headline numbers, or None per cell where no eval was run."""
    out = {}
    pf = mcq(tag, "polyfact_clean")
    if pf:
        acc = pf["per_language_accuracy"]
        out["pf_acc"] = 100 * sum(v["accuracy"] for v in acc.values()) / len(acc)
        out["pf_tc"] = 100 * pf["total_consistency"]["all_langs_correct_fraction"]
        out["pf_rankc"] = 100 * pf["rankc"]["average"]
    for key, bench in (("bm", "bmlama53"), ("gm", "gmmlu_lite")):
        d = mcq(tag, bench)
        if d:
            a = d["per_language_accuracy"]
            out[key] = 100 * sum(v["accuracy"] for v in a.values()) / len(a)
    k = klar(tag)
    if k:
        out["kl_seen"] = sum(k[l] for l in KLAR_SEEN) / len(KLAR_SEEN)
        out["kl_ood"] = sum(k[l] for l in KLAR_OOD) / len(KLAR_OOD)
    return out


def card(repo, tag, desc, base, cfg, base_tag):
    m, b = metrics(tag), metrics(base_tag)
    f = lambda d, k: f"{d[k]:.2f}" if k in d else "--"

    def row(name, d):
        return ("| " + name + " | " + " | ".join(
            f(d, k) for k in ("pf_acc", "pf_tc", "pf_rankc", "bm", "gm",
                              "kl_seen", "kl_ood")) + " |")

    return f"""---
base_model: {base}
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
- lora
- peft
- multilingual
- cross-lingual-consistency
- base_model:adapter:{base}
datasets:
- jvonrad/PolyFact-Clean
language: [en, de, es, fr, pt, id, ru, zh, ar, ja, sw, bn]
---

# {repo}

{desc}

A LoRA adapter (r={cfg['r']}, alpha={cfg['lora_alpha']}) over
[`{base}`](https://huggingface.co/{base}), trained for the paper
*Improving Cross-Lingual Factual Recall via Consistency-Driven Reinforcement
Learning*. It is one arm of a controlled comparison in which SFT, DCO, CM-Align
and GRPO all see the **same 10,000 facts** from
[`jvonrad/PolyFact-Clean`](https://huggingface.co/datasets/jvonrad/PolyFact-Clean)
across the same 12 languages, so the methods differ only in objective.

## Evaluation

Accuracy (%) unless noted. PolyFact-Clean is the 2,039-fact curated test split
with byte-normalised log-likelihood scoring; TotCons is the fraction of facts
answered correctly in **all 12** languages; RankC is RankC@4 (floor 9.02,
chance 37.68). KLAR is free-form generation over 17 languages, split into the
7 seen in training and the 10 held out.

| Model | PolyFact | TotCons | RankC | BMLAMA-53 | G-MMLU-Lite | KLAR seen | KLAR held-out |
|---|---|---|---|---|---|---|---|
{row(f"Base (`{base}`)", b)}
{row("**This model**", m)}

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base}", dtype="bfloat16",
                                            device_map="auto")
model = PeftModel.from_pretrained(base, "jvonrad/{repo}")
tok = AutoTokenizer.from_pretrained("jvonrad/{repo}")
```

Evaluation used the closed-book prompt `Question: {{q}}\\nAnswer:` with the
options hidden, matching `evaluate/evaluate_crosslingual_consistency.py`.

## Citation

```bibtex
@misc{{polyfact2026,
  title  = {{Improving Cross-Lingual Factual Recall via Consistency-Driven
            Reinforcement Learning}},
  author = {{von Rad, Jonathan}},
  year   = {{2026}},
  eprint = {{2606.06586}},
  archivePrefix = {{arXiv}}
}}
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()["name"]

    plan = [e for e in ENTRIES if a.only in (None, e[0])]
    for tag, repo, path, desc in plan:
        cfg = json.load(io.open(f"{path}/adapter_config.json", encoding="utf-8"))
        base = cfg["base_model_name_or_path"]
        base_tag = BASE_TAG[base]
        files = [f for f in UPLOAD if os.path.exists(f"{path}/{f}")]
        gb = sum(os.path.getsize(f"{path}/{f}") for f in files) / 2**30
        rid = f"{who}/{repo}"
        print(f"\n{rid}\n  from {path}\n  base {base}  r={cfg['r']}  "
              f"{gb:.2f} GB  files={files}")
        if a.dry_run:
            continue
        api.create_repo(rid, repo_type="model", private=a.private, exist_ok=True)
        io.open(f"{path}/README.md", "w", encoding="utf-8").write(
            card(repo, tag, desc, base, cfg, base_tag))
        api.upload_folder(repo_id=rid, folder_path=path,
                          allow_patterns=files + ["README.md"],
                          commit_message="Add LoRA adapter and model card")
        print(f"  -> https://huggingface.co/{rid}")

    for rid in UNPRIVATE:
        if a.only:
            break
        print(f"\n{rid}: make public")
        if not a.dry_run:
            api.update_repo_settings(repo_id=rid, private=False)
            print(f"  -> https://huggingface.co/{rid}")


if __name__ == "__main__":
    main()

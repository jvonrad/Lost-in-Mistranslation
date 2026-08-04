#!/usr/bin/env python
"""Compute LAHIS attention-head importance for a model, headless.

Replaces the Colab-only LAHIS_pipeline_olmo2.ipynb: no Drive mount, no clone,
no chdir, no notebook globals. Data prep and the per-language sweep are one
command so a batch job can run the whole thing.

Writes results/lahis/<tag>/lahis_<lang>.pth -- one [num_layers x num_heads]
tensor per language, the same artefact the notebook produced under
results/olmo2/.

The TED corpus (TED2025/multi_way.jsonl, from Zhu et al., EMNLP 2025) is the one
the original analysis used, so tensors computed here stay comparable to the
committed base/SFT ones -- provided --data_num and --max_length match. They are
recorded in each output directory's meta.json precisely because the committed
tensors did NOT record theirs.

Usage:
  python mechanistic-interpretability/run_lahis.py --prepare-data     # once
  python mechanistic-interpretability/run_lahis.py --model <id> --tag <name>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
DEFAULT_TED = os.environ.get(
    "TED_JSONL",
    "/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/datasets/TED2025/TED2025/multi_way.jsonl")
DEFAULT_DATA_DIR = os.path.join(HERE, "data", "ted")


def prepare_data(ted_jsonl, data_dir, langs, join_n=5):
    """Build ted_<lang>.json once; both LAHIS and LAPE read these."""
    import ted_loader as TED
    missing = [l for l in langs
               if not os.path.exists(os.path.join(data_dir, f"ted_{l}.json"))]
    if not missing:
        print(f"all {len(langs)} language files already present in {data_dir}")
        return
    print(f"building {missing} from {ted_jsonl} ...", flush=True)
    records = TED.load_ted_jsonl(ted_jsonl)
    print(f"  {len(records):,} TED records", flush=True)
    streams = TED.build_monolingual_streams(records, missing, join_n_sentences=join_n)
    TED.save_monolingual_json(streams, data_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-data", action="store_true")
    ap.add_argument("--ted_jsonl", default=DEFAULT_TED)
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--model", default=None, help="HF id or local path")
    ap.add_argument("--tag", default=None, help="output subdirectory name")
    ap.add_argument("--arch", default="olmo2", help="model_handler architecture key")
    ap.add_argument("--langs", nargs="+", default=LANGS)
    ap.add_argument("--data_num", type=int, default=500,
                    help="TED chunks per language; the notebook's DATA_NUM")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--out_root", default="results/lahis")
    a = ap.parse_args()

    os.makedirs(a.data_dir, exist_ok=True)
    if a.prepare_data:
        prepare_data(a.ted_jsonl, a.data_dir, a.langs)
        if not a.model:
            return
    if not a.model:
        ap.error("--model is required unless only --prepare-data is given")

    import torch
    import model_handler
    from attn_matrix_ted import get_attn_head_matrix_ted

    out_dir = os.path.join(a.out_root, a.tag or os.path.basename(a.model.rstrip("/")))
    os.makedirs(out_dir, exist_ok=True)

    # model_handler resolves "hf:<path>" to any HF id or local dir and applies the
    # OLMo-2 head-mask patch; the bare "olmo2" key hardcodes the base checkpoint.
    key = a.model if a.model in ("olmo2", "llama2") else f"hf:{a.model}"
    print(f"loading {key} ...", flush=True)
    model, tokenizer = model_handler.load_model(
        key, device="cuda", half_precision=True, local=False)
    model.eval()
    nl = model.config.num_hidden_layers
    nh = model.config.num_attention_heads
    print(f"  {nl} layers x {nh} heads", flush=True)

    for lang in a.langs:
        out = os.path.join(out_dir, f"lahis_{lang}.pth")
        if os.path.exists(out):
            print(f"[{lang}] cached", flush=True)
            continue
        m = get_attn_head_matrix_ted(model, tokenizer, lan=lang, model_name=a.arch,
                                     data_dir=a.data_dir, data_num=a.data_num,
                                     max_length=a.max_length)
        torch.save(m, out)
        print(f"[{lang}] saved -> {out}  range [{m.min():.3f}, {m.max():.3f}]", flush=True)

    json.dump({"model": a.model, "arch": a.arch, "langs": a.langs,
               "data_num": a.data_num, "max_length": a.max_length,
               "ted_jsonl": a.ted_jsonl, "num_layers": nl, "num_heads": nh},
              open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()

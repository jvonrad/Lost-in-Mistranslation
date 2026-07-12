#!/usr/bin/env python3
"""
KLAR-CLC evaluator — batched generation.

Loads a causal-LM, runs it over the KLAR-CLC factual-recall dataset with
batched greedy generation, and prints final per-language accuracy. Nothing
is written to disk.

BASELINE="allenai/OLMo-2-1124-7B"
ALIGNED_SFT="jonny-vr/olmo-2-7b-aligned-wikifact-sft"
ALIGNED_GRPO="jonny-vr/olmo-2-7b-finetranslation-wikifact-grpo-att-mlp-checkpoint"
ONLY_GRPO="jonny-vr/olmo-2-7b-wikifact-grpo"
ONLY_SFT="jonny-vr/olmo-2-7B.wikifact-sft-consistent"
ONLY_ALIGNED="jonny-vr/olmo-2-7b-finetranslations"

Usage
-----
    python evaluate/evaluate_klar.py \
        --klar-root /data/jonathan/Lost-in-Mistranslation/datasets/KLAR-CLC \
        --model allenai/OLMo-2-1124-7B \
        --batch-size 32
        
    python evaluate/evaluate_klar.py \
  --klar-root /data/jonathan/Lost-in-Mistranslation/datasets/KLAR-CLC \
  --model jvonrad/olmo-2-7b-wikifact-sft \
  --langs ar ca el en es fa fr he hu ja ko nl ru tr uk vi zh
"""



from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np


# Punctuation we strip before matching. Covers ASCII, CJK, and common
# typographic quotes / guillemets used across KLAR's 17 languages.
_PUNCT_RE = re.compile(
    r"[\s\.\,\!\?\:\;\"\'\(\)\[\]\{\}«»“”‘’、。，．！？；：]+"
)


# ──────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_KLAR_ROOT = "./KLAR-CLC"
DEFAULT_LANGS = ["es", "fr", "ru", "zh", "ja", "ar"]

ALL_RELATIONS = [
    "applies_to_jurisdiction", "capital", "capital_of", "continent",
    "country_of_citizenship", "developer", "field_of_work",
    "headquarters_location", "instrument", "language_of_work_or_name",
    "languages_spoken", "location_of_formation", "manufacturer",
    "native_language", "occupation", "official_language", "owned_by",
    "place_of_birth", "place_of_death", "religion",
]

N_SHOT = 3
RANDOM_SEED = 42
PROMPT_TEMPLATE_IDX = 1
MAX_NEW_TOKENS = 10
BATCH_SIZE = 32


# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def normalise(s: Any) -> str:
    if not s:
        return ""
    return unicodedata.normalize("NFC", str(s)).lower().strip()


def _clean(s: str) -> str:
    """NFC-normalise, lowercase, strip punctuation & collapse whitespace."""
    s = normalise(s)
    return _PUNCT_RE.sub(" ", s).strip()


def is_correct(pred: str, target: str, strict: bool = False) -> bool:
    """Union of the original-notebook matcher and punctuation-stripping.

    A prediction is correct if ANY of the following hold after lowercasing,
    NFC-normalising, and stripping punctuation:

        1. exact match
        2. target appears as a whole token anywhere in pred
           (e.g. "france paris" matches "paris")
        3. pred starts with target's characters
           (e.g. "parisian" matches "paris", "Римская" matches "Рим",
                 "北京市" matches "北京")

    `strict=True` disables rules 2 and 3 — only exact match counts.
    """
    p, t = _clean(pred), _clean(target)
    if not p or not t:
        return False
    if p == t:
        return True
    if strict:
        return False

    # Rule A: target is a whole token of pred (only meaningful for
    # space-delimited scripts; for CJK .split() returns one big token).
    if t in p.split():
        return True

    # Rule B: pred starts with target's characters (bare prefix).
    if p.startswith(t):
        return True

    return False


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def format_template(template: str, subject: str) -> str:
    """Fill <subject> and strip the trailing <mask> so the prompt is open-ended."""
    return (template
            .replace("<subject>", subject)
            .replace(" <mask>", "")
            .replace("<mask>", "")
            .rstrip())


# ──────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDING
# ──────────────────────────────────────────────────────────────────────────────

def build_samples(
    klar_root: Path,
    langs: list[str],
    relations: list[str],
    n_shot: int,
    template_idx: int,
    max_per_relation: int | None,
) -> list[dict]:
    out: list[dict] = []
    for lang in langs:
        for rel in relations:
            path = klar_root / lang / f"{rel}.json"
            if not path.exists():
                print(f"  [warn] missing {lang}/{rel}.json")
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            templates = data.get("prompt_templates", [])
            if not templates:
                print(f"  [warn] no templates in {lang}/{rel}.json")
                continue
            tmpl = templates[template_idx % len(templates)]

            samples = data.get("samples", [])
            if max_per_relation is not None:
                samples = samples[:max_per_relation]

            inputs = [format_template(tmpl, s["subject"]) for s in samples]

            for i, s in enumerate(samples):
                pool = [j for j in range(len(samples)) if j != i]
                demo_idxs = random.sample(pool, min(n_shot, len(pool)))
                prompt = "".join(f"{inputs[j]} {samples[j]['object']}\n" for j in demo_idxs)
                prompt += f"{inputs[i]} "

                out.append({
                    "language": lang,
                    "relation": rel,
                    "index":    s.get("index"),
                    "subject":  s.get("subject", ""),
                    "target":   s["object"],
                    "prompt":   prompt,
                })
    print(f"Built {len(out)} prompts across {len(langs)} langs × {len(relations)} relations.")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# DEVICE
# ──────────────────────────────────────────────────────────────────────────────

def get_device(requested: str):
    """auto -> cuda > xla(Neuron) > cpu. Returns (torch.device, kind_str)."""
    import torch
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return torch.device("cuda"), "cuda"
    if requested in ("xla", "auto"):
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device(), "xla"
        except Exception:
            if requested == "xla":
                raise
    return torch.device("cpu"), "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION — BATCHED GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def _greedy_xla(model, tokenizer, prompts, max_new_tokens, device, max_length):
    """Fixed-shape greedy decoding for Neuron/XLA.

    HF `model.generate` does not compile on Neuron: its cached-attention mask and
    stopping-criteria emit an int64 `dot` the compiler rejects (NCC_EVRF035). We
    therefore decode manually with plain `forward` passes (the same op set the
    repo's consistency eval already runs on Neuron). Every forward runs over one
    FIXED buffer of length `max_length + max_new_tokens`, so exactly one graph is
    compiled and reused for every step and every batch. Left padding + explicit
    position_ids (cumsum of the mask) keep RoPE positions correct.
    """
    import torch
    import torch_xla.core.xla_model as xm

    enc = tokenizer(
        prompts, return_tensors="pt",
        padding="max_length", truncation=True, max_length=max_length,
    )
    B = enc["input_ids"].shape[0]
    pad_id = tokenizer.pad_token_id
    ids = torch.cat(
        [enc["input_ids"], torch.full((B, max_new_tokens), pad_id, dtype=torch.long)], dim=1)
    am = torch.cat(
        [enc["attention_mask"], torch.zeros((B, max_new_tokens), dtype=torch.long)], dim=1)
    ids, am = ids.to(device), am.to(device)

    gen_cols = []
    with torch.no_grad():
        for t in range(max_new_tokens):
            # cumsum lowers to a matmul; keep it in fp32 so it is not an int64
            # `dot` (which the Neuron compiler rejects, NCC_EVRF035).
            pos_ids = (am.to(torch.float32).cumsum(-1) - 1).clamp(min=0).to(torch.long)
            out = model(input_ids=ids, attention_mask=am,
                        position_ids=pos_ids, use_cache=False)
            read = max_length - 1 + t          # logits here predict column max_length+t
            nxt = out.logits[:, read, :].argmax(-1)
            col = max_length + t
            ids[:, col] = nxt
            am[:, col] = 1
            gen_cols.append(nxt.unsqueeze(1))
            xm.mark_step()

    gen = torch.cat(gen_cols, dim=1).cpu()
    decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
    return [normalise(d.split("\n")[0]) for d in decoded]


def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int,
                   device=None, device_kind: str = "cuda",
                   max_length: int = 448) -> list[str]:
    """Greedy-generate for a batch of prompts. Returns normalised first-line strings."""
    import torch
    if device is None:
        device = model.device

    if device_kind == "xla":
        return _greedy_xla(model, tokenizer, prompts, max_new_tokens, device, max_length)

    enc = tokenizer(
        prompts, return_tensors="pt",
        padding=True, truncation=False,
    ).to(device)
    prompt_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        output = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Slice off the prompt portion (left-padding means the prompt ends at prompt_len-1
    # for every row, so the newly generated tokens start at `prompt_len`).
    new_tokens = output[:, prompt_len:]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [normalise(d.split("\n")[0]) for d in decoded]


def evaluate(
    samples: list[dict],
    model_id: str,
    langs: list[str],
    tokenizer_id: str | None,
    max_new_tokens: int,
    batch_size: int,
    strict: bool = False,
    device_str: str = "auto",
    max_length: int = 128,
    contaminated_keys: set | None = None,
) -> tuple[dict[str, tuple[int, int]], list[dict]]:
    """Return ({lang: (n_correct, n_total)}, per_sample_records)."""
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, device_kind = get_device(device_str)
    tokenizer_src = tokenizer_id or model_id
    print(f"\nModel:      {model_id}")
    print(f"Tokenizer:  {tokenizer_src}")
    print(f"Device:     {device_kind}")
    print(f"Batch size: {batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Critical for causal-LM batched generation: pad on the LEFT so every prompt
    # ends at the same position. Otherwise right-padded sequences start
    # generating from pad tokens and produce garbage.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16,
        device_map="auto" if device_kind == "cuda" else None,
    ).eval()
    if device_kind != "cuda":
        model = model.to(device)

    contaminated_keys = contaminated_keys or set()

    def is_contam(s):
        return (s["relation"], s["index"]) in contaminated_keys

    per_lang: dict[str, tuple[int, int]] = {}
    records: list[dict] = []
    for lang in langs:
        lang_data = [s for s in samples if s["language"] == lang]
        if not lang_data:
            per_lang[lang] = (0, 0)
            continue

        n_correct = 0
        pbar = tqdm(total=len(lang_data), desc=f"[{lang}]")
        for i in range(0, len(lang_data), batch_size):
            batch = lang_data[i:i + batch_size]
            prompts = [s["prompt"] for s in batch]
            # On XLA keep the batch dim fixed so the graph is reused; pad the
            # final short batch by repeating its first prompt, then discard.
            pad_n = 0
            if device_kind == "xla" and len(prompts) < batch_size:
                pad_n = batch_size - len(prompts)
                prompts = prompts + [prompts[0]] * pad_n
            preds = generate_batch(
                model, tokenizer, prompts, max_new_tokens,
                device=device, device_kind=device_kind, max_length=max_length,
            )
            if pad_n:
                preds = preds[:len(batch)]
            for s, pred in zip(batch, preds):
                correct = is_correct(pred, s["target"], strict=strict)
                if correct:
                    n_correct += 1
                records.append({
                    "language": lang, "relation": s["relation"],
                    "index": s["index"], "subject": s["subject"],
                    "target": s["target"], "pred": pred,
                    "correct": bool(correct), "contaminated": bool(is_contam(s)),
                })
            pbar.update(len(batch))
        pbar.close()
        per_lang[lang] = (n_correct, len(lang_data))

    return per_lang, records


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--klar-root", default=DEFAULT_KLAR_ROOT)
    p.add_argument("--langs", nargs="+", default=DEFAULT_LANGS)
    p.add_argument("--relations", nargs="+", default=ALL_RELATIONS)
    p.add_argument("--n-shot", type=int, default=N_SHOT)
    p.add_argument("--template-idx", type=int, default=PROMPT_TEMPLATE_IDX,
                   help="Which prompt_templates[i] to use (0–4).")
    p.add_argument("--max-per-relation", type=int, default=None,
                   help="Cap samples per relation (for quick tests).")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--model", required=True, help="HF model ID or local path.")
    p.add_argument("--tokenizer", default=None, help="Optional tokenizer path (default: same as --model).")
    p.add_argument("--strict", action="store_true",
                   help="Exact match only (after lowercasing + punctuation strip). "
                        "Default allows leading-token / CJK-prefix matching.")
    p.add_argument("--device", choices=["auto", "cuda", "xla", "cpu"], default="auto")
    p.add_argument("--max-length", type=int, default=448,
                   help="Fixed prompt length on XLA (avoids recompilation; "
                        "max KLAR prompt is 392 tokens).")
    p.add_argument("--contamination-labels", default=None,
                   help="JSON from data_analysis/contamination_analysis.py "
                        "(evaluate/alignments/klar_polyfact_contamination.json). "
                        "Splits results into contaminated vs clean subsets.")
    p.add_argument("--output-json", default=None,
                   help="Write per-sample predictions here for offline slicing.")
    args = p.parse_args()

    set_seed()

    # Contamination labels: set of (relation_name, index) that are memorizable
    # from PolyFact-train, plus the shared-relation universe (for the 'clean' set).
    contaminated_keys, shared_keys = set(), set()
    if args.contamination_labels:
        with open(args.contamination_labels, encoding="utf-8") as f:
            lab = json.load(f)
        contaminated_keys = {(r, i) for r, i in lab.get("contaminated_keys", [])}
        shared_keys = {(r, i) for r, i in lab.get("shared_relation_keys", [])}
        print(f"Loaded {len(contaminated_keys)} contaminated / "
              f"{len(shared_keys)} shared-relation KLAR facts.")

    samples = build_samples(
        klar_root=Path(args.klar_root),
        langs=args.langs,
        relations=args.relations,
        n_shot=args.n_shot,
        template_idx=args.template_idx,
        max_per_relation=args.max_per_relation,
    )

    per_lang, records = evaluate(
        samples=samples,
        model_id=args.model,
        langs=args.langs,
        tokenizer_id=args.tokenizer,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        strict=args.strict,
        device_str=args.device,
        max_length=args.max_length,
        contaminated_keys=contaminated_keys,
    )

    # Final report
    mode = "strict" if args.strict else "lenient"
    print("\n" + "=" * 40)
    print(f"RESULTS — {args.model}")
    print(f"match mode: {mode}")
    print("=" * 40)
    total_correct, total_n = 0, 0
    for lang in args.langs:
        n_correct, n = per_lang.get(lang, (0, 0))
        total_correct += n_correct
        total_n += n
        if n == 0:
            print(f"  {lang}:      n/a  (0 samples)")
        else:
            print(f"  {lang}:    {n_correct / n:.3f}   ({n_correct}/{n})")
    print("  " + "-" * 36)
    if total_n:
        print(f"  all:    {total_correct / total_n:.3f}   ({total_correct}/{total_n})")

    # Contamination breakdown (uses per-sample records; language-aggregated).
    subset_acc = None
    if args.contamination_labels:
        def acc(recs):
            c = sum(r["correct"] for r in recs)
            return (c, len(recs), c / len(recs) if recs else float("nan"))

        contam = [r for r in records if r["contaminated"]]
        shared = [r for r in records if (r["relation"], r["index"]) in shared_keys]
        clean_shared = [r for r in shared if not r["contaminated"]]
        clean_all = [r for r in records if not r["contaminated"]]
        subset_acc = {
            "full": acc(records),
            "contaminated_157": acc(contam),
            "clean_shared_1050": acc(clean_shared),
            "clean_all": acc(clean_all),
        }
        print("\n" + "=" * 40)
        print("CONTAMINATION BREAKDOWN (all langs pooled)")
        print("=" * 40)
        for name, (c, n, a) in subset_acc.items():
            print(f"  {name:20s}: {a:.4f}  ({c}/{n})")

    if args.output_json:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({
                "model": args.model,
                "langs": args.langs,
                "match_mode": mode,
                "per_lang": {k: list(v) for k, v in per_lang.items()},
                "subset_acc": subset_acc,
                "records": records,
            }, f, ensure_ascii=False)
        print(f"\nWrote per-sample predictions -> {args.output_json}")


if __name__ == "__main__":
    main()
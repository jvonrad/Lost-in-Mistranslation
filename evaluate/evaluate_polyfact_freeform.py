#!/usr/bin/env python
"""PolyFact-Clean scored as FREE-FORM GENERATION, using KLAR's exact matcher.

Every PolyFact number in the paper is log-likelihood scoring over four given
options: the model never generates, it ranks. That measures closed-book
*recognition*. KLAR measures generation, but on a different fact distribution,
so the two cannot be compared directly -- and the gap between them is exactly
where the methods separate (Qwen spans ~2pp on PolyFact MCQ and ~43pp on KLAR).

This script closes that gap by evaluating the SAME PolyFact-Clean test facts in
generation mode, so in-domain recognition and in-domain generation differ only
in the scoring protocol. It imports `normalise` and `is_correct` from
evaluate_klar.py rather than reimplementing them, so "same scoring as KLAR" is
enforced by construction, not by comment.

Two metrics matter here and neither is available from the MCQ evaluator:

  freeform accuracy   per language, greedy decode + KLAR string match
  freeform agreement  fraction of facts where every language names the SAME
                      ENTITY, split into correct / incorrect
  resolution rate     fraction of generations that name any of the four known
                      entities at all (the rest are off-vocabulary answers)

Agreement is measured at ENTITY level, not string level. The gold answer is a
different string in every language -- "Ivry-sur-Seine" / "塞纳河畔伊夫里" /
"イヴリー＝シュル＝セーヌ" -- so comparing generated strings across languages
detects nothing: a model that correctly says "paris"/"パリ"/"париж" everywhere
would score zero agreement. Each generation is therefore matched against ITS OWN
language's four option surface forms, and the matched option's Wikidata QID is
compared across languages. This is the same identity relation the MCQ evaluator
uses via `option_ids`, applied to generated text.

So measured, free-form agreement is stronger evidence than RankC@4: agreeing
among four *given* options has a chance floor of 0.09, whereas a model must
produce the right surface form unprompted in all twelve languages here.

Usage:
  python evaluate/evaluate_polyfact_freeform.py --model <path> \
      --output_json results/<tag>_polyfact_freeform.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_klar import is_correct, normalise  # noqa: E402

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
HIGH = ["en", "de", "pt", "ar", "es", "ru", "fr", "ja", "zh"]
LOW = ["id", "bn", "sw"]

# Same wrapper the MCQ evaluator uses, so the only difference between the two
# PolyFact numbers is generate-vs-score, not the prompt.
PROMPT = "Question: {q}\nAnswer:"

# N-SHOT. At 0-shot only 14-27% of generations named any of the four known
# entities: the models were never trained to answer this prompt without the
# option list, so the metric mostly measured format compliance rather than
# recall, and every consistency cell came back 0.00. KLAR -- where the methods
# do separate -- uses 3 shots. Exemplars are drawn from the TRAIN split in the
# SAME language, are fixed by seed, and are excluded from scoring, so no test
# fact is ever shown.
def build_shots(dataset_id, config, n_shot, seed=0):
    """lang -> prompt prefix of n_shot solved examples from the train split."""
    if n_shot <= 0:
        return {l: "" for l in LANGS}
    import random
    from datasets import load_dataset
    ds = load_dataset(dataset_id, config, split="train")
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), min(len(ds), n_shot * 20))
    per = {l: [] for l in LANGS}
    for i in idx:
        tr = ds[i].get("translations") or {}
        for l in LANGS:
            if len(per[l]) >= n_shot:
                continue
            it = tr.get(l)
            if not isinstance(it, dict):
                continue
            q, ai = it.get("question"), it.get("answer_index")
            opts = [it.get(f"option_{c}") for c in "abcd"]
            if q is None or ai is None or any(o is None for o in opts):
                continue
            per[l].append(PROMPT.format(q=q) + " " + str(opts[int(ai)]))
        if all(len(v) >= n_shot for v in per.values()):
            break
    return {l: ("\n\n".join(v) + "\n\n" if v else "") for l, v in per.items()}


def build_items(dataset_id, config, split, limit=None):
    from datasets import load_dataset
    ds = load_dataset(dataset_id, config, split=split)
    items = []
    for row in ds:
        tr = row.get("translations") or {}
        per = {}
        for lang in LANGS:
            it = tr.get(lang)
            if not isinstance(it, dict):
                continue
            q = it.get("question")
            ai = it.get("answer_index")
            opts = [it.get(f"option_{c}") for c in "abcd"]
            oids = it.get("option_ids")
            if q is None or ai is None or any(o is None for o in opts):
                continue
            per[lang] = {"q": q, "gold": opts[int(ai)], "options": opts,
                         "option_ids": list(oids) if isinstance(oids, list) else None}
        if len(per) == len(LANGS):
            items.append({"fact_id": row.get("fact_id"), "langs": per})
        if limit and len(items) >= limit:
            break
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--hf_dataset", default="jvonrad/PolyFact-Clean")
    ap.add_argument("--hf_config", default="parallel")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=768)
    ap.add_argument("--n_shot", type=int, default=0,
                    help="in-language exemplars from the TRAIN split prepended to "
                         "each prompt; 3 matches KLAR")
    ap.add_argument("--strict", action="store_true",
                    help="exact match only; default is KLAR's lenient matcher")
    ap.add_argument("--output_json", default=None)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()

    items = build_items(a.hf_dataset, a.hf_config, a.split, a.limit)
    shots = build_shots(a.hf_dataset, a.hf_config, a.n_shot)
    print(f"{len(items)} facts x {len(LANGS)} languages, n_shot={a.n_shot}", flush=True)

    # one flat work list, so batching is dense across facts and languages
    work = [(i, lang) for i, it in enumerate(items) for lang in LANGS]
    preds = {}
    for s in range(0, len(work), a.batch_size):
        chunk = work[s:s + a.batch_size]
        prompts = [shots[l] + PROMPT.format(q=items[i]["langs"][l]["q"])
                   for i, l in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=a.max_length).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for (i, l), txt in zip(chunk, tok.batch_decode(gen, skip_special_tokens=True)):
            # KLAR's extraction: first line, normalised.
            preds[(i, l)] = normalise(txt.split("\n")[0])
        if (s // a.batch_size) % 50 == 0:
            print(f"  {s + len(chunk)}/{len(work)}", flush=True)

    per_lang = {l: [0, 0] for l in LANGS}
    n_all_correct = n_agree_correct = n_agree_wrong = 0
    n_resolved = n_slots = 0
    records = {}
    for i, it in enumerate(items):
        ok, strings, ents = {}, {}, {}
        for l in LANGS:
            meta = it["langs"][l]
            p = preds[(i, l)]
            c = is_correct(p, meta["gold"], strict=a.strict)
            per_lang[l][0] += int(c)
            per_lang[l][1] += 1
            ok[l], strings[l] = c, p
            # Which of THIS language's four options did the generation name?
            # The QID makes the answer comparable across languages even though
            # the surface forms are unrelated strings.
            ent = None
            if meta["option_ids"]:
                for oi, opt in enumerate(meta["options"]):
                    if is_correct(p, opt, strict=a.strict):
                        ent = meta["option_ids"][oi]
                        break
            ents[l] = ent
            n_slots += 1
            n_resolved += int(ent is not None)
        all_ok = all(ok.values())
        n_all_correct += int(all_ok)
        vals = list(ents.values())
        if all(v is not None for v in vals) and len(set(vals)) == 1:
            if all_ok:
                n_agree_correct += 1
            else:
                n_agree_wrong += 1
        records[it["fact_id"]] = {l: {"pred": strings[l], "correct": ok[l],
                                      "entity": ents[l], "gold": it["langs"][l]["gold"]}
                                  for l in LANGS}

    n = len(items)
    acc = {l: 100 * c / t for l, (c, t) in per_lang.items()}
    mean = lambda L: sum(acc[l] for l in L) / len(L)
    res = {
        "config": {"model": a.model, "benchmark": "polyfact_freeform",
                   "dataset": a.hf_dataset, "split": a.split, "n_facts": n,
                   "langs": LANGS, "matcher": "strict" if a.strict else "klar_lenient",
                   "max_new_tokens": a.max_new_tokens},
        "per_language_accuracy": {l: {"accuracy": acc[l] / 100, "n": per_lang[l][1]}
                                  for l in LANGS},
        "high_resource_mean": mean(HIGH), "low_resource_mean": mean(LOW),
        "total_consistency": {"all_langs_correct_fraction": n_all_correct / n,
                              "n_facts_all_langs": n, "n_facts_total": n},
        "freeform_agreement": {"same_entity_and_correct": 100 * n_agree_correct / n,
                               "same_entity_and_wrong": 100 * n_agree_wrong / n,
                               "same_entity_any": 100 * (n_agree_correct + n_agree_wrong) / n,
                               "resolution_rate": 100 * n_resolved / n_slots},
        "predictions": records,
    }
    print(f"\nHigh {mean(HIGH):.2f}  Low {mean(LOW):.2f}  "
          f"TotC {100*n_all_correct/n:.2f}  "
          f"agree(correct/wrong) {100*n_agree_correct/n:.2f}/{100*n_agree_wrong/n:.2f}  "
          f"resolved {100*n_resolved/n_slots:.1f}%")
    print("per-language: " + "  ".join(f"{l} {acc[l]:.1f}" for l in LANGS))
    if a.output_json:
        os.makedirs(os.path.dirname(a.output_json), exist_ok=True)
        json.dump(res, io.open(a.output_json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"wrote {a.output_json}")


if __name__ == "__main__":
    main()

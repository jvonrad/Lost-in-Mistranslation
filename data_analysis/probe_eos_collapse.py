#!/usr/bin/env python
"""Why does a GRPO arm answer under its training prompt but emit NOTHING on KLAR?

qwen-nobonus-s6000 produces correct answers under the PolyFact training
scaffold (step-6000 rollouts are clean in all 12 languages, resolution_rate
1.000) yet returns an empty string for 70.7% of KLAR items, against 0.0% for
the base model. Its PolyFact/BMLAMA/Global-MMLU accuracy is near-normal, so the
model is not broken in general -- something is specific to free-form generation
off the training prompt.

HYPOTHESIS: the reward had a free "safe action". Empty output scored 0.0 while
non-empty-but-unparseable scored -0.5, so whenever the policy was uncertain,
silence strictly dominated guessing. Over 6,000 steps that is learnable as a
general fallback: stay fluent where confident, emit EOS where not. On the
training prompt the model is confident and answers; on an unseen prompt format
the fallback fires.

That predicts a specific, measurable thing: P(EOS as the FIRST generated token)
should be near zero for this model under the training prompt and large under the
KLAR prompt, while the base model stays near zero under both. This script
measures exactly that -- no generation, one forward pass per prompt, so the
number is not confounded by decoding settings.

Usage:
  python data_analysis/probe_eos_collapse.py --model <merged_dir> --label nobonus
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# The two prompt formats. TRAIN is the eval-style wrapper the GRPO trainer's own
# periodic eval uses; KLAR is the 3-shot free-form template evaluate_klar.py
# builds. Kept literal here so the probe cannot drift from what was measured.
TRAIN_PROMPTS = {
    "en": "Question: Which country is Paris the capital of?\nAnswer:",
    "es": "Pregunta: ¿De qué país es Madrid la capital?\nRespuesta:",
    "fr": "Question : De quel pays Paris est-elle la capitale ?\nRéponse :",
    "ja": "質問: 東京はどの国の首都ですか？\n答え:",
}
KLAR_PROMPTS = {
    "en": ("Berlin is located in the country of Germany\n"
           "Rome is located in the country of Italy\n"
           "Madrid is located in the country of Spain\n"
           "Paris is located in the country of"),
    "es": ("Berlín está ubicado en el país de Alemania\n"
           "Roma está ubicada en el país de Italia\n"
           "Madrid está ubicado en el país de España\n"
           "París está ubicado en el país de"),
    "fr": ("Berlin est situé dans le pays de l'Allemagne\n"
           "Rome est située dans le pays de l'Italie\n"
           "Madrid est situé dans le pays de l'Espagne\n"
           "Paris est situé dans le pays de"),
    "ja": ("ベルリンはドイツという国にあります\n"
           "ローマはイタリアという国にあります\n"
           "マドリードはスペインという国にあります\n"
           "パリは"),
}


def eos_mass(model, tok, prompt, eos_ids):
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits[0, -1].float()
    p = torch.softmax(logits, dim=-1)
    top = torch.topk(p, 5)
    return (float(p[list(eos_ids)].sum()),
            [(tok.decode([i]), round(float(v), 4)) for v, i in zip(top.values, top.indices)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--output_json", default=None)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    # Qwen has several stop tokens; count all of them as "stop now".
    eos_ids = {tok.eos_token_id}
    for t in ("<|endoftext|>", "<|im_end|>", "<|end_of_text|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            eos_ids.add(i)
    eos_ids.discard(None)

    print(f"\n=== {a.label or a.model}   (stop tokens: {sorted(eos_ids)})")
    print(f"{'lang':<6} {'P(stop | TRAIN prompt)':>24} {'P(stop | KLAR prompt)':>23}")
    out = {}
    for lang in TRAIN_PROMPTS:
        p_train, top_train = eos_mass(model, tok, TRAIN_PROMPTS[lang], eos_ids)
        p_klar, top_klar = eos_mass(model, tok, KLAR_PROMPTS[lang], eos_ids)
        out[lang] = {"train": p_train, "klar": p_klar,
                     "top_train": top_train, "top_klar": top_klar}
        print(f"{lang:<6} {p_train:>24.4f} {p_klar:>23.4f}")
    print("\ntop-5 next tokens under the KLAR prompt:")
    for lang, v in out.items():
        print(f"  {lang}: {v['top_klar']}")

    if a.output_json:
        json.dump({"model": a.model, "label": a.label, "probe": out},
                  open(a.output_json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

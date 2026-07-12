#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-lingual consistency evaluation: Total Consistency + RankC.

Motivation (reviewer feedback): per-language accuracy does not measure whether
the model gives the SAME answer to parallel prompts across languages. This
script evaluates, for each fact that exists in all languages:

  1. Per-language accuracy (sanity check, matches evaluate_accuracy.py)
  2. Total Consistency: fraction of facts answered correctly in ALL languages
     (plus the full distribution over "correct in k languages")
  3. Pairwise answer agreement: fraction of facts where two languages pick the
     same underlying candidate entity (regardless of correctness)
  4. RankC (Qi et al., EMNLP 2023): ranking-based consistency per language pair

RankC definition (Qi et al. 2023, eqs. 3-5):
    RankC(l, l') = mean over queries of  sum_{j=1..N} w_j * P@j
    P@j  = |top_j(l) ∩ top_j(l')| / j          (overlap of candidate SETS)
    w_j  = exp(N - j) / sum_k exp(N - k)       (softmax favouring rank 1)

Candidate alignment across languages
------------------------------------
PolyFact options are the same 4 entities per fact (1 gold + 3 distractors),
localized into each language and independently shuffled, WITHOUT stored
distractor entity ids. We therefore align each language's options to the
English options ("slots"):
  - the gold option is aligned via answer_text (always exact),
  - distractors are aligned by normalized string match first (handles
    identical names, mostly Latin scripts), then by multilingual sentence
    embeddings (LaBSE) with Hungarian assignment on the remaining 3x3 (or
    smaller) cost matrix.
Alignments are cached to JSON (--alignment_cache) so they are computed once
and reused across all model variants. Facts that cannot be aligned are
excluded from RankC / agreement (but still count for accuracy and Total
Consistency, which need no alignment); exclusion counts are reported.

Global-MMLU is fully parallel with option order preserved across languages,
so alignment is simply the option index and sample_id joins languages.

Examples
--------
PolyFact:
  python evaluate/evaluate_crosslingual_consistency.py \
    --benchmark polyfact \
    --hf_dataset jvonrad/WIKI-FACT --split test \
    --model allenai/OLMo-2-1124-7B \
    --batch_size 8 \
    --alignment_cache results/polyfact_test_alignment.json \
    --output_json results/olmo2_base_polyfact_consistency.json

Global-MMLU (all 12 paper languages, letter scoring as in lm-eval-harness):
  python evaluate/evaluate_crosslingual_consistency.py \
    --benchmark global_mmlu \
    --model allenai/OLMo-2-1124-7B \
    --batch_size 8 \
    --output_json results/olmo2_base_gmmlu_consistency.json

Device is auto-detected: CUDA > Neuron/XLA (Trainium) > CPU. On XLA, batches
and sequence lengths are padded to fixed shapes to avoid recompilation.
"""

import os
import json
import math
import argparse
import unicodedata
from collections import defaultdict
from itertools import combinations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
REF_LANG = "en"
N_OPTIONS = 4

GMMLU_DATASETS = {
    "global_mmlu": "CohereLabs/Global-MMLU",
    "global_mmlu_lite": "CohereLabs/Global-MMLU-Lite",
}


def is_gmmlu(benchmark: str) -> bool:
    return benchmark in GMMLU_DATASETS


# --------------------------------------------------------------------------
# Prompting (kept identical to evaluate/evaluate_accuracy.py for PolyFact)
# --------------------------------------------------------------------------

def build_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def build_gmmlu_letter_prompt(item: dict) -> str:
    q = item["question"].strip()
    opts = item["options"]
    letters = ["A", "B", "C", "D"]
    lines = [f"Question: {q}"]
    for letter, opt in zip(letters, opts):
        lines.append(f"{letter}. {opt}")
    lines.append("Answer:")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_polyfact_facts(args):
    """
    Returns: dict fact_id -> lang -> {"question", "options", "gold_idx"}
    Supports the nested schema of jvonrad/WIKI-FACT ({"langs": {...}}) both
    from the HF hub and from a local JSONL.
    """
    if args.input_jsonl:
        print(f"Loading rows from local JSONL: {args.input_jsonl}")
        rows = []
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    print(f"[WARN] skipping bad json line {i}: {e}")
    else:
        print(f"Loading {args.hf_dataset} [{args.split}] from Hugging Face")
        if args.hf_config:
            ds = load_dataset(args.hf_dataset, args.hf_config, split=args.split)
        else:
            ds = load_dataset(args.hf_dataset, split=args.split)
        rows = list(ds)

    facts = {}
    for row in rows:
        if "langs" not in row or not isinstance(row["langs"], dict):
            continue
        fact_id = row.get("fact_id")
        per_lang = {}
        for lang in LANGS:
            item = row["langs"].get(lang)
            if not item:
                continue
            question = (item.get("question") or "").strip()
            options = item.get("options", [])
            gold = (item.get("answer_text") or "").strip()
            if (
                not question
                or not isinstance(options, list)
                or len(options) != N_OPTIONS
                or gold not in options
                or len(set(options)) != N_OPTIONS
            ):
                continue
            per_lang[lang] = {
                "question": question,
                "options": list(options),
                "gold_idx": options.index(gold),
            }
        if per_lang:
            facts[fact_id] = per_lang

    return facts


def load_gmmlu_facts(args, langs):
    """
    Returns: dict sample_id -> lang -> {"question", "options", "gold_idx"}
    Options are parallel by index across languages; gold is the answer letter.
    Global-MMLU-Lite has no Russian config, so the caller passes the language
    subset to iterate (see --langs); we never hardcode the full LANGS here.
    """
    dataset = GMMLU_DATASETS[args.benchmark]
    letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    facts = defaultdict(dict)
    for lang in langs:
        print(f"Loading {dataset} [{lang}] split={args.split}")
        ds = load_dataset(dataset, lang, split=args.split)
        for row in ds:
            gold_idx = letter_to_idx.get((row.get("answer") or "").strip())
            options = [
                row.get("option_a"),
                row.get("option_b"),
                row.get("option_c"),
                row.get("option_d"),
            ]
            if gold_idx is None or any(o is None or not str(o).strip() for o in options):
                continue
            facts[row["sample_id"]][lang] = {
                "question": (row.get("question") or "").strip(),
                "options": [str(o).strip() for o in options],
                "gold_idx": gold_idx,
            }
    return dict(facts)


# --------------------------------------------------------------------------
# Cross-lingual option alignment (PolyFact)
# --------------------------------------------------------------------------

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold().strip()
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))
    s = " ".join(s.split())
    return s


def hungarian_3x3(cost):
    """Brute-force optimal assignment for tiny matrices (<= 3x3)."""
    from itertools import permutations

    n_rows = len(cost)
    n_cols = len(cost[0]) if n_rows else 0
    best, best_perm = None, None
    for perm in permutations(range(n_cols), n_rows):
        total = sum(cost[r][c] for r, c in enumerate(perm))
        if best is None or total < best:
            best, best_perm = total, perm
    return list(best_perm) if best_perm is not None else []


class OptionAligner:
    """
    Aligns each language's options to REF_LANG "slots" per fact.
    slot semantics: slot k == index of the corresponding option in REF_LANG.
    Produces: fact_id -> lang -> list[int] mapping option_idx -> slot.
    """

    def __init__(self, embedding_model: str, use_embeddings: bool = True):
        self.embedding_model_name = embedding_model
        self.use_embeddings = use_embeddings
        self._encoder = None
        self.stats = defaultdict(int)

    def _encode(self, texts):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model for alignment: {self.embedding_model_name}")
            self._encoder = SentenceTransformer(self.embedding_model_name, device="cpu")
        import numpy as np
        emb = self._encoder.encode(
            texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True
        )
        return np.asarray(emb)

    def align(self, facts):
        """
        facts: fact_id -> lang -> {"options", "gold_idx", ...}
        returns: fact_id -> lang -> list[int] (option_idx -> slot) or None
        """
        alignment = {}
        pending = []  # (fact_id, lang, [(opt_idx, [candidate_ref_idx,...])])
        all_texts = set()

        for fact_id, per_lang in facts.items():
            if REF_LANG not in per_lang:
                self.stats["fact_missing_ref_lang"] += 1
                continue
            ref = per_lang[REF_LANG]
            ref_norm = [normalize_text(o) for o in ref["options"]]
            alignment[fact_id] = {REF_LANG: list(range(N_OPTIONS))}

            for lang, item in per_lang.items():
                if lang == REF_LANG:
                    continue
                mapping = [None] * N_OPTIONS
                used_ref = set()

                # 1) gold aligns to gold
                mapping[item["gold_idx"]] = ref["gold_idx"]
                used_ref.add(ref["gold_idx"])

                # 2) exact normalized string match for distractors
                for oi, opt in enumerate(item["options"]):
                    if mapping[oi] is not None:
                        continue
                    n = normalize_text(opt)
                    matches = [
                        ri for ri in range(N_OPTIONS)
                        if ri not in used_ref and ref_norm[ri] == n
                    ]
                    if len(matches) == 1:
                        mapping[oi] = matches[0]
                        used_ref.add(matches[0])

                open_opts = [oi for oi in range(N_OPTIONS) if mapping[oi] is None]
                open_refs = [ri for ri in range(N_OPTIONS) if ri not in used_ref]

                if not open_opts:
                    alignment[fact_id][lang] = mapping
                    self.stats["aligned_exact"] += 1
                elif len(open_opts) == 1:
                    mapping[open_opts[0]] = open_refs[0]
                    alignment[fact_id][lang] = mapping
                    self.stats["aligned_exact"] += 1
                elif self.use_embeddings:
                    pending.append((fact_id, lang, mapping, open_opts, open_refs))
                    for oi in open_opts:
                        all_texts.add(per_lang[lang]["options"][oi])
                    for ri in open_refs:
                        all_texts.add(ref["options"][ri])
                else:
                    alignment[fact_id][lang] = None
                    self.stats["unaligned"] += 1

        if pending:
            texts = sorted(all_texts)
            text_to_idx = {t: i for i, t in enumerate(texts)}
            emb = self._encode(texts)

            for fact_id, lang, mapping, open_opts, open_refs in pending:
                opts = facts[fact_id][lang]["options"]
                refs = facts[fact_id][REF_LANG]["options"]
                cost = [
                    [
                        1.0 - float(
                            emb[text_to_idx[opts[oi]]] @ emb[text_to_idx[refs[ri]]]
                        )
                        for ri in open_refs
                    ]
                    for oi in open_opts
                ]
                assign = hungarian_3x3(cost)
                for k, oi in enumerate(open_opts):
                    mapping[oi] = open_refs[assign[k]]
                alignment[fact_id][lang] = mapping
                self.stats["aligned_embedding"] += 1

        return alignment


def get_polyfact_alignment(facts, args):
    if args.alignment_cache and os.path.exists(args.alignment_cache):
        print(f"Loading cached alignment: {args.alignment_cache}")
        with open(args.alignment_cache, "r", encoding="utf-8") as f:
            return json.load(f)

    aligner = OptionAligner(
        embedding_model=args.embedding_model,
        use_embeddings=not args.no_embedding_alignment,
    )
    alignment = aligner.align(facts)
    print(f"Alignment stats: {dict(aligner.stats)}")

    if args.alignment_cache:
        os.makedirs(os.path.dirname(args.alignment_cache) or ".", exist_ok=True)
        with open(args.alignment_cache, "w", encoding="utf-8") as f:
            json.dump(alignment, f, ensure_ascii=False)
        print(f"Saved alignment cache: {args.alignment_cache}")

    return alignment


# --------------------------------------------------------------------------
# Scoring (per-option length-normalized conditional logprob)
# --------------------------------------------------------------------------

def get_device(requested: str):
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return torch.device("cuda"), "cuda"
    if requested in ("xla", "auto"):
        try:
            import torch_xla.core.xla_model as xm  # noqa: F401
            return xm.xla_device(), "xla"
        except Exception:
            if requested == "xla":
                raise
    return torch.device("cpu"), "cpu"


def score_candidates_batch(
    model, tokenizer, examples, device, device_kind,
    score_mode="avg", max_length=512, fixed_batch_size=None,
):
    """
    examples: list of dicts with keys "prompt" and "options".
    Returns list[list[float]]: score per option per example.
    On XLA, pads to (fixed_batch_size * N_OPTIONS, max_length) so every
    forward pass has an identical shape (single compilation).
    """
    flat_texts, meta = [], []
    for ex_idx, ex in enumerate(examples):
        for opt_idx, opt in enumerate(ex["options"]):
            flat_texts.append(ex["prompt"] + " " + opt)
            meta.append((ex_idx, opt_idx))

    pad_rows = 0
    if device_kind == "xla" and fixed_batch_size is not None:
        target = fixed_batch_size * N_OPTIONS
        pad_rows = target - len(flat_texts)
        flat_texts.extend([flat_texts[0]] * pad_rows)

    padding = "max_length" if device_kind == "xla" else True
    enc = tokenizer(
        flat_texts,
        return_tensors="pt",
        padding=padding,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logits = logits[:, :-1, :].float()
        target_ids = input_ids[:, 1:]
        # log p(target) = logit[target] - logsumexp(logits): avoids materializing
        # a second [B, T, V] log-softmax tensor
        gathered = torch.gather(logits, 2, target_ids.unsqueeze(-1)).squeeze(-1)
        token_logprobs = gathered - torch.logsumexp(logits, dim=-1)  # [B, T-1]

    # Materialize once on CPU (also forces XLA execution exactly once per batch)
    token_logprobs = token_logprobs.cpu()
    target_mask = enc["attention_mask"][:, 1:]

    prompt_lens = [
        len(ids) for ids in tokenizer(
            [examples[m[0]]["prompt"] for m in meta],
            padding=False, truncation=True, max_length=max_length,
        )["input_ids"]
    ]

    scores = [[0.0] * len(ex["options"]) for ex in examples]
    for row_idx, (ex_idx, opt_idx) in enumerate(meta):
        plen = prompt_lens[row_idx]
        seq_len = int(target_mask[row_idx].sum().item()) + 1
        if padding == "max_length" and tokenizer.padding_side == "left":
            offset = max_length - seq_len
        else:
            offset = 0
        start = offset + max(plen - 1, 0)
        end = offset + seq_len - 1
        opt_lp = token_logprobs[row_idx, start:end]
        if opt_lp.numel() == 0:
            score = -1e9
        elif score_mode == "sum":
            score = float(opt_lp.sum().item())
        else:
            score = float(opt_lp.mean().item())
        scores[ex_idx][opt_idx] = score

    return scores


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def rankc_weights(n: int):
    ws = [math.exp(n - j) for j in range(1, n + 1)]
    z = sum(ws)
    return [w / z for w in ws]


RANKC_W = rankc_weights(N_OPTIONS)


def rankc_pair(rank_slots_1, rank_slots_2):
    """rank_slots_x: list of slots ordered by descending model score."""
    score = 0.0
    for j in range(1, N_OPTIONS + 1):
        top1 = set(rank_slots_1[:j])
        top2 = set(rank_slots_2[:j])
        score += RANKC_W[j - 1] * len(top1 & top2) / j
    return score


def compute_metrics(facts, predictions, alignment, langs):
    """
    predictions: fact_id -> lang -> {"scores": [4], "pred_idx": int, "correct": bool}
    alignment:   fact_id -> lang -> option_idx->slot list (or None); identity for gmmlu
    """
    results = {}

    # ---- per-language accuracy ----
    per_lang = {}
    for lang in langs:
        pairs = [p[lang] for p in predictions.values() if lang in p]
        per_lang[lang] = {
            "accuracy": sum(x["correct"] for x in pairs) / max(len(pairs), 1),
            "n": len(pairs),
        }
    results["per_language_accuracy"] = per_lang

    # ---- total consistency (facts present in ALL langs) ----
    complete = {
        fid: p for fid, p in predictions.items()
        if all(l in p for l in langs)
    }
    k_correct_hist = defaultdict(int)
    all_correct = 0
    for fid, p in complete.items():
        k = sum(p[l]["correct"] for l in langs)
        k_correct_hist[k] += 1
        if k == len(langs):
            all_correct += 1
    n_complete = len(complete)
    results["total_consistency"] = {
        "all_langs_correct_fraction": all_correct / max(n_complete, 1),
        "n_facts_all_langs": n_complete,
        "n_facts_total": len(predictions),
        "correct_language_count_histogram": {
            str(k): k_correct_hist[k] for k in sorted(k_correct_hist)
        },
        "mean_languages_correct": (
            sum(k * v for k, v in k_correct_hist.items()) / max(n_complete, 1)
        ),
    }

    # ---- pairwise: RankC + answer agreement ----
    def slots_for(fid, lang):
        m = alignment.get(fid, {}).get(lang)
        if m is None:
            return None
        scores = predictions[fid][lang]["scores"]
        order = sorted(range(N_OPTIONS), key=lambda i: scores[i], reverse=True)
        return [m[i] for i in order]

    rankc_matrix, agree_matrix, pair_n = {}, {}, {}
    for l1, l2 in combinations(langs, 2):
        rc_sum, agree_sum, n = 0.0, 0, 0
        for fid, p in predictions.items():
            if l1 not in p or l2 not in p:
                continue
            s1, s2 = slots_for(fid, l1), slots_for(fid, l2)
            if s1 is None or s2 is None:
                continue
            rc_sum += rankc_pair(s1, s2)
            agree_sum += int(s1[0] == s2[0])
            n += 1
        key = f"{l1}-{l2}"
        pair_n[key] = n
        rankc_matrix[key] = rc_sum / max(n, 1)
        agree_matrix[key] = agree_sum / max(n, 1)

    n_pairs = len(rankc_matrix)
    results["rankc"] = {
        "pairwise": rankc_matrix,
        "average": sum(rankc_matrix.values()) / max(n_pairs, 1),
        "average_en_x": (
            sum(v for k, v in rankc_matrix.items() if k.startswith("en-") or k.endswith("-en"))
            / max(len(langs) - 1, 1)
        ),
        "pair_n": pair_n,
        "weights": RANKC_W,
    }
    results["answer_agreement"] = {
        "pairwise": agree_matrix,
        "average": sum(agree_matrix.values()) / max(n_pairs, 1),
    }

    return results


def print_pair_matrix(title, pairwise, langs):
    print(f"\n{title}")
    header = "      " + "".join(f"{l:>7}" for l in langs)
    print(header)
    for l1 in langs:
        cells = []
        for l2 in langs:
            if l1 == l2:
                cells.append("      -")
            else:
                key = f"{l1}-{l2}" if f"{l1}-{l2}" in pairwise else f"{l2}-{l1}"
                cells.append(f"{pairwise.get(key, float('nan')):7.3f}")
        print(f"{l1:>4}  " + "".join(cells))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark",
                    choices=["polyfact", "global_mmlu", "global_mmlu_lite"],
                    default="polyfact")
    ap.add_argument("--hf_dataset", default="jvonrad/WIKI-FACT")
    ap.add_argument("--hf_config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--input_jsonl", default=None, help="Local JSONL instead of HF (polyfact)")
    ap.add_argument("--model", default="allenai/OLMo-2-1124-7B")
    ap.add_argument("--batch_size", type=int, default=8, help="Facts per forward batch")
    ap.add_argument("--max_facts", type=int, default=0, help="0 = all")
    ap.add_argument("--score_mode", choices=["sum", "avg"], default="avg")
    ap.add_argument("--max_length", type=int, default=512,
                    help="Fixed sequence length on XLA; truncation limit elsewhere")
    ap.add_argument("--device", choices=["auto", "cuda", "xla", "cpu"], default="auto")
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--alignment_cache", default=None,
                    help="JSON path to cache/reuse cross-lingual option alignment")
    ap.add_argument("--embedding_model", default="sentence-transformers/LaBSE")
    ap.add_argument("--no_embedding_alignment", action="store_true",
                    help="Skip embedding alignment; unaligned facts drop out of RankC")
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]

    # ---- data ----
    if args.benchmark == "polyfact":
        facts = load_polyfact_facts(args)
    else:
        facts = load_gmmlu_facts(args, langs)
    print(f"Loaded {len(facts):,} facts")

    if args.max_facts > 0:
        keep = sorted(facts.keys())[: args.max_facts]
        facts = {k: facts[k] for k in keep}
        print(f"Truncated to {len(facts):,} facts (--max_facts)")

    # ---- alignment ----
    if args.benchmark == "polyfact":
        alignment = get_polyfact_alignment(facts, args)
    else:
        identity = list(range(N_OPTIONS))
        alignment = {
            fid: {lang: identity for lang in per_lang}
            for fid, per_lang in facts.items()
        }

    # ---- model ----
    device, device_kind = get_device(args.device)
    print(f"Device: {device_kind}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right padding keeps offset logic trivial and matches evaluate_accuracy.py
    tokenizer.padding_side = "right"

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if device_kind in ("cuda", "xla") else torch.float32,
        trust_remote_code=True,
        device_map="auto" if device_kind == "cuda" else None,
    )
    if device_kind != "cuda":
        model = model.to(device)
    model.eval()

    # ---- scoring ----
    predictions = defaultdict(dict)
    for lang in langs:
        examples = []
        for fid, per_lang in facts.items():
            item = per_lang.get(lang)
            if not item:
                continue
            if is_gmmlu(args.benchmark):
                prompt = build_gmmlu_letter_prompt(item)
                options = ["A", "B", "C", "D"]
            else:
                prompt = build_prompt(item["question"])
                options = item["options"]
            examples.append({
                "fact_id": fid,
                "prompt": prompt,
                "options": options,
                "gold_idx": item["gold_idx"],
            })

        print(f"\n[{lang}] scoring {len(examples):,} facts")
        correct = 0
        for i in range(0, len(examples), args.batch_size):
            batch = examples[i : i + args.batch_size]
            score_lists = score_candidates_batch(
                model, tokenizer, batch, device, device_kind,
                score_mode=args.score_mode,
                max_length=args.max_length,
                fixed_batch_size=args.batch_size,
            )
            for ex, scores in zip(batch, score_lists):
                pred_idx = max(range(len(scores)), key=lambda k: scores[k])
                is_correct = pred_idx == ex["gold_idx"]
                correct += int(is_correct)
                predictions[ex["fact_id"]][lang] = {
                    "scores": scores,
                    "pred_idx": pred_idx,
                    "correct": is_correct,
                }
            done = min(i + args.batch_size, len(examples))
            if done % (args.batch_size * 50) < args.batch_size:
                print(f"  [{lang}] {done}/{len(examples)}  acc so far={correct/max(done,1):.4f}",
                      flush=True)
        print(f"[{lang}] accuracy = {correct / max(len(examples), 1):.4f}")

    # ---- metrics ----
    results = compute_metrics(facts, predictions, alignment, langs)
    results["config"] = {
        "benchmark": args.benchmark,
        "model": args.model,
        "split": args.split,
        "score_mode": args.score_mode,
        "langs": langs,
        "n_facts": len(facts),
    }

    print("\n" + "=" * 80)
    print("PER-LANGUAGE ACCURACY")
    for lang in langs:
        m = results["per_language_accuracy"][lang]
        print(f"  {lang:>2}  acc={m['accuracy']:.4f}  n={m['n']}")

    tc = results["total_consistency"]
    print("\nTOTAL CONSISTENCY (fully parallel facts)")
    print(f"  correct in ALL {len(langs)} languages: {tc['all_langs_correct_fraction']:.4f} "
          f"({tc['n_facts_all_langs']} facts)")
    print(f"  mean #languages correct per fact:  {tc['mean_languages_correct']:.2f}")
    print(f"  histogram (k langs correct -> facts): {tc['correct_language_count_histogram']}")

    print(f"\nRANKC  average={results['rankc']['average']:.4f}  "
          f"en-X average={results['rankc']['average_en_x']:.4f}")
    print_pair_matrix("RankC matrix:", results["rankc"]["pairwise"], langs)
    print(f"\nANSWER AGREEMENT  average={results['answer_agreement']['average']:.4f}")
    print_pair_matrix("Agreement matrix:", results["answer_agreement"]["pairwise"], langs)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        results["predictions"] = {
            fid: {
                lang: {"pred_idx": p["pred_idx"], "correct": p["correct"], "scores": p["scores"]}
                for lang, p in per_lang.items()
            }
            for fid, per_lang in predictions.items()
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print(f"\nSaved results: {args.output_json}")


if __name__ == "__main__":
    main()

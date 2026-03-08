import json
import math
from collections import defaultdict
from typing import Iterable, Dict, List, Optional, Set

# -------------------------
# Config
# -------------------------
MGSM_LANGS = ["en", "de", "id", "pt", "bn", "sw", "es", "ru", "fr", "ja", "zh-cn", "ar"]

# -------------------------
# Utilities
# -------------------------
def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def nonempty_str(x) -> bool:
    return isinstance(x, str) and len(x.strip()) > 0

def present_langs_in_row(ex: Dict, target_langs: Iterable[str]) -> Set[str]:
    pd = ex.get("para_data", {}) or {}
    return {l for l in target_langs if nonempty_str(pd.get(l))}

def find_eligible_talk_ids(path: str, target_langs: List[str], min_langs_per_talk: int = 2) -> Set[str]:
    talk_to_langs = defaultdict(set)
    for ex in iter_jsonl(path):
        tid = ex.get("talk_id")
        if tid:
            talk_to_langs[tid].update(present_langs_in_row(ex, target_langs))
    return {tid for tid, langs in talk_to_langs.items() if len(langs) >= min_langs_per_talk}

def prune_to_target_langs(ex: Dict, target_langs: List[str]) -> Dict:
    pd = ex.get("para_data", {}) or {}
    new_pd = {l: pd[l] for l in target_langs if nonempty_str(pd.get(l))}
    new_ex = dict(ex)
    new_ex["para_data"] = new_pd
    return new_ex

def filter_examples(
    path: str,
    target_langs: List[str],
    min_langs_per_talk: int = 2,
    min_langs_per_row: int = 2,
) -> List[Dict]:
    eligible = find_eligible_talk_ids(path, target_langs, min_langs_per_talk)

    out = []
    for ex in iter_jsonl(path):
        if ex.get("talk_id") not in eligible:
            continue
        ex2 = prune_to_target_langs(ex, target_langs)
        if len(ex2["para_data"]) >= min_langs_per_row:
            out.append(ex2)

    return out

# -------------------------
# Stats helpers
# -------------------------
def percentile(sorted_vals: List[int], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 100:
        return float(sorted_vals[-1])

    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

def summarize_lengths(lengths: List[int]) -> Dict[str, float]:
    if not lengths:
        return {}

    s = sorted(lengths)
    n = len(s)
    return {
        "count": n,
        "min": s[0],
        "mean": sum(s) / n,
        "median": percentile(s, 50),
        "p90": percentile(s, 90),
        "p95": percentile(s, 95),
        "p99": percentile(s, 99),
        "max": s[-1],
    }

# -------------------------
# Token Counting / Row Analysis
# -------------------------
def analyze_tokens_hf(
    examples: List[Dict],
    langs: List[str],
    tokenizer_name: str,
    hf_token: Optional[str] = None,
    thresholds: Optional[List[int]] = None,
) -> Dict:

    from transformers import AutoTokenizer
    from huggingface_hub import login

    if hf_token:
        login(token=hf_token)

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if tok.eos_token is None:
        raise ValueError("Tokenizer has no eos_token; please set one explicitly.")

    if thresholds is None:
        thresholds = [256, 512, 768, 1024, 1536, 2048, 3072, 4096]

    counts_by_lang = {l: 0 for l in langs}

    # row-level lengths
    row_total_tokens = []
    row_num_langs = []
    row_tokens_by_lang_count = defaultdict(list)   # e.g. rows with exactly 2 langs, 3 langs, ...
    per_lang_row_lengths = defaultdict(list)       # token length of each language string individually
    rows_exceeding_threshold = {t: 0 for t in thresholds}

    for ex in examples:
        pd = ex["para_data"]

        present = [l for l in langs if l in pd]
        row_num_langs.append(len(present))

        # token count per language entry in this row
        row_lang_token_sum = 0
        for l in present:
            n = len(tok.encode(pd[l], add_special_tokens=False))
            counts_by_lang[l] += n
            per_lang_row_lengths[l].append(n)
            row_lang_token_sum += n

        # this matches your training formatting more closely:
        # texts joined with "\n\n" and one eos at the end
        formatted_parts = [pd[l] for l in present]
        formatted_text = "\n\n".join(formatted_parts) + tok.eos_token
        row_total = len(tok.encode(formatted_text, add_special_tokens=False))

        row_total_tokens.append(row_total)
        row_tokens_by_lang_count[len(present)].append(row_total)

        for t in thresholds:
            if row_total > t:
                rows_exceeding_threshold[t] += 1

    total_tokens = sum(counts_by_lang.values())

    return {
        "counts_by_lang": counts_by_lang,
        "total_tokens_raw_sum": total_tokens,
        "row_total_tokens": row_total_tokens,
        "row_num_langs": row_num_langs,
        "row_tokens_by_lang_count": dict(row_tokens_by_lang_count),
        "per_lang_row_lengths": dict(per_lang_row_lengths),
        "rows_exceeding_threshold": rows_exceeding_threshold,
        "thresholds": thresholds,
    }

# -------------------------
# Summary
# -------------------------
def summarize(
    path: str,
    langs: List[str],
    tokenizer_name: str,
    min_langs_per_talk: int = 2,
    min_langs_per_row: int = 2,
    hf_token: Optional[str] = None,
):
    examples = filter_examples(
        path,
        langs,
        min_langs_per_talk=min_langs_per_talk,
        min_langs_per_row=min_langs_per_row,
    )

    print(f"Selected langs: {langs}")
    print(f"Matching rows:  {len(examples):,}")

    analysis = analyze_tokens_hf(
        examples,
        langs,
        tokenizer_name=tokenizer_name,
        hf_token=hf_token,
    )

    counts_by_lang = analysis["counts_by_lang"]
    row_total_tokens = analysis["row_total_tokens"]
    row_num_langs = analysis["row_num_langs"]
    row_tokens_by_lang_count = analysis["row_tokens_by_lang_count"]
    per_lang_row_lengths = analysis["per_lang_row_lengths"]
    rows_exceeding_threshold = analysis["rows_exceeding_threshold"]

    print("\nRaw token counts by language (sum of individual language strings):")
    for l in langs:
        print(f"  {l:<6}: {counts_by_lang[l]:,}")
    print(f"  total : {analysis['total_tokens_raw_sum']:,}")

    overall = summarize_lengths(row_total_tokens)
    print("\nPer-row total token stats (after formatting row as training text):")
    print(f"  count : {overall['count']:,}")
    print(f"  min   : {overall['min']:.0f}")
    print(f"  mean  : {overall['mean']:.2f}")
    print(f"  median: {overall['median']:.2f}")
    print(f"  p90   : {overall['p90']:.2f}")
    print(f"  p95   : {overall['p95']:.2f}")
    print(f"  p99   : {overall['p99']:.2f}")
    print(f"  max   : {overall['max']:.0f}")

    print("\nRows exceeding token thresholds:")
    total_rows = max(1, len(row_total_tokens))
    for t in sorted(rows_exceeding_threshold):
        c = rows_exceeding_threshold[t]
        pct = 100.0 * c / total_rows
        print(f"  > {t:<4}: {c:>8,} rows ({pct:6.2f}%)")

    lang_count_stats = summarize_lengths(row_num_langs)
    print("\nNumber of selected languages present per kept row:")
    print(f"  min   : {lang_count_stats['min']:.0f}")
    print(f"  mean  : {lang_count_stats['mean']:.2f}")
    print(f"  median: {lang_count_stats['median']:.2f}")
    print(f"  p90   : {lang_count_stats['p90']:.2f}")
    print(f"  max   : {lang_count_stats['max']:.0f}")

    print("\nPer-row total tokens grouped by number of languages in the row:")
    for k in sorted(row_tokens_by_lang_count):
        stats = summarize_lengths(row_tokens_by_lang_count[k])
        print(
            f"  {k:>2} langs | count={stats['count']:>8,} | "
            f"mean={stats['mean']:>8.2f} | p95={stats['p95']:>8.2f} | max={stats['max']:>6.0f}"
        )

    print("\nPer-language single-string token stats:")
    for l in langs:
        vals = per_lang_row_lengths.get(l, [])
        if not vals:
            print(f"  {l:<6}: no rows")
            continue
        stats = summarize_lengths(vals)
        print(
            f"  {l:<6}: count={stats['count']:>8,} | "
            f"mean={stats['mean']:>8.2f} | p95={stats['p95']:>8.2f} | max={stats['max']:>6.0f}"
        )

    return {
        "examples": examples,
        "analysis": analysis,
    }

# -------------------------
# Usage
# -------------------------
if __name__ == "__main__":
    DATA_PATH = "/data/jonathan/Lost-in-Mistranslation/datasets/TED2025/multi_way.jsonl"

    summarize(
        DATA_PATH,
        MGSM_LANGS,
        tokenizer_name="allenai/OLMo-2-1124-7B",
        min_langs_per_talk=2,
        min_langs_per_row=2,
    )
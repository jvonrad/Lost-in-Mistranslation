import json
from collections import Counter, defaultdict
from typing import Iterable, Dict, List, Tuple, Optional, Set

# -------------------------
# Config: pick languages here
# -------------------------
MGSM_LANGS = ["de","en","es","fr","ja","ru","zh"] #,"te","th","bn","sw",

def iter_jsonl(path: str):
    """Yield dicts from a JSONL file (one JSON object per line)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def has_all_langs(ex: Dict, langs: Iterable[str]) -> bool:
    """True iff para_data contains non-empty strings for all langs."""
    pd = ex.get("para_data", {})
    for l in langs:
        if l not in pd:
            return False
        v = pd[l]
        if v is None:
            return False
        if isinstance(v, str) and len(v.strip()) == 0:
            return False
    return True

def filter_examples(path: str, langs: List[str]) -> List[Dict]:
    """Return examples where all requested langs exist in para_data."""
    return [ex for ex in iter_jsonl(path) if has_all_langs(ex, langs)]

def count_talks(examples: List[Dict]) -> Tuple[int, Set[str]]:
    talk_ids = {ex.get("talk_id") for ex in examples if ex.get("talk_id") is not None}
    return len(talk_ids), talk_ids

def count_chars(examples: List[Dict], langs: List[str]) -> Dict[str, int]:
    """Fallback: character counts per language + total."""
    out = {l: 0 for l in langs}
    for ex in examples:
        pd = ex["para_data"]
        for l in langs:
            out[l] += len(pd[l])
    out["total"] = sum(out.values())
    return out

def count_tokens_hf(
    examples: List[Dict],
    langs: List[str],
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
    hf_token: Optional[str] = None,
) -> Dict[str, int]:
    """
    Token counts using a Hugging Face tokenizer.
    Note: Llama-2 tokenizer is gated -> set hf_token or login beforehand.
    """
    from transformers import AutoTokenizer
    from huggingface_hub import login

    if hf_token:
        login(token=hf_token)

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    counts = {l: 0 for l in langs}
    for ex in examples:
        pd = ex["para_data"]
        for l in langs:
            # no special tokens, just raw segmentation
            counts[l] += len(tok.encode(pd[l], add_special_tokens=False))
    counts["total"] = sum(counts.values())
    return counts

def summarize(path: str, langs: List[str], do_tokens: bool = True, **token_kwargs):
    examples = filter_examples(path, langs)
    n_rows = len(examples)
    n_talks, talk_ids = count_talks(examples)

    print(f"Selected langs: {langs}")
    print(f"Matching rows:  {n_rows}")
    print(f"Unique talks:   {n_talks}")

    if do_tokens:
        try:
            tok_counts = count_tokens_hf(examples, langs, **token_kwargs)
            print("\nToken counts:")
            for l in langs:
                print(f"  {l}: {tok_counts[l]:,}")
            print(f"  total: {tok_counts['total']:,}")
        except Exception as e:
            print(f"\n[Token count failed: {e}]")
            char_counts = count_chars(examples, langs)
            print("Fallback char counts:")
            for l in langs:
                print(f"  {l}: {char_counts[l]:,}")
            print(f"  total: {char_counts['total']:,}")

    return examples  # so you can reuse them downstream

# -------------------------
# Usage
# -------------------------
if __name__ == "__main__":
    DATA_PATH = "/data/jonathan/Lost-in-Mistranslation/TED2025/multi_way.jsonl"  # <-- change this

    # Example 1: your MGSM overlap set
    summarize(
        DATA_PATH,
        MGSM_LANGS,
        do_tokens=True,
        tokenizer_name="meta-llama/Llama-2-7b-hf",
        # hf_token="YOUR_HF_TOKEN",  # optional if you haven't logged in
    )

    # Example 2: custom subset (modular)
    # summarize(DATA_PATH, ["en", "de", "fr"], do_tokens=False)
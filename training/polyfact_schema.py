"""Shared row normalisation for PolyFact-Clean / PolyFact / WIKI-FACT.

The same facts ship in two different shapes and every training script used to
re-implement the parsing (which is how the RankC letter-vs-text bug survived so
long). Everything schema-specific lives here now.

  jvonrad/PolyFact-Clean, `parallel` config  (CURRENT TRAINING SET)
      row["translations"][lang] = {question, option_a..option_d, answer_index,
                                   answer_text, option_ids}
      NOTE: `answer_index` arrives as a STRING ('1'), not an int.
      Configs are ['ar',...,'zh','parallel'] with NO default, so a config name
      is mandatory — load_dataset(id, "parallel").
      Splits: train 56,324 / validation 444 / test 2,039.

  jvonrad/WIKI-FACT  (legacy)
      row["langs"][lang] = {question, options: [4 strings], answer_text}
      Single default config; no option ids, no answer_index.

Gold resolution prefers `answer_index` when present (PolyFact-Clean verifies
option_ids[answer_index] == object_id for every fact) and falls back to matching
`answer_text` against the options, which is all WIKI-FACT can offer.
"""

from typing import Any, Dict, List, Optional, Tuple

LETTERS = ("A", "B", "C", "D")
N_OPTIONS = 4


def safe_strip(x: Any) -> str:
    return str(x).strip() if x is not None else ""


def lang_blocks(ex: Dict[str, Any]) -> Dict[str, Any]:
    """Per-language block of a row, from either schema ({} if neither)."""
    block = ex.get("translations")
    if not isinstance(block, dict):
        block = ex.get("langs")
    return block if isinstance(block, dict) else {}


def normalize_lang_item(item: Any) -> Optional[Tuple[str, List[str], str, int]]:
    """-> (question, [4 options], answer_text, gold_idx) or None if unusable.

    Accepts either the option_a..option_d form (PolyFact / PolyFact-Clean) or
    the options-list form (WIKI-FACT).
    """
    if not isinstance(item, dict):
        return None

    question = safe_strip(item.get("question"))
    answer_text = safe_strip(item.get("answer_text"))
    if not question:
        return None

    opts = item.get("options")
    if isinstance(opts, list):
        options = [safe_strip(x) for x in opts]
    else:
        options = [safe_strip(item.get(f"option_{c}")) for c in ("a", "b", "c", "d")]

    if len(options) != N_OPTIONS or any(not o for o in options):
        return None

    gold_idx: Optional[int] = None
    raw_idx = item.get("answer_index")
    if raw_idx is not None and raw_idx != "":
        try:
            cand = int(raw_idx)          # PolyFact-Clean stores this as a string
            if 0 <= cand < N_OPTIONS:
                gold_idx = cand
        except (TypeError, ValueError):
            gold_idx = None

    # Cross-check against answer_text when both exist; trust answer_text if they
    # disagree, since a wrong gold label is far more damaging than a dropped row.
    if answer_text and answer_text in options:
        text_idx = options.index(answer_text)
        gold_idx = text_idx if gold_idx is None or options[gold_idx] != answer_text else gold_idx
    elif gold_idx is None:
        return None

    if gold_idx is None:
        return None
    if not answer_text:
        answer_text = options[gold_idx]
    return question, options, answer_text, gold_idx


def gold_letter(gold_idx: int) -> str:
    return LETTERS[gold_idx]


def option_map(options: List[str]) -> Dict[str, str]:
    return {LETTERS[i]: options[i] for i in range(N_OPTIONS)}


def load_split_dict(dataset_id: str, dataset_config: Optional[str]):
    """load_dataset that passes a config only when one is given.

    PolyFact-Clean has no default config, so omitting it raises; WIKI-FACT has
    only a default, so passing one raises. Hence the branch.
    """
    from datasets import load_dataset
    if dataset_config:
        return load_dataset(dataset_id, dataset_config)
    return load_dataset(dataset_id)

import json
from collections import defaultdict
from typing import Dict, List, Set
from transformers import AutoTokenizer

# -----------------
# CONFIG
# -----------------
JSONL_PATH = "/data/jonathan/Lost-in-Mistranslation/TED2025/multi_way.jsonl"

BASE_MODEL = "allenai/OLMo-2-1124-7B"

REQ_LANGS = ["en","de","id","pt","ar","bn","sw","es","ru","fr","ja","zh-cn"]
MIN_LANGS_PER_ROW = 1
MIN_LANGS_PER_TALK = 2
USE_TAGS = False

tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

# -----------------
# Helpers
# -----------------

def nonempty_str(x):
    return isinstance(x, str) and len(x.strip()) > 0


def prune_to_selected_langs(para_data: Dict, selected_langs: List[str]) -> Dict:
    pd = para_data or {}
    return {l: pd[l] for l in selected_langs if nonempty_str(pd.get(l))}


def eligible_talk_ids(path: str, selected_langs: List[str], k: int = 2) -> Set[str]:
    talk_to_langs = defaultdict(set)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            tid = obj.get("talk_id")
            if not tid:
                continue

            pd = prune_to_selected_langs(obj.get("para_data", {}), selected_langs)
            talk_to_langs[tid].update(pd.keys())

    return {tid for tid, langs in talk_to_langs.items() if len(langs) >= k}


def format_segment(para_data, use_tags, lang_order=None):

    pd = prune_to_selected_langs(para_data, lang_order)

    if not pd:
        return ""

    langs = [l for l in lang_order if l in pd]

    if use_tags:
        parts = [f"<{l}>\n{pd[l]}" for l in langs]
    else:
        parts = [pd[l] for l in langs]

    return "\n\n".join(parts) + tok.eos_token


# -----------------
# Dump 2 talks
# -----------------

def dump_two_talks():

    eligible = eligible_talk_ids(JSONL_PATH, REQ_LANGS, MIN_LANGS_PER_TALK)

    talks_seen = set()

    with open(JSONL_PATH, "r", encoding="utf-8") as f, \
         open("debug_two_talks.txt", "w", encoding="utf-8") as out:

        for line in f:

            obj = json.loads(line)

            talk_id = obj.get("talk_id")

            if talk_id not in eligible:
                continue

            pd = prune_to_selected_langs(obj.get("para_data", {}), REQ_LANGS)

            if len(pd) < MIN_LANGS_PER_ROW:
                continue

            if talk_id not in talks_seen:

                if len(talks_seen) >= 2:
                    break

                talks_seen.add(talk_id)

                out.write("\n" + "="*80 + "\n")
                out.write(f"TALK {len(talks_seen)} | talk_id={talk_id}\n")
                out.write("="*80 + "\n\n")

            seg = format_segment(pd, USE_TAGS, REQ_LANGS)

            out.write(seg)
            out.write("\n")

    print("Saved to debug_two_talks.txt")


if __name__ == "__main__":
    dump_two_talks()
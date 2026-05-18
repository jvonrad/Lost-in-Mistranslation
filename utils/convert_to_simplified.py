#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from opencc import OpenCC

INPUT_PATH = "/data/jonathan/Lost-in-Mistranslation/datasets/Wiki-triplets/multilingual_mcq_text_filtered.jsonl"
OUTPUT_PATH = "/data/jonathan/Lost-in-Mistranslation/datasets/Wiki-triplets/multilingual_mcq_text_filtered_zh_simplified.jsonl"

cc = OpenCC("t2s")

def convert_text(x):
    if isinstance(x, str):
        return cc.convert(x)
    return x

n = 0

with open(INPUT_PATH, "r", encoding="utf-8") as fin, open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    for line in fin:
        row = json.loads(line)

        if "langs" in row and "zh" in row["langs"]:
            zh = row["langs"]["zh"]

            # convert main fields
            if "question" in zh:
                zh["question"] = convert_text(zh["question"])

            if "answer_text" in zh:
                zh["answer_text"] = convert_text(zh["answer_text"])

            if "options" in zh:
                zh["options"] = [convert_text(o) for o in zh["options"]]

            if "_expected_gold_answer" in zh:
                zh["_expected_gold_answer"] = convert_text(zh["_expected_gold_answer"])

            if "_candidate_options" in zh:
                zh["_candidate_options"] = [convert_text(o) for o in zh["_candidate_options"]]

            # convert ONLY zh inside accepted_answers
            if "_accepted_answers" in zh and "zh" in zh["_accepted_answers"]:
                zh["_accepted_answers"]["zh"] = convert_text(zh["_accepted_answers"]["zh"])

        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1

print(f"done: wrote {n:,} rows to {OUTPUT_PATH}")
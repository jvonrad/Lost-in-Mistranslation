#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import bz2
import json
import argparse

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_bz2", required=True)
    ap.add_argument("--ids_txt", required=True)
    ap.add_argument("--output_jsonl", required=True)
    return ap.parse_args()

def parse_entity_line(line: str):
    line = line.strip()
    if not line or line == "[" or line == "]":
        return None
    if line.endswith(","):
        line = line[:-1]
    return json.loads(line)

def main():
    args = parse_args()

    with open(args.ids_txt, "r", encoding="utf-8") as f:
        wanted = {line.strip() for line in f if line.strip()}

    print(f"Need labels for {len(wanted):,} IDs")

    found = 0
    seen = 0

    with bz2.open(args.dump_bz2, "rt", encoding="utf-8") as f_in, \
         open(args.output_jsonl, "w", encoding="utf-8") as f_out:

        for line in f_in:
            obj = parse_entity_line(line)
            if obj is None:
                continue

            seen += 1
            ent_id = obj.get("id")
            if ent_id not in wanted:
                continue

            raw_labels = obj.get("labels", {})
            labels = {}
            for lang in LANGS:
                if lang in raw_labels and "value" in raw_labels[lang]:
                    labels[lang] = raw_labels[lang]["value"]

            if labels:
                f_out.write(json.dumps({
                    "id": ent_id,
                    "labels": labels
                }, ensure_ascii=False) + "\n")
                found += 1

            if found % 10000 == 0 and found > 0:
                print(f"found={found:,} seen={seen:,}")

    print(f"Done. wrote {found:,} rows")

if __name__ == "__main__":
    main()
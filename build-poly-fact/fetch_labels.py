#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import requests
import os

INPUT = "/data/jonathan/Lost-in-Mistranslation/datasets/Wiki-triplets/all_ids.txt"
OUTPUT = "/data/jonathan/Lost-in-Mistranslation/datasets/Wiki-triplets/labels.jsonl"
BATCH_SIZE = 50

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def main():
    
    existing_ids = set()

    if os.path.exists(OUTPUT):
        print("Loading existing labels...")
        with open(OUTPUT, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    existing_ids.add(row["id"])
                except:
                    continue

    print(f"Found {len(existing_ids):,} existing IDs")
    
    with open(INPUT, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
        ids = [i for i in ids if i not in existing_ids]
        print(f"Remaining IDs to fetch: {len(ids):,}")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "LostInMistranslationLabelFetcher/1.0 (research project)"
    })

    written = 0

    with open(OUTPUT, "a", encoding="utf-8") as out:
        for batch_idx, batch in enumerate(chunks(ids, BATCH_SIZE), start=1):
            if all(i in existing_ids for i in batch):
                continue
            ids_str = "|".join(batch)

            params = {
                "action": "wbgetentities",
                "ids": ids_str,
                "props": "labels",
                "languages": "|".join(LANGS),
                "format": "json",
            }

            success = False
            for attempt in range(8):
                try:
                    r = session.get(
                        "https://www.wikidata.org/w/api.php",
                        params=params,
                        timeout=60,
                    )

                    if r.status_code == 429:
                        wait_s = min(5 * (attempt + 1), 120)
                        print(f"[batch {batch_idx}] HTTP 429, sleeping {wait_s}s...")
                        time.sleep(wait_s)
                        continue

                    if r.status_code != 200:
                        wait_s = 10 * (attempt + 1)
                        print(f"[batch {batch_idx}] HTTP {r.status_code}, sleeping {wait_s}s...")
                        time.sleep(wait_s)
                        continue

                    data = r.json()
                    entities = data.get("entities", {})

                    for eid, entity in entities.items():
                        raw_labels = entity.get("labels", {})
                        labels = {
                            lang: raw_labels[lang]["value"]
                            for lang in raw_labels
                            if "value" in raw_labels[lang]
                        }

                        out.write(json.dumps({
                            "id": eid,
                            "labels": labels
                        }, ensure_ascii=False) + "\n")
                        written += 1

                    success = True
                    break
                    
                

                except requests.RequestException as e:
                    wait_s = 10 * (attempt + 1)
                    print(f"[batch {batch_idx}] request failed: {e}; sleeping {wait_s}s...")
                    time.sleep(wait_s)
            if not success:
                print(f"[batch {batch_idx}] failed permanently, skipping batch")

            if batch_idx % 100 == 0:
                print(f"processed batches={batch_idx:,} written={written:,}")

            time.sleep(0.25)

    print(f"Done. wrote {written:,} label rows to {OUTPUT}")


if __name__ == "__main__":
    main()
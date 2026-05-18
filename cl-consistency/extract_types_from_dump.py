#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import bz2
import json
import argparse


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_bz2", required=True)
    ap.add_argument("--ids_txt", required=True)
    ap.add_argument("--output_jsonl", required=True)
    return ap.parse_args()


def parse_entity_line(line: str):
    line = line.strip()
    if not line or line in {"[", "]"}:
        return None
    if line.endswith(","):
        line = line[:-1]
    return json.loads(line)


def extract_p31_types(entity_obj):
    claims = entity_obj.get("claims", {})
    p31_claims = claims.get("P31", [])
    out = set()

    for claim in p31_claims:
        try:
            mainsnak = claim.get("mainsnak", {})
            if mainsnak.get("snaktype") != "value":
                continue
            dv = mainsnak.get("datavalue", {})
            if dv.get("type") != "wikibase-entityid":
                continue
            val = dv.get("value", {})
            if val.get("entity-type") != "item":
                continue
            qid = val.get("id")
            if qid:
                out.add(qid)
        except Exception:
            continue

    return sorted(out)


def main():
    args = parse_args()

    with open(args.ids_txt, "r", encoding="utf-8") as f:
        wanted = {line.strip() for line in f if line.strip()}

    print(f"Need types for {len(wanted):,} IDs")

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

            types = extract_p31_types(obj)
            f_out.write(json.dumps({
                "id": ent_id,
                "types": types
            }, ensure_ascii=False) + "\n")
            found += 1

            if found % 10000 == 0:
                print(f"found={found:,} seen={seen:,}")

    print(f"Done. wrote {found:,} rows")


if __name__ == "__main__":
    main()
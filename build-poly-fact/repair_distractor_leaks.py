#!/usr/bin/env python3
"""Targeted relation-matched replacements for two residual distractor leaks.

The facts and questions remain unchanged.  Only the leaking distractor entity is
replaced, consistently across all 12 flat configs and the parallel config.
"""

import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq


LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
SPLITS = ["train", "validation", "test"]
OPTIONS = ["option_a", "option_b", "option_c", "option_d"]
REPAIRS = {
    # 258th Infantry Division's Chinese name contains the Germany distractor.
    "Q218318|P17|Q7318": {"old": "Q183", "new": "Q38"},
    # The mistranslated Chinese subject "US Montauban" contains the US distractor.
    "Q3547511|P17|Q142": {"old": "Q30", "new": "Q183"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    def source(config, split):
        return os.path.join(args.data_dir, "data", config, "{}.parquet".format(split))

    # Canonical replacement labels come from surviving facts where the entity is
    # the gold, exactly as the full distractor resampler builds its label map.
    needed = {repair["new"] for repair in REPAIRS.values()}
    labels = {lang: {} for lang in LANGS}
    for split in SPLITS:
        for row in pq.read_table(source("parallel", split)).to_pylist():
            if row["object_id"] in needed:
                for lang in LANGS:
                    labels[lang][row["object_id"]] = row["translations"][lang]["answer_text"]
    for lang in LANGS:
        missing = needed - set(labels[lang])
        if missing:
            raise ValueError("missing canonical {} labels for {}".format(lang, sorted(missing)))

    os.makedirs(args.out_dir, exist_ok=True)
    touched_flat = set()
    for lang in LANGS:
        directory = os.path.join(args.out_dir, "data", lang)
        os.makedirs(directory, exist_ok=True)
        for split in SPLITS:
            table = pq.read_table(source(lang, split))
            rows = table.to_pylist()
            if split == "train":
                for row in rows:
                    repair = REPAIRS.get(row["fact_id"])
                    if not repair:
                        continue
                    old, new = repair["old"], repair["new"]
                    ids = list(row["option_ids"])
                    if ids.count(old) != 1 or new in ids:
                        raise ValueError("unexpected option ids for {} / {}".format(row["fact_id"], lang))
                    index = ids.index(old)
                    if index == row["answer_index"]:
                        raise ValueError("refusing to replace a gold option")
                    ids[index] = new
                    row["option_ids"] = ids
                    row[OPTIONS[index]] = labels[lang][new]
                    touched_flat.add((row["fact_id"], lang))
            pq.write_table(
                pa.Table.from_pylist(rows, schema=table.schema),
                os.path.join(directory, "{}.parquet".format(split)),
                compression="snappy",
            )

    parallel_dir = os.path.join(args.out_dir, "data", "parallel")
    os.makedirs(parallel_dir, exist_ok=True)
    touched_parallel = set()
    for split in SPLITS:
        table = pq.read_table(source("parallel", split))
        rows = table.to_pylist()
        if split == "train":
            for row in rows:
                repair = REPAIRS.get(row["fact_id"])
                if not repair:
                    continue
                old, new = repair["old"], repair["new"]
                for lang in LANGS:
                    translation = row["translations"][lang]
                    ids = list(translation["option_ids"])
                    if ids.count(old) != 1 or new in ids:
                        raise ValueError("unexpected parallel ids for {} / {}".format(row["fact_id"], lang))
                    index = ids.index(old)
                    if index == translation["answer_index"]:
                        raise ValueError("refusing to replace a parallel gold option")
                    ids[index] = new
                    translation["option_ids"] = ids
                    translation[OPTIONS[index]] = labels[lang][new]
                    touched_parallel.add((row["fact_id"], lang))
        pq.write_table(
            pa.Table.from_pylist(rows, schema=table.schema),
            os.path.join(parallel_dir, "{}.parquet".format(split)),
            compression="snappy",
        )

    expected = {(fact_id, lang) for fact_id in REPAIRS for lang in LANGS}
    if touched_flat != expected or touched_parallel != expected:
        raise ValueError("not every repair was applied in flat and parallel configs")
    print("replaced {} leaking distractors across 12 languages and parallel".format(len(REPAIRS)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Apply full-census verification flags and gated question repairs to v13.

Unlike the earlier partial-review patcher, this stage consumes every worksheet
listed in ``fullcensus/index.json``, including the English-vs-gold review.  A
confirmed English conflict makes every language for that fact unverified until
that language receives an accepted replacement, because translation fidelity to
a broken English reference is not evidence that the translated question is true.

Existing ``question_regenerated`` values are preserved.  This matters when v13
is based on v12: a prior accepted repair remains identifiable even if the new
census independently verifies its current text.

``--target_split`` (default ``test``) selects which split the census/proposals/
adjudications apply to; every other split passes through byte-identical. Used
with ``--target_split validation`` to apply the analogous validation census.
"""

import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq


LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
SPLITS = ["train", "validation", "test"]
OPTIONS = ["option_a", "option_b", "option_c", "option_d"]
NO_WORD_BOUNDARY = {"ja", "zh"}
SCRIPT = {
    "ar": "ARABIC",
    "bn": "BENGALI",
    "ru": "CYRILLIC",
    "zh": "CJK",
    "ja": ("CJK", "HIRAGANA", "KATAKANA"),
}
ROW_RE = re.compile(
    r"^\|\s*\d+\s*\|\s*([^|\s]+)\s*\|\s*([^|\s]+)\s*\|\s*([^|\s]+)\s*\|",
    re.MULTILINE,
)


def leaks(needle, question, lang):
    floor = 2 if lang in NO_WORD_BOUNDARY else 3
    if not needle or len(needle) < floor:
        return False
    needle = needle.casefold()
    question = question.casefold()
    if lang in NO_WORD_BOUNDARY:
        return needle in question
    return re.search(r"(?<!\w){}(?!\w)".format(re.escape(needle)), question) is not None


def has_script(text, lang):
    wanted = SCRIPT.get(lang)
    if not wanted:
        return True
    wanted = (wanted,) if isinstance(wanted, str) else wanted
    for char in text:
        if not char.isalpha():
            continue
        try:
            prefix = unicodedata.name(char).split()[0]
        except ValueError:
            continue
        if prefix in wanted:
            return True
    return False


def worksheet_fact_ids(path):
    with open(path, encoding="utf-8") as handle:
        return ["|".join(parts) for parts in ROW_RE.findall(handle.read())]


def load_verdicts(review_dir):
    with open(os.path.join(review_dir, "index.json"), encoding="utf-8") as handle:
        index = json.load(handle)
    verdicts = {}
    counts = Counter()
    for entry in index:
        with open(entry["out"], encoding="utf-8") as handle:
            rows = json.load(handle)
        expected = worksheet_fact_ids(entry["file"])
        if len(rows) != entry["n"] or len(expected) != entry["n"]:
            raise ValueError("row-count mismatch for {}".format(entry["out"]))
        got = [row.get("fact_id") for row in rows]
        subjects = [fact_id.split("|", 1)[0] for fact_id in expected]
        if got == subjects:
            # Five early files used the worksheet's displayed subject QID rather
            # than joining its adjacent subject/relation/object ID columns.
            for row, fact_id in zip(rows, expected):
                row["fact_id"] = fact_id
        elif got != expected:
            raise ValueError("fact_id/order mismatch for {}".format(entry["out"]))
        lang = entry["lang"] if entry["task"] == "translation" else "en"
        for row in rows:
            key = (row["fact_id"], lang)
            if key in verdicts:
                raise ValueError("duplicate verdict for {} / {}".format(*key))
            verdicts[key] = row["verdict"]
            counts[(lang, row["verdict"])] += 1
    return verdicts, counts


def load_proposals(retrans_dir):
    proposals = {}
    if not retrans_dir:
        return proposals
    for filename in sorted(os.listdir(retrans_dir)):
        match = re.match(r"retranslated_([a-z]{2})\.json$", filename)
        if not match:
            continue
        lang = match.group(1)
        with open(os.path.join(retrans_dir, filename), encoding="utf-8") as handle:
            for row in json.load(handle):
                question = row.get("question")
                if question:
                    key = (row["fact_id"], lang)
                    if key in proposals:
                        raise ValueError("duplicate proposal for {} / {}".format(*key))
                    proposals[key] = question.strip()
    return proposals


def put_column(table, name, values, arrow_type):
    array = pa.array(values, type=arrow_type)
    index = table.schema.get_field_index(name)
    if index < 0:
        return table.append_column(name, array)
    return table.set_column(index, name, array)


def ensure_inner_fields(inner):
    fields = list(inner)
    names = {field.name for field in fields}
    if "question_verified" not in names:
        fields.append(pa.field("question_verified", pa.bool_()))
    if "question_regenerated" not in names:
        fields.append(pa.field("question_regenerated", pa.bool_()))
    return pa.struct(fields)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--review_dir", required=True)
    parser.add_argument("--retrans_dir")
    parser.add_argument("--english_adjudication")
    parser.add_argument("--translation_adjudication")
    parser.add_argument("--report", default="results/full_census_apply_report.json")
    parser.add_argument("--classes", default="subject,type,relation,conflict")
    parser.add_argument("--target_split", default="test", choices=SPLITS,
                         help="which split the census/proposals/adjudications apply to; "
                              "every other split passes through unchanged")
    args = parser.parse_args()
    allowed_classes = {value.strip() for value in args.classes.split(",") if value.strip()}

    verdicts, verdict_counts = load_verdicts(args.review_dir)
    adjudications = []
    if args.english_adjudication:
        with open(args.english_adjudication, encoding="utf-8") as handle:
            adjudications = json.load(handle)
        if len({row["fact_id"] for row in adjudications}) != len(adjudications):
            raise ValueError("duplicate fact_id in English adjudication")
        allowed_actions = {"question_wrong", "gold_wrong", "not_conflict", "unsure"}
        for row in adjudications:
            if row.get("action") not in allowed_actions:
                raise ValueError("invalid English action: {!r}".format(row.get("action")))
            key = (row["fact_id"], "en")
            if verdicts.get(key) != "conflict":
                raise ValueError("adjudication is not a census conflict: {}".format(row["fact_id"]))
            if row["action"] == "not_conflict":
                verdicts[key] = "ok"
    translation_adjudications = []
    if args.translation_adjudication:
        with open(args.translation_adjudication, encoding="utf-8") as handle:
            translation_adjudications = json.load(handle)
        for row in translation_adjudications:
            if row.get("action") != "not_defect":
                raise ValueError("invalid translation adjudication action")
            key = (row["fact_id"], row["lang"])
            if verdicts.get(key) not in ("subject", "type", "relation"):
                raise ValueError("translation adjudication is not a census defect: {}".format(key))
            verdicts[key] = "ok"
    proposals = load_proposals(args.retrans_dir)
    if adjudications:
        expected_english = {
            row["fact_id"]: row["corrected_question"].strip()
            for row in adjudications if row["action"] == "question_wrong"
        }
        actual_english = {
            fact_id: question for (fact_id, lang), question in proposals.items()
            if lang == "en"
        }
        if actual_english != expected_english:
            missing = sorted(set(expected_english) - set(actual_english))
            extra = sorted(set(actual_english) - set(expected_english))
            changed = sorted(
                fact_id for fact_id in set(expected_english) & set(actual_english)
                if expected_english[fact_id] != actual_english[fact_id]
            )
            raise ValueError(
                "English proposals do not exactly match adjudication "
                "(missing={}, extra={}, changed={})".format(missing, extra, changed)
            )
    english_conflicts = {
        fact_id for (fact_id, lang), verdict in verdicts.items()
        if lang == "en" and verdict == "conflict"
    }
    print("loaded {:,} verdicts, {:,} proposals, {:,} English conflicts".format(
        len(verdicts), len(proposals), len(english_conflicts)
    ))

    def source(split, config):
        return os.path.join(args.data_dir, "data", config, "{}.parquet".format(split))

    test_rows = pq.read_table(source(args.target_split, "parallel")).to_pylist()
    known_proposal_keys = {
        (row["fact_id"], lang) for row in test_rows for lang in LANGS
    }
    unknown_proposals = sorted(set(proposals) - known_proposal_keys)
    if unknown_proposals:
        raise ValueError("proposals not present in {} data: {}".format(
            args.target_split, unknown_proposals))
    lengths = defaultdict(list)
    for row in test_rows:
        for lang in LANGS:
            lengths[lang].append(len(row["translations"][lang]["question"]))
    bounds = {
        lang: (
            max(8, int(0.35 * sum(values) / len(values))),
            int(3.0 * sum(values) / len(values)),
        )
        for lang, values in lengths.items()
    }

    accepted = {}
    rejected = []
    for row in test_rows:
        fact_id = row["fact_id"]
        for lang in LANGS:
            key = (fact_id, lang)
            proposed = proposals.get(key)
            if proposed is None:
                continue
            translation = row["translations"][lang]
            gold = translation["answer_text"]
            other_options = [translation[name] for name in OPTIONS if translation[name] != gold]
            verdict = verdicts.get(key)
            reason = None
            if fact_id in english_conflicts and lang != "en":
                reason = "English source conflicts with gold; target repair requires a corrected source"
            elif verdict not in allowed_classes:
                reason = "class not selected ({})".format(verdict)
            elif proposed == translation["question"]:
                reason = "unchanged"
            elif leaks(gold, proposed, lang):
                reason = "leakage: contains the gold answer"
            elif any(leaks(option, proposed, lang) for option in other_options):
                reason = "contains a distractor"
            elif not has_script(proposed, lang):
                reason = "wrong script"
            else:
                # Long proper-name subjects can exceed a language's ordinary
                # question-length ceiling before any interrogative wording is
                # added.  Keep the lower bound, but make the upper bound large
                # enough to name the subject plus a short question frame.
                maximum = max(bounds[lang][1], len(row["subject"]) + 20)
                if not (bounds[lang][0] <= len(proposed) <= maximum):
                    reason = "implausible length {} (expected {}-{})".format(
                        len(proposed), bounds[lang][0], maximum
                    )
            if reason:
                rejected.append(
                    {"fact_id": fact_id, "lang": lang, "reason": reason, "proposed": proposed[:200]}
                )
            else:
                accepted[key] = proposed

    def verification(fact_id, lang):
        if (fact_id, lang) in accepted:
            return True
        # A target-language `ok` verdict was measured against English.  When
        # English itself is false, that verdict cannot establish factual truth.
        if fact_id in english_conflicts and lang != "en":
            return False
        verdict = verdicts.get((fact_id, lang))
        if verdict in ("unsure", "vague") or verdict is None:
            return None
        return verdict == "ok"

    for config in LANGS + ["parallel"]:
        os.makedirs(os.path.join(args.out_dir, "data", config), exist_ok=True)

    for split in SPLITS:
        for lang in LANGS:
            table = pq.read_table(source(split, lang))
            fact_ids = table.column("fact_id").to_pylist()
            questions = table.column("question").to_pylist()
            old_regenerated = (
                table.column("question_regenerated").to_pylist()
                if "question_regenerated" in table.column_names
                else [False] * table.num_rows
            )
            old_verified = (
                table.column("question_verified").to_pylist()
                if "question_verified" in table.column_names
                else [None] * table.num_rows
            )
            verified = []
            regenerated = []
            for i, fact_id in enumerate(fact_ids):
                if split == args.target_split and (fact_id, lang) in accepted:
                    questions[i] = accepted[(fact_id, lang)]
                if split == args.target_split:
                    verified.append(verification(fact_id, lang))
                    regenerated.append(bool(old_regenerated[i] or (fact_id, lang) in accepted))
                else:
                    verified.append(old_verified[i])
                    regenerated.append(bool(old_regenerated[i]))
            table = put_column(table, "question", questions, pa.string())
            table = put_column(table, "question_verified", verified, pa.bool_())
            table = put_column(table, "question_regenerated", regenerated, pa.bool_())
            pq.write_table(
                table,
                os.path.join(args.out_dir, "data", lang, "{}.parquet".format(split)),
                compression="snappy",
            )

        parallel = pq.read_table(source(split, "parallel"))
        outer = parallel.schema.field("translations").type
        inner = ensure_inner_fields(outer.field(LANGS[0]).type)
        parallel_schema = pa.schema(
            [
                field if field.name != "translations" else pa.field(
                    "translations", pa.struct([pa.field(lang, inner) for lang in LANGS])
                )
                for field in parallel.schema
            ],
            metadata=parallel.schema.metadata,
        )
        rows = parallel.to_pylist()
        for row in rows:
            for lang in LANGS:
                translation = row["translations"][lang]
                key = (row["fact_id"], lang)
                if split == args.target_split and key in accepted:
                    translation["question"] = accepted[key]
                if split == args.target_split:
                    translation["question_verified"] = verification(row["fact_id"], lang)
                translation["question_regenerated"] = bool(
                    translation.get("question_regenerated", False) or
                    (split == args.target_split and key in accepted)
                )
        pq.write_table(
            pa.Table.from_pylist(rows, schema=parallel_schema),
            os.path.join(args.out_dir, "data", "parallel", "{}.parquet".format(split)),
            compression="snappy",
        )
        print("wrote {}".format(split))

    report = {
        "verdict_counts": {"{}:{}".format(*key): value for key, value in verdict_counts.items()},
        "effective_english_conflicts": len(english_conflicts),
        "english_conflicts": sorted(english_conflicts),
        "english_adjudication": adjudications,
        "translation_adjudication": translation_adjudications,
        "proposed": len(proposals),
        "accepted": [
            {"fact_id": fact_id, "lang": lang, "question": question}
            for (fact_id, lang), question in sorted(accepted.items())
        ],
        "rejected": rejected,
    }
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("accepted {:,}, rejected {:,}; wrote {}".format(
        len(accepted), len(rejected), args.report
    ))


if __name__ == "__main__":
    main()

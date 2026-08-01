#!/usr/bin/env python3
"""CPU-only final integrity verifier for PolyFact-Clean v13.

The verifier is deliberately read-only.  It checks the final 13-config parquet
tree against the pre-fix v13 tree and exits nonzero if any invariant fails.

Example:

    python build-poly-fact/verify_v13.py \
        --data_dir "$SCRATCH/v13-final" \
        --reference_dir "$SCRATCH/v13"
"""

import argparse
import os
import re
import sys
from collections import Counter

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - depends on the runtime environment
    print("ERROR: verify_v13.py requires pyarrow: {}".format(exc), file=sys.stderr)
    raise SystemExit(2)


LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
CONFIGS = LANGS + ["parallel"]
SPLITS = ["train", "validation", "test"]
EXPECTED_ROWS = {"train": 56324, "validation": 444, "test": 2523}
OPTIONS = ["option_a", "option_b", "option_c", "option_d"]
FLAGS = ["question_verified", "question_regenerated"]
NO_WORD_BOUNDARY = {"ja", "zh"}
MIN_GOLD_LEN = 3
MIN_GOLD_LEN_CJK = 2

FLAT_REQUIRED = [
    "fact_id", "language", "subject", "relation", "object", "question",
] + OPTIONS + [
    "answer_text", "answer_index", "option_ids", "n_langs_verified",
] + FLAGS

PARALLEL_REQUIRED = [
    "fact_id", "subject", "subject_id", "relation", "property_id", "object",
    "object_id", "translations", "n_langs_verified",
]

TRANSLATION_REQUIRED = [
    "question", "answer_text", "answer_index",
] + OPTIONS + ["option_ids"] + FLAGS

FLAT_METADATA_FIELDS = ["subject", "relation", "object", "n_langs_verified"]
FACT_ID_RE = re.compile(r"^Q[0-9]+\|P[0-9]+\|Q[0-9]+$")


def source(root, config, split):
    return os.path.join(root, "data", config, "{}.parquet".format(split))


def leaks(needle, question, lang):
    """Use the same answer-boundary rules as the retranslation gate."""
    floor = MIN_GOLD_LEN_CJK if lang in NO_WORD_BOUNDARY else MIN_GOLD_LEN
    if not needle or len(needle) < floor:
        return False
    needle = needle.casefold()
    question = question.casefold()
    if lang in NO_WORD_BOUNDARY:
        return needle in question
    return re.search(r"(?<!\w){}(?!\w)".format(re.escape(needle)), question) is not None


class Errors:
    """Count every failure while bounding terminal output."""

    def __init__(self, max_examples=80):
        self.counts = Counter()
        self.examples = []
        self.max_examples = max_examples

    @property
    def total(self):
        return sum(self.counts.values())

    def add(self, category, message):
        self.counts[category] += 1
        if len(self.examples) < self.max_examples:
            self.examples.append((category, message))

    def print_report(self):
        print("\nFAILED: {:,} verification error(s)".format(self.total), file=sys.stderr)
        for category, count in self.counts.most_common():
            print("  {:<30} {:>8,}".format(category, count), file=sys.stderr)
        if self.examples:
            print("\nExamples (first {}):".format(len(self.examples)), file=sys.stderr)
            for category, message in self.examples:
                print("  [{}] {}".format(category, message), file=sys.stderr)


def duplicate_names(fields):
    counts = Counter(field.name for field in fields)
    return sorted(name for name, count in counts.items() if count != 1)


def validate_no_duplicate_fields(fields, location, errors):
    """Check duplicate names recursively, including nested list structs."""
    for name in duplicate_names(fields):
        errors.add("duplicate schema field", "{} has duplicate field {!r}".format(location, name))
    for field in fields:
        arrow_type = field.type
        if pa.types.is_struct(arrow_type):
            validate_no_duplicate_fields(list(arrow_type), "{}.{}".format(location, field.name), errors)
        elif pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
            value_type = arrow_type.value_type
            if pa.types.is_struct(value_type):
                validate_no_duplicate_fields(
                    list(value_type), "{}.{}[]".format(location, field.name), errors
                )


def one_field(fields, name):
    matches = [field for field in fields if field.name == name]
    return matches[0] if len(matches) == 1 else None


def require_fields(fields, required, location, errors):
    names = Counter(field.name for field in fields)
    for name in required:
        if names[name] == 0:
            errors.add("missing schema field", "{} is missing {!r}".format(location, name))


def validate_flag_fields(fields, location, errors):
    for name in FLAGS:
        field = one_field(fields, name)
        if field is None:
            continue
        if not pa.types.is_boolean(field.type):
            errors.add(
                "flag schema",
                "{}.{} must be bool, found {}".format(location, name, field.type),
            )
        if not field.nullable:
            errors.add(
                "flag schema",
                "{}.{} must be nullable".format(location, name),
            )


def validate_option_ids_field(fields, location, errors):
    field = one_field(fields, "option_ids")
    if field is None:
        return
    if not (pa.types.is_list(field.type) or pa.types.is_large_list(field.type)):
        errors.add("option_ids schema", "{}.option_ids must be a list".format(location))
    elif not pa.types.is_string(field.type.value_type):
        errors.add(
            "option_ids schema",
            "{}.option_ids values must be strings, found {}".format(
                location, field.type.value_type
            ),
        )


def validate_flat_schema(schema, location, errors):
    fields = list(schema)
    validate_no_duplicate_fields(fields, location, errors)
    require_fields(fields, FLAT_REQUIRED, location, errors)
    validate_flag_fields(fields, location, errors)
    validate_option_ids_field(fields, location, errors)


def validate_parallel_schema(schema, location, errors):
    fields = list(schema)
    validate_no_duplicate_fields(fields, location, errors)
    require_fields(fields, PARALLEL_REQUIRED, location, errors)
    translations = one_field(fields, "translations")
    if translations is None or not pa.types.is_struct(translations.type):
        if translations is not None:
            errors.add("translations schema", "{}.translations must be a struct".format(location))
        return
    outer_fields = list(translations.type)
    outer_counts = Counter(field.name for field in outer_fields)
    for lang in LANGS:
        if outer_counts[lang] == 0:
            errors.add(
                "translations schema",
                "{}.translations is missing language {!r}".format(location, lang),
            )
    extras = sorted(set(outer_counts) - set(LANGS))
    if extras:
        errors.add(
            "translations schema",
            "{}.translations has unexpected languages {}".format(location, extras),
        )
    for lang in LANGS:
        lang_field = one_field(outer_fields, lang)
        if lang_field is None:
            continue
        if not pa.types.is_struct(lang_field.type):
            errors.add(
                "translations schema",
                "{}.translations.{} must be a struct".format(location, lang),
            )
            continue
        inner = list(lang_field.type)
        inner_location = "{}.translations.{}".format(location, lang)
        require_fields(inner, TRANSLATION_REQUIRED, inner_location, errors)
        validate_flag_fields(inner, inner_location, errors)
        validate_option_ids_field(inner, inner_location, errors)


class RowStream:
    """Take exact row counts despite differing parquet row-group boundaries."""

    def __init__(self, parquet_file, columns, batch_size=1024):
        self._iterator = parquet_file.iter_batches(batch_size=batch_size, columns=columns)
        self._rows = []
        self._offset = 0
        self._done = False

    def _refill(self):
        if self._done:
            return False
        try:
            self._rows = next(self._iterator).to_pylist()
            self._offset = 0
            return True
        except StopIteration:
            self._rows = []
            self._offset = 0
            self._done = True
            return False

    def take(self, count):
        result = []
        while len(result) < count:
            if self._offset >= len(self._rows) and not self._refill():
                break
            available = min(count - len(result), len(self._rows) - self._offset)
            result.extend(self._rows[self._offset:self._offset + available])
            self._offset += available
        return result

    def has_more(self):
        if self._offset < len(self._rows):
            return True
        return self._refill()


def load_reference_ids(reference_dir, split, errors):
    path = source(reference_dir, "parallel", split)
    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        errors.add("reference read", "cannot open {}: {}".format(path, exc))
        return set()
    ids = []
    try:
        for batch in parquet.iter_batches(batch_size=8192, columns=["fact_id"]):
            ids.extend(batch.column(0).to_pylist())
    except Exception as exc:
        errors.add("reference read", "cannot read fact_ids from {}: {}".format(path, exc))
        return set()
    if len(ids) != len(set(ids)):
        errors.add(
            "reference duplicate fact_id",
            "{} has {:,} duplicate row(s)".format(path, len(ids) - len(set(ids))),
        )
    return set(ids)


def short_values(values, limit=6):
    return repr(sorted(values)[:limit])


def check_translation(row, translation, lang, split, position, errors):
    fact_id = row.get("fact_id")
    where = "{}/{} row {:,} ({})".format(split, lang, position, fact_id)
    if not isinstance(translation, dict):
        errors.add("translation payload", "{} is not a struct/dict".format(where))
        return

    question = translation.get("question")
    answer = translation.get("answer_text")
    option_ids = translation.get("option_ids")
    answer_index = translation.get("answer_index")
    options = [translation.get(name) for name in OPTIONS]

    if not isinstance(question, str):
        errors.add("question value", "{} question is not a string".format(where))
    if not isinstance(answer, str):
        errors.add("answer value", "{} answer_text is not a string".format(where))
    if any(not isinstance(option, str) for option in options):
        errors.add("option value", "{} has a non-string option".format(where))

    ids_ok = isinstance(option_ids, list)
    if not ids_ok:
        errors.add("option_ids value", "{} option_ids is not a list".format(where))
    else:
        if len(option_ids) != len(OPTIONS):
            errors.add(
                "option_ids length",
                "{} has {} option_ids, expected {}".format(where, len(option_ids), len(OPTIONS)),
            )
        if len(set(option_ids)) != len(option_ids):
            errors.add("option_ids duplicate", "{} has duplicate option_ids".format(where))
        if any(not isinstance(value, str) or not value for value in option_ids):
            errors.add("option_ids value", "{} has a null/empty/non-string option_id".format(where))

    index_ok = type(answer_index) is int and 0 <= answer_index < len(OPTIONS)
    if not index_ok:
        errors.add(
            "answer_index",
            "{} has invalid answer_index {!r}".format(where, answer_index),
        )
    else:
        if isinstance(answer, str) and isinstance(options[answer_index], str):
            if answer != options[answer_index]:
                errors.add(
                    "answer_text mismatch",
                    "{} answer_text does not equal {}".format(where, OPTIONS[answer_index]),
                )
        if ids_ok and len(option_ids) == len(OPTIONS):
            if option_ids[answer_index] != row.get("object_id"):
                errors.add(
                    "object_id mismatch",
                    "{} option_ids[answer_index]={!r}, object_id={!r}".format(
                        where, option_ids[answer_index], row.get("object_id")
                    ),
                )

    if isinstance(question, str) and isinstance(answer, str):
        if leaks(answer, question, lang):
            errors.add("gold leakage", "{} question contains answer_text".format(where))
        for option_name, option in zip(OPTIONS, options):
            if option == answer or not isinstance(option, str):
                continue
            if leaks(option, question, lang):
                errors.add(
                    "distractor leakage",
                    "{} question contains {}".format(where, option_name),
                )


def compare_flat_row(flat, parallel, lang, split, position, errors):
    fact_id = parallel.get("fact_id")
    where = "{}/{} row {:,} ({})".format(split, lang, position, fact_id)
    if flat.get("fact_id") != fact_id:
        errors.add(
            "fact_id order",
            "{} flat={!r}, parallel={!r}".format(where, flat.get("fact_id"), fact_id),
        )
    if flat.get("language") != lang:
        errors.add(
            "language value",
            "{} has language={!r}".format(where, flat.get("language")),
        )
    for name in FLAT_METADATA_FIELDS:
        if flat.get(name) != parallel.get(name):
            errors.add(
                "flat metadata mismatch",
                "{} field {!r}: flat={!r}, parallel={!r}".format(
                    where, name, flat.get(name), parallel.get(name)
                ),
            )
    translations = parallel.get("translations") or {}
    translated = translations.get(lang)
    if not isinstance(translated, dict):
        errors.add("translation payload", "{} parallel translation is missing".format(where))
        return
    for name in TRANSLATION_REQUIRED:
        if flat.get(name) != translated.get(name):
            errors.add(
                "flat/parallel mismatch",
                "{} field {!r}: flat={!r}, parallel={!r}".format(
                    where, name, flat.get(name), translated.get(name)
                ),
            )


def check_parallel_row(row, split, position, errors):
    fact_id = row.get("fact_id")
    where = "parallel/{} row {:,} ({})".format(split, position, fact_id)
    if not isinstance(fact_id, str) or not FACT_ID_RE.match(fact_id):
        errors.add("fact_id format", "{} has malformed fact_id".format(where))
    else:
        encoded = "{}|{}|{}".format(
            row.get("subject_id"), row.get("property_id"), row.get("object_id")
        )
        if encoded != fact_id:
            errors.add(
                "fact_id components",
                "{} components encode {!r}".format(where, encoded),
            )
    translations = row.get("translations")
    if not isinstance(translations, dict):
        errors.add("translation payload", "{} translations is not a struct/dict".format(where))
        return
    for lang in LANGS:
        check_translation(row, translations.get(lang), lang, split, position, errors)


def verify_split(split, files, reference_ids, errors):
    parallel = files[("parallel", split)]
    flat_streams = {
        lang: RowStream(files[(lang, split)], FLAT_REQUIRED)
        for lang in LANGS
    }
    seen_parallel = set()
    seen_flat = {lang: set() for lang in LANGS}
    row_count = 0

    for batch in parallel.iter_batches(batch_size=512, columns=PARALLEL_REQUIRED):
        parallel_rows = batch.to_pylist()
        flat_chunks = {
            lang: flat_streams[lang].take(len(parallel_rows))
            for lang in LANGS
        }
        for lang, rows in flat_chunks.items():
            if len(rows) != len(parallel_rows):
                errors.add(
                    "row count",
                    "{}/{} ended at {:,}; parallel has more rows".format(split, lang, row_count),
                )
        usable = min([len(parallel_rows)] + [len(rows) for rows in flat_chunks.values()])
        for offset in range(usable):
            position = row_count + offset
            prow = parallel_rows[offset]
            fact_id = prow.get("fact_id")
            if fact_id in seen_parallel:
                errors.add(
                    "duplicate fact_id",
                    "parallel/{} duplicates {!r} at row {:,}".format(split, fact_id, position),
                )
            seen_parallel.add(fact_id)
            check_parallel_row(prow, split, position, errors)
            for lang in LANGS:
                frow = flat_chunks[lang][offset]
                flat_id = frow.get("fact_id")
                if flat_id in seen_flat[lang]:
                    errors.add(
                        "duplicate fact_id",
                        "{}/{} duplicates {!r} at row {:,}".format(lang, split, flat_id, position),
                    )
                seen_flat[lang].add(flat_id)
                compare_flat_row(frow, prow, lang, split, position, errors)
        row_count += len(parallel_rows)

    for lang, stream in flat_streams.items():
        if stream.has_more():
            errors.add(
                "row count",
                "{}/{} has rows after parallel ended at {:,}".format(lang, split, row_count),
            )

    if seen_parallel != reference_ids:
        missing = reference_ids - seen_parallel
        added = seen_parallel - reference_ids
        errors.add(
            "reference fact set",
            "{} differs from reference: {:,} missing {}, {:,} added {}".format(
                split, len(missing), short_values(missing), len(added), short_values(added)
            ),
        )
    return seen_parallel, row_count


def open_and_validate_files(data_dir, errors):
    files = {}
    for config in CONFIGS:
        for split in SPLITS:
            path = source(data_dir, config, split)
            if not os.path.isfile(path):
                errors.add("missing file", path)
                continue
            try:
                parquet = pq.ParquetFile(path)
            except Exception as exc:
                errors.add("parquet read", "cannot open {}: {}".format(path, exc))
                continue
            files[(config, split)] = parquet
            expected = EXPECTED_ROWS[split]
            actual = parquet.metadata.num_rows
            if actual != expected:
                errors.add(
                    "split size",
                    "{}/{} has {:,} rows, expected {:,}".format(config, split, actual, expected),
                )
            location = "{}/{}".format(config, split)
            if config == "parallel":
                validate_parallel_schema(parquet.schema_arrow, location, errors)
            else:
                validate_flat_schema(parquet.schema_arrow, location, errors)
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Verify a final PolyFact-Clean v13 parquet tree without modifying it."
    )
    parser.add_argument("--data_dir", required=True, help="final v13 directory")
    parser.add_argument(
        "--reference_dir", required=True, help="pre-fix v13 directory used for fact-set checks"
    )
    parser.add_argument(
        "--max_errors", type=int, default=80, help="maximum error examples to print"
    )
    args = parser.parse_args()

    errors = Errors(max_examples=max(1, args.max_errors))
    files = open_and_validate_files(args.data_dir, errors)
    expected_files = len(CONFIGS) * len(SPLITS)
    if len(files) != expected_files or errors.total:
        # Schema failures make content reads ambiguous (especially duplicate
        # fields), so fail before attempting the row-level pass.
        errors.print_report()
        return 1

    reference = {
        split: load_reference_ids(args.reference_dir, split, errors)
        for split in SPLITS
    }
    if errors.total:
        errors.print_report()
        return 1

    split_sets = {}
    checked_rows = 0
    for split in SPLITS:
        split_set, rows = verify_split(split, files, reference[split], errors)
        split_sets[split] = split_set
        checked_rows += rows
        print("checked {:<10} {:>6,} facts across 13 configs".format(split, rows))

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                errors.add(
                    "cross-split overlap",
                    "{} / {} share {:,} fact_id(s): {}".format(
                        left, right, len(overlap), short_values(overlap)
                    ),
                )

    if errors.total:
        errors.print_report()
        return 1

    translated_rows = checked_rows * len(LANGS)
    print("\nPASS: PolyFact-Clean v13 final build verified")
    print("  configs:              13 (12 flat + parallel)")
    print("  facts:                {:,}".format(checked_rows))
    print("  language rows checked:{:>10,}".format(translated_rows))
    print("  split sizes:          train={:,}, validation={:,}, test={:,}".format(
        EXPECTED_ROWS["train"], EXPECTED_ROWS["validation"], EXPECTED_ROWS["test"]
    ))
    print("  fact sets:            unchanged from reference; no duplicates or overlap")
    print("  options/answers:      consistent in every language")
    print("  flat/parallel:        exact match, including verification flags")
    print("  leakage:              0 gold, 0 distractor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

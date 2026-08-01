#!/usr/bin/env python3
"""Validate and aggregate the strict-prompt full test census.

The worksheet table stores the three components of a fact id in adjacent
columns.  A few early reviewers copied only the first (subject) component into
their JSON.  ``--repair-subject-only-ids`` upgrades that legacy representation
only when every row is otherwise in the worksheet's exact order.
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


VERDICTS = {
    "translation": {"ok", "subject", "type", "relation", "unsure"},
    "english_gold": {"ok", "conflict", "vague", "unsure"},
}
TRANSLATION_DEFECTS = {"subject", "type", "relation"}
ROW_RE = re.compile(
    r"^\|\s*\d+\s*\|\s*([^|\s]+)\s*\|\s*([^|\s]+)\s*\|\s*([^|\s]+)\s*\|",
    re.MULTILINE,
)


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def worksheet_fact_ids(path):
    text = path.read_text(encoding="utf-8")
    return ["|".join(parts) for parts in ROW_RE.findall(text)]


def validate_entry(entry, repair=False):
    worksheet = Path(entry["file"])
    verdict_file = Path(entry["out"])
    expected = worksheet_fact_ids(worksheet)
    errors = []
    changed = False
    if len(expected) != entry["n"]:
        errors.append(f"worksheet has {len(expected)} rows, index says {entry['n']}")
    if not verdict_file.exists():
        return None, ["missing verdict file"], False
    try:
        rows = json.loads(verdict_file.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, [f"unreadable JSON: {exc}"], False
    if not isinstance(rows, list):
        return None, ["top-level JSON value is not an array"], False
    if len(rows) != len(expected):
        errors.append(f"has {len(rows)} verdicts, expected {len(expected)}")

    if len(rows) == len(expected):
        got = [row.get("fact_id") if isinstance(row, dict) else None for row in rows]
        subjects = [fact_id.split("|", 1)[0] for fact_id in expected]
        if got != expected:
            if got == subjects and repair:
                for row, fact_id in zip(rows, expected):
                    row["fact_id"] = fact_id
                changed = True
            else:
                first = next(
                    (i for i, (actual, wanted) in enumerate(zip(got, expected), 1)
                     if actual != wanted),
                    None,
                )
                errors.append(f"fact_id/order mismatch, first at row {first}")

    allowed = VERDICTS[entry["task"]]
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            errors.append(f"row {i} is not an object")
            continue
        extra = set(row) - {"fact_id", "verdict", "note"}
        if extra:
            errors.append(f"row {i} has unexpected keys: {sorted(extra)}")
        verdict = row.get("verdict")
        if verdict not in allowed:
            errors.append(f"row {i} has invalid verdict {verdict!r}")
        if verdict != "ok" and not str(row.get("note", "")).strip():
            errors.append(f"row {i} non-ok verdict lacks a note")
    if changed and not errors:
        verdict_file.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return rows, errors, changed


def pct(value):
    return f"{100 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--out", default="results/full_test_census_report.md")
    parser.add_argument("--repair-subject-only-ids", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write a partial report instead of failing on missing/invalid files",
    )
    args = parser.parse_args()

    review_dir = Path(args.review_dir)
    entries = json.loads((review_dir / "index.json").read_text(encoding="utf-8"))
    problems = []
    repaired = []
    translation = defaultdict(Counter)
    english = Counter()
    defects = []
    conflicts = []
    reviewed_files = 0

    for entry in entries:
        rows, errors, changed = validate_entry(entry, args.repair_subject_only_ids)
        name = Path(entry["out"]).name
        if errors:
            problems.extend(f"{name}: {error}" for error in errors)
            continue
        reviewed_files += 1
        if changed:
            repaired.append(name)
        for row in rows:
            verdict = row["verdict"]
            if entry["task"] == "translation":
                translation[entry["lang"]][verdict] += 1
                if verdict in TRANSLATION_DEFECTS:
                    defects.append(
                        {
                            "fact_id": row["fact_id"],
                            "lang": entry["lang"],
                            "verdict": verdict,
                            "note": row.get("note", ""),
                        }
                    )
            else:
                english[verdict] += 1
                if verdict == "conflict":
                    conflicts.append(
                        {"fact_id": row["fact_id"], "note": row.get("note", "")}
                    )

    if problems and not args.allow_incomplete:
        print("Census validation failed:")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)

    lines = [
        "# Full strict-prompt test census",
        "",
        f"Validated verdict files: **{reviewed_files}/{len(entries)}**.",
        "",
    ]
    if repaired:
        lines += [
            "Mechanically expanded subject-only IDs to full triple IDs in: "
            + ", ".join(f"`{name}`" for name in repaired),
            "",
        ]
    if problems:
        lines += ["## Incomplete or invalid inputs", ""]
        lines.extend(f"- {problem}" for problem in problems)
        lines.append("")

    lines += [
        "## Translation review",
        "",
        "| language | reviewed | ok | subject | type | relation | unsure | defect rate | 95% Wilson CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    total = Counter()
    for lang in sorted(translation):
        counts = translation[lang]
        total.update(counts)
        n = sum(counts.values())
        bad = sum(counts[v] for v in TRANSLATION_DEFECTS)
        lo, hi = wilson(bad, n)
        lines.append(
            f"| {lang} | {n:,} | {counts['ok']:,} | {counts['subject']:,} | "
            f"{counts['type']:,} | {counts['relation']:,} | {counts['unsure']:,} | "
            f"{pct(bad / n)} | [{pct(lo)}, {pct(hi)}] |"
        )
    n = sum(total.values())
    bad = sum(total[v] for v in TRANSLATION_DEFECTS)
    if n:
        lo, hi = wilson(bad, n)
        lines.append(
            f"| **all** | **{n:,}** | **{total['ok']:,}** | **{total['subject']:,}** | "
            f"**{total['type']:,}** | **{total['relation']:,}** | **{total['unsure']:,}** | "
            f"**{pct(bad / n)}** | **[{pct(lo)}, {pct(hi)}]** |"
        )

    lines += ["", "## English-vs-gold review", ""]
    n_en = sum(english.values())
    if n_en:
        k = english["conflict"]
        lo, hi = wilson(k, n_en)
        lines += [
            "| verdict | count | share |",
            "|---|---:|---:|",
        ]
        for verdict in ("ok", "conflict", "vague", "unsure"):
            lines.append(f"| {verdict} | {english[verdict]:,} | {pct(english[verdict] / n_en)} |")
        lines += [
            "",
            f"Conflict rate: **{pct(k / n_en)}** (95% Wilson CI [{pct(lo)}, {pct(hi)}]).",
            "",
        ]

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    data_path = output.with_name(output.stem + "_data.json")
    data_path.write_text(
        json.dumps(
            {
                "reviewed_files": reviewed_files,
                "expected_files": len(entries),
                "problems": problems,
                "translation": {lang: dict(counts) for lang, counts in translation.items()},
                "english": dict(english),
                "translation_defects": defects,
                "english_conflicts": conflicts,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output} and {data_path}")


if __name__ == "__main__":
    main()

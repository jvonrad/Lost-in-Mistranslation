"""Convert jvonrad/WIKI-FACT into per-language + parallel configs and reupload.

Per-language configs (one per language): flat schema, one row per (fact, language).
Parallel config: one row per fact with a `translations` dict keyed by language code.

Usage:
    1. Paste your HF token into `login("")` below.
    2. Set TARGET to your chosen repo name (e.g. "jvonrad/mFacts-12").
    3. python data_analysis/convert_wikifact_dataset.py
"""

from collections import Counter
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, create_repo, login

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
login("")  # ← paste HF token with write access

SOURCE = "jvonrad/WIKI-FACT"
TARGET = "jvonrad/PolyFact"  # ← rename if you pick a different name
LICENSE = "cc-by-sa-4.0"       # ← adjust if needed
SPLITS = ["train", "validation", "test"]
OUT = Path("./PolyFact")

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def explode(options, answer_text):
    a, b, c, d = options
    try:
        idx = options.index(answer_text)
    except ValueError:
        idx = -1
    return a, b, c, d, idx


def valid_item(item):
    if not isinstance(item, dict):
        return False
    opts = item.get("options")
    if not isinstance(opts, list) or len(opts) != 4:
        return False
    if not all(isinstance(o, str) and o for o in opts):
        return False
    if not isinstance(item.get("question"), str) or not item["question"]:
        return False
    if item.get("answer_text") not in opts:
        return False
    return True


def per_lang_row(ex, lang, item):
    a, b, c, d, idx = explode(item["options"], item["answer_text"])
    return {
        "fact_id": ex["fact_id"],
        "language": lang,
        "subject": ex["subject"],
        "relation": ex["relation"],
        "object": ex["object"],
        "question": item["question"],
        "option_a": a,
        "option_b": b,
        "option_c": c,
        "option_d": d,
        "answer_text": item["answer_text"],
        "answer_index": idx,
    }


def parallel_row(ex):
    translations = {}
    for lang, item in ex["langs"].items():
        if not valid_item(item):
            continue
        a, b, c, d, idx = explode(item["options"], item["answer_text"])
        translations[lang] = {
            "question": item["question"],
            "option_a": a,
            "option_b": b,
            "option_c": c,
            "option_d": d,
            "answer_text": item["answer_text"],
            "answer_index": idx,
        }
    return {
        "fact_id": ex["fact_id"],
        "subject": ex["subject"],
        "subject_id": ex["subject_id"],
        "relation": ex["relation"],
        "property_id": ex["property_id"],
        "object": ex["object"],
        "object_id": ex["object_id"],
        "translations": translations,
    }


def detect_languages(raw):
    counter = Counter()
    for split in SPLITS:
        for ex in raw[split]:
            for lang in ex["langs"].keys():
                counter[lang] += 1
    langs = sorted(counter.keys())
    print(f"Detected {len(langs)} languages: {langs}")
    return langs


# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Loading {SOURCE} ...")
    raw = load_dataset(SOURCE)

    langs = detect_languages(raw)

    # Per-language parquet files
    for lang in langs:
        lang_dir = OUT / "data" / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            rows = []
            for ex in raw[split]:
                item = ex["langs"].get(lang)
                if item is not None and valid_item(item):
                    rows.append(per_lang_row(ex, lang, item))
            Dataset.from_list(rows).to_parquet(str(lang_dir / f"{split}.parquet"))
            print(f"  {lang}/{split}: {len(rows)} rows")

    # Parallel parquet files
    par_dir = OUT / "data" / "parallel"
    par_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        rows = [parallel_row(ex) for ex in raw[split]]
        Dataset.from_list(rows).to_parquet(str(par_dir / f"{split}.parquet"))
        print(f"  parallel/{split}: {len(rows)} rows")

    # README with config declarations
    lang_yaml = "\n".join(f"  - {l}" for l in langs)
    lang_blocks = "\n".join(
        f"  - config_name: {l}\n"
        f"    data_files:\n"
        f"      - split: train\n        path: data/{l}/train.parquet\n"
        f"      - split: validation\n        path: data/{l}/validation.parquet\n"
        f"      - split: test\n        path: data/{l}/test.parquet"
        for l in langs
    )
    repo_short = TARGET.split("/")[-1]
    readme = f"""---
license: {LICENSE}
task_categories:
  - multiple-choice
  - question-answering
language:
{lang_yaml}
configs:
{lang_blocks}
  - config_name: parallel
    data_files:
      - split: train
        path: data/parallel/train.parquet
      - split: validation
        path: data/parallel/validation.parquet
      - split: test
        path: data/parallel/test.parquet
---

# {repo_short}

Parallel multilingual factual multiple-choice QA grounded in Wikidata. 100K facts × {len(langs)} languages, fully aligned by `fact_id` across all per-language configs.

## Usage

```python
from datasets import load_dataset

# One language at a time (SFT / eval)
ds = load_dataset("{TARGET}", "en")
print(ds["train"][0])

# All languages aligned per fact (cross-lingual training)
par = load_dataset("{TARGET}", "parallel")
print(par["train"][0]["translations"]["en"])
```

## Schema

**Per-language configs** ({", ".join(f"`{l}`" for l in langs)}) — flat, one row per (fact, language):

| Column | Type | Description |
|---|---|---|
| `fact_id` | string | Cross-language join key — `<subject_qid>\\|<property_pid>\\|<object_qid>` |
| `language` | string | ISO language code |
| `subject` | string | Subject entity label |
| `relation` | string | Relation name (e.g. `educated at`) |
| `object` | string | Object entity label (the canonical answer) |
| `question` | string | Natural-language question in `language` |
| `option_a`..`option_d` | string | Four multiple-choice options |
| `answer_text` | string | The correct option as text (matches one of `option_a`..`option_d`) |
| `answer_index` | int | 0-based index of the correct option |

For Wikidata Q-/P-ids (`subject_id`, `property_id`, `object_id`), load the `parallel` config and join on `fact_id`.

**`parallel` config** — one row per fact:

| Column | Type | Description |
|---|---|---|
| `fact_id`, `subject`, `subject_id`, `relation`, `property_id`, `object`, `object_id` | — | Shared across languages; Wikidata grounding lives here |
| `translations` | dict | `{{lang_code: {{question, option_a..d, answer_text, answer_index}}}}` |

Splits are parallel across languages: every `fact_id` in a split is present in all per-language configs.
"""
    (OUT / "README.md").write_text(readme)
    print(f"Wrote {OUT / 'README.md'}")

    # Upload
    print(f"Creating repo {TARGET} ...")
    create_repo(TARGET, repo_type="dataset", exist_ok=True)

    print(f"Uploading {OUT} ...")
    api = HfApi()
    api.upload_folder(folder_path=str(OUT), repo_id=TARGET, repo_type="dataset")

    print(f"Done. https://huggingface.co/datasets/{TARGET}")


if __name__ == "__main__":
    main()

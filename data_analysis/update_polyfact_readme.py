"""Update only the README of jvonrad/PolyFact to reflect the schema changes
(per-language configs no longer carry Q/P-ids).

Usage:
    1. Paste your HF token into `login("")` below.
    2. python data_analysis/update_polyfact_readme.py
"""

from huggingface_hub import HfApi, login

HF_TOKEN = os.environ.get("HF_TOKEN")

TARGET = "jvonrad/PolyFact"
LICENSE = "cc-by-sa-4.0"
LANGS = ["ar", "bn", "de", "en", "es", "fr", "id", "ja", "pt", "ru", "sw", "zh"]

lang_yaml = "\n".join(f"  - {l}" for l in LANGS)
lang_blocks = "\n".join(
    f"  - config_name: {l}\n"
    f"    data_files:\n"
    f"      - split: train\n        path: data/{l}/train.parquet\n"
    f"      - split: validation\n        path: data/{l}/validation.parquet\n"
    f"      - split: test\n        path: data/{l}/test.parquet"
    for l in LANGS
)
repo_short = TARGET.split("/")[-1]

USAGE_BLOCK = (
    "```python\n"
    "from datasets import load_dataset\n\n"
    f'ds = load_dataset("{TARGET}", "en")\n'
    'print(ds["train"][0])\n\n'
    f'par = load_dataset("{TARGET}", "parallel")\n'
    'print(par["train"][0]["translations"]["en"])\n'
    "```"
)

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

Parallel multilingual factual multiple-choice QA grounded in Wikidata. 100K facts x {len(LANGS)} languages, fully aligned by `fact_id` across all per-language configs.

## Usage

{USAGE_BLOCK}

## Schema

**Per-language configs** ({", ".join(f"`{l}`" for l in LANGS)}) - flat, one row per (fact, language):

| Column | Type | Description |
|---|---|---|
| `fact_id` | string | Cross-language join key - `<subject_qid>\\|<property_pid>\\|<object_qid>` |
| `language` | string | ISO language code |
| `subject` | string | Subject entity label |
| `relation` | string | Relation name (e.g. `educated at`) |
| `object` | string | Object entity label (the canonical answer) |
| `question` | string | Natural-language question in `language` |
| `option_a`..`option_d` | string | Four multiple-choice options |
| `answer_text` | string | The correct option as text (matches one of `option_a`..`option_d`) |
| `answer_index` | int | 0-based index of the correct option |

For Wikidata Q-/P-ids (`subject_id`, `property_id`, `object_id`), load the `parallel` config and join on `fact_id`.

**`parallel` config** - one row per fact:

| Column | Type | Description |
|---|---|---|
| `fact_id`, `subject`, `subject_id`, `relation`, `property_id`, `object`, `object_id` | - | Shared across languages; Wikidata grounding lives here |
| `translations` | dict | `{{lang_code: {{question, option_a..d, answer_text, answer_index}}}}` |

Splits are parallel across languages: every `fact_id` in a split is present in all per-language configs.
"""

HfApi().upload_file(
    path_or_fileobj=readme.encode("utf-8"),
    path_in_repo="README.md",
    repo_id=TARGET,
    repo_type="dataset",
    commit_message="Update README: drop Q/P-ids from per-language schema",
)
print(f"Updated README at https://huggingface.co/datasets/{TARGET}")

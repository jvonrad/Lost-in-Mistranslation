#!/usr/bin/env python3
"""
Run an OpenAI model as the LLM judge over the sampled PolyFact items and
fill in each item's `llm_eval` field. Resumable: items that already have
`llm_eval.label` set are skipped.

The judge rubric must stay in sync with judge_prompt.md (the prompts are
duplicated here so the script is self-contained).

Usage:
    export OPENAI_API_KEY=...
    python run_llm_judge.py
    python run_llm_judge.py --model gpt-4o --langs en fr --workers 8
    python run_llm_judge.py --redo   # ignore existing llm_eval and re-label all
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

SAMPLES_DIR = Path(__file__).parent / "samples"
DEFAULT_LANGS = ["en", "fr", "es", "ar", "zh", "de", "ru"]

SYSTEM_PROMPT = """\
You are a careful multilingual fact-checking judge. You will be shown a
multiple-choice question, four options (A-D), and the option that the dataset
labels as the correct answer. Your job is to decide whether that labeling is
correct, ambiguous, or incorrect, using the rubric below.

Rubric:
  A = Correct: the marked answer is unambiguously right and no other option
      is also valid; the question is well-formed.
  B = Ambiguous / underspecified: multiple options could be valid, the
      question is missing context needed to pick one, or the wording is
      problematic enough that a competent reader would hesitate.
  C = Incorrect: the marked answer is factually wrong, the answer is not
      among the options, or the item is nonsensical.

Tie-breakers: A vs B -> choose B. B vs C -> choose B unless plainly wrong.
If you do not know the underlying fact and cannot reason about it from the
options, return B and say so in the rationale. Do not guess A.

The question may be in a non-English language. Evaluate it in that language;
do not penalize an item just because it is not in English. However, if the
wording is unnatural or grammatically broken in that language to a degree
that affects answerability, that is grounds for B.

Respond with the requested JSON object only."""

USER_TEMPLATE = """\
Language: {language}
Question: {question}

Options:
  A) {option_a}
  B) {option_b}
  C) {option_c}
  D) {option_d}

Dataset-labeled correct answer: {answer_text}

Apply the rubric and respond with JSON."""


class JudgeVerdict(BaseModel):
    label: Literal["A", "B", "C"] = Field(
        description="A=Correct, B=Ambiguous/underspecified, C=Incorrect"
    )
    rationale: str = Field(description="One or two sentences in English.")


def build_user_prompt(item: dict) -> str:
    return USER_TEMPLATE.format(
        language=item["language"],
        question=item["question"],
        option_a=item["options"]["A"],
        option_b=item["options"]["B"],
        option_c=item["options"]["C"],
        option_d=item["options"]["D"],
        answer_text=item["answer_text"],
    )


def judge_item(client: OpenAI, model: str, item: dict) -> JudgeVerdict:
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(item)},
        ],
        response_format=JudgeVerdict,
        temperature=0,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        refusal = completion.choices[0].message.refusal
        raise RuntimeError(f"model returned no parsed output (refusal={refusal!r})")
    return parsed


def process_file(
    client: OpenAI,
    path: Path,
    model: str,
    workers: int,
    redo: bool,
    save_every: int,
):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data["items"]
    todo_indices = [
        i for i, it in enumerate(items)
        if redo or it.get("llm_eval", {}).get("label") is None
    ]
    if not todo_indices:
        print(f"  {path.name}: nothing to do ({len(items)} items already labeled)")
        return

    print(f"  {path.name}: judging {len(todo_indices)}/{len(items)} items "
          f"with {workers} workers")

    write_lock = threading.Lock()
    completed = 0
    errors = 0

    def work(idx: int):
        try:
            verdict = judge_item(client, model, items[idx])
            items[idx]["llm_eval"] = {
                "label": verdict.label,
                "rationale": verdict.rationale,
                "model": model,
            }
            return idx, None
        except Exception as e:
            return idx, e

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, idx) for idx in todo_indices]
        for fut in as_completed(futures):
            idx, err = fut.result()
            completed += 1
            if err is not None:
                errors += 1
                print(f"    [{completed}/{len(todo_indices)}] item {idx} ERROR: {err}",
                      file=sys.stderr)
            elif completed % 10 == 0 or completed == len(todo_indices):
                print(f"    [{completed}/{len(todo_indices)}] last label="
                      f"{items[idx]['llm_eval']['label']}")
            if completed % save_every == 0:
                with write_lock:
                    _atomic_write(path, data)

    _atomic_write(path, data)
    print(f"  {path.name}: done. errors={errors}")


def _atomic_write(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--langs", nargs="+", default=DEFAULT_LANGS)
    ap.add_argument("--workers", type=int, default=2,
                    help="Concurrent requests. Tier-1 (30K TPM) typically tolerates 2-3.")
    ap.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR)
    ap.add_argument("--redo", action="store_true",
                    help="Re-label items even if llm_eval.label is already set")
    ap.add_argument("--save-every", type=int, default=20,
                    help="Checkpoint to disk every N completed items per file")
    ap.add_argument("--max-retries", type=int, default=10,
                    help="OpenAI client retry count for 429 / 5xx. SDK uses exponential "
                         "backoff respecting Retry-After.")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("error: OPENAI_API_KEY is not set")

    client = OpenAI(max_retries=args.max_retries)

    for lang in args.langs:
        path = args.samples_dir / f"polyfact_eval_{lang}.json"
        if not path.exists():
            print(f"skip {lang}: {path} does not exist", file=sys.stderr)
            continue
        process_file(
            client,
            path,
            model=args.model,
            workers=args.workers,
            redo=args.redo,
            save_every=args.save_every,
        )


if __name__ == "__main__":
    main()

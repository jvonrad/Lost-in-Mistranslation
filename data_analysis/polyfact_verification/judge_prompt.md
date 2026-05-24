# PolyFact Verification — LLM Judge Prompt

This prompt is used to label PolyFact multiple-choice QA pairs as
**A) Correct**, **B) Ambiguous / underspecified**, or **C) Incorrect**.
The LLM judge should reach the same conclusion a careful human annotator would, so the labeling rubric below is shared between the two.

The runner script ([run_llm_judge.py](run_llm_judge.py)) sends these prompts to OpenAI (default `gpt-4o`) using structured outputs (Pydantic schema → guaranteed `A` / `B` / `C` labels). The prompt strings in `run_llm_judge.py` must stay in sync with this file.

## Label rubric (applies to both human and LLM)

Given a `question`, four `options` (A–D), and an `answer_text` marked as correct, decide:

- **A — Correct**
  The marked answer is unambiguously the right answer to the question, and no other option is also a valid answer. The question is well-formed and uniquely specified.

- **B — Ambiguous / underspecified**
  The marked answer is plausible, but the item is not cleanly correct because *at least one* of the following holds:
  - More than one option could reasonably be the right answer (multiple options are true of the subject).
  - The question is underspecified — it doesn't pin down a single relation, time period, or context, so the "correct" answer depends on an unstated assumption.
  - The question is grammatically or semantically awkward to the point that a competent native speaker would have trouble committing to a single answer.
  - The marked answer is technically correct but stylistically off (e.g., a transliteration vs. local-language form, an alternate name) in a way that makes it unclear whether the item is testing facts or surface form.

- **C — Incorrect**
  The marked answer is factually wrong, OR the question/options are nonsensical, OR the answer doesn't appear among the options, OR the marked answer contradicts widely accepted knowledge.

When in doubt between A and B, choose **B**. When in doubt between B and C, choose **B** unless the item is plainly wrong.

You may rely on widely accepted general knowledge. If you genuinely don't know the fact and cannot reason about it, return **B** with a rationale that says so — do not guess **A**.

## LLM judge — system prompt

```
You are a careful multilingual fact-checking judge. You will be shown a
multiple-choice question, four options (A–D), and the option that the dataset
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

Output STRICT JSON with exactly these keys and nothing else:
{
  "label": "A" | "B" | "C",
  "rationale": "<one or two sentences, in English>"
}
```

## LLM judge — user prompt template

```
Language: {language}
Question: {question}

Options:
  A) {option_a}
  B) {option_b}
  C) {option_c}
  D) {option_d}

Dataset-labeled correct answer: {answer_text}

Apply the rubric and respond with strict JSON only.
```

## Notes for the human annotator

- Look at the same fields the LLM sees: question, four options, and the dataset-labeled answer.
- Fill in `human_eval.label` with `"A"`, `"B"`, or `"C"`.
- Use `human_eval.notes` for anything that helped you decide — especially for B labels, name the source of ambiguity (e.g., "two options are both alma maters", "transliteration mismatch").
- Don't look at `llm_eval` while you label; we want independent judgments to compute agreement.

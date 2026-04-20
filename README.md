# Lost in Multilinguality: Unlocking Latent Multilingual Knowledge via Consistency-Driven Reinforcement Learning

> **Louis Arts, George Burgess, Eleftheria Kolokytha, Harry O'Donnell, Ektor Oikonomidis Doumpas, Jonathan von Rad**
> University College London

![Two-stage pipeline](./main-figure.png)

## Overview

LLMs trained on English data encode vast world knowledge but often fail to express it in other languages — **cross-lingual factual inconsistency**. We propose a two-stage pipeline to fix this without large-scale retraining:

1. **Light Continual Pretraining (CPT)** on 1B tokens of parallel data
2. **Consistency-Driven RL (GRPO)** to reshape internal representations for consistent cross-lingual factual recall

Applied to OLMo-2-7B across the 12 most widely spoken languages (18.5% → 70% of the global population). Code and dataset open-sourced.

## Results

| Model | WIKI-FACT (High/Low) | KLAR (Seen/OOD) | Global-MMLU (High/Low) |
|---|---|---|---|
| Baseline | 57.93 / 51.80 | 24.6 / 13.3 | 38.72 / 31.79 |
| SFT | 56.33 / 50.04 | 18.1 / 7.8 | 35.40 / 30.32 |
| **GRPO** | **60.71 / 54.41** | **29.0 / 16.7** | **39.22 / 32.00** |
| Aligned + GRPO | 61.26 / 54.48 | 29.8 / 17.6 | 36.34 / 29.61 |

GRPO outperforms SFT across all benchmarks and transfers to **11 unseen languages** not seen during training.

## Dataset: WIKI-FACT

🤗 [`jvonrad/WIKI-FACT`](https://huggingface.co/datasets/jvonrad/WIKI-FACT) — 100K facts × 12 languages, fully parallel, grounded in Wikidata. Split: 95K train / 2.5K val / 2.5K test.

## Installation

```bash
git clone https://github.com/jvonrad/Lost-in-Mistranslation
cd Lost-in-Mistranslation
pip install -r requirements.txt
```

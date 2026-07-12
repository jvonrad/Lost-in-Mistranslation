# Bootstrap CIs and paired significance tests

Recomputed per fact from the raw per-option scores saved in `results/*_consistency.json`; every recomputed aggregate was validated against the reported value (tolerance 5e-3) before bootstrapping.

## Global-MMLU-Lite (11 languages, 400 parallel facts)

Point estimates with 95% percentile bootstrap CIs (10,000 resamples of facts; facts resampled as units so all languages of a fact stay together). All values in %.

| Family | Model | Avg accuracy | Total consistency | RankC | Answer agreement |
|---|---|---|---|---|---|
| OLMo | Base | 44.75 [41.93, 47.59] | 3.50 [1.75, 5.50] | 55.12 [53.71, 56.58] | 46.48 [44.56, 48.50] |
| OLMo | CPT | 37.70 [35.00, 40.39] | 0.75 [0.00, 1.75] | 52.79 [51.48, 54.07] | 42.73 [41.01, 44.48] |
| OLMo | SFT | 44.05 [41.09, 46.98] | 3.25 [1.50, 5.01] | 57.99 [56.54, 59.48] | 49.79 [47.80, 51.85] |
| OLMo | CPT+SFT | 42.36 [39.43, 45.36] | 4.00 [2.25, 6.00] | 58.26 [56.77, 59.74] | 50.03 [48.00, 52.07] |
| OLMo | GRPO | 44.05 [41.05, 47.09] | 4.75 [2.75, 7.00] | 58.44 [56.89, 60.01] | 50.65 [48.51, 52.86] |
| OLMo | CPT+GRPO | 43.86 [40.91, 46.77] | 3.00 [1.50, 4.75] | 58.03 [56.59, 59.53] | 49.66 [47.70, 51.71] |
| Qwen | Base | 63.20 [59.95, 66.43] | 13.50 [10.25, 17.00] | 69.70 [68.18, 71.30] | 64.49 [62.33, 66.75] |
| Qwen | CPT | 60.61 [57.36, 63.89] | 13.25 [10.00, 16.75] | 68.28 [66.74, 69.84] | 62.85 [60.67, 65.13] |
| Qwen | SFT | 61.05 [57.86, 64.07] | 9.00 [6.25, 12.00] | 67.63 [66.11, 69.14] | 61.34 [59.20, 63.49] |
| Qwen | CPT+SFT | 53.82 [50.75, 56.89] | 5.00 [3.00, 7.25] | 62.08 [60.56, 63.63] | 54.91 [52.82, 57.06] |
| Qwen | GRPO | 63.11 [59.93, 66.25] | 11.25 [8.25, 14.50] | 68.60 [67.09, 70.19] | 63.11 [60.99, 65.33] |
| Qwen | CPT+GRPO | 61.18 [57.98, 64.39] | 11.75 [8.75, 15.00] | 68.62 [67.11, 70.15] | 63.24 [61.08, 65.45] |

**Paired deltas** (same bootstrap fact indices for both models; two-sided bootstrap p-value):

| Family | Comparison | Metric | Δ (pp) | 95% CI | p |
|---|---|---|---|---|---|
| OLMo | SFT − Base | Avg accuracy | -0.70 | [-3.43, +2.02] | 0.6068 |
| OLMo | SFT − Base | Total consistency | -0.25 | [-2.50, +2.00] | 0.9052 |
| OLMo | SFT − Base | RankC | +2.86 * | [+1.22, +4.44] | 0.0004 |
| OLMo | SFT − Base | Answer agreement | +3.31 * | [+1.01, +5.55] | 0.0044 |
| OLMo | CPT − Base | Avg accuracy | -7.05 * | [-9.16, -4.93] | 0.0001 |
| OLMo | CPT − Base | Total consistency | -2.75 * | [-4.50, -1.00] | 0.0014 |
| OLMo | CPT − Base | RankC | -2.34 * | [-3.74, -0.97] | 0.0010 |
| OLMo | CPT − Base | Answer agreement | -3.75 * | [-5.70, -1.84] | 0.0002 |
| OLMo | GRPO − Base | Avg accuracy | -0.70 | [-1.75, +0.36] | 0.1948 |
| OLMo | GRPO − Base | Total consistency | +1.25 | [-0.25, +3.00] | 0.1582 |
| OLMo | GRPO − Base | RankC | +3.32 * | [+2.48, +4.19] | 0.0001 |
| OLMo | GRPO − Base | Answer agreement | +4.17 * | [+2.91, +5.46] | 0.0001 |
| OLMo | CPT+SFT − Base | Avg accuracy | -2.39 | [-5.16, +0.39] | 0.0936 |
| OLMo | CPT+SFT − Base | Total consistency | +0.50 | [-2.00, +3.00] | 0.7744 |
| OLMo | CPT+SFT − Base | RankC | +3.14 * | [+1.56, +4.69] | 0.0001 |
| OLMo | CPT+SFT − Base | Answer agreement | +3.55 * | [+1.26, +5.75] | 0.0020 |
| OLMo | CPT+GRPO − Base | Avg accuracy | -0.89 | [-3.05, +1.25] | 0.4214 |
| OLMo | CPT+GRPO − Base | Total consistency | -0.50 | [-2.25, +1.25] | 0.6332 |
| OLMo | CPT+GRPO − Base | RankC | +2.91 * | [+1.54, +4.23] | 0.0001 |
| OLMo | CPT+GRPO − Base | Answer agreement | +3.18 * | [+1.24, +5.05] | 0.0006 |
| OLMo | GRPO − SFT | Avg accuracy | +0.00 | [-2.73, +2.73] | 0.9908 |
| OLMo | GRPO − SFT | Total consistency | +1.50 | [-0.75, +3.75] | 0.2344 |
| OLMo | GRPO − SFT | RankC | +0.45 | [-1.21, +2.16] | 0.6004 |
| OLMo | GRPO − SFT | Answer agreement | +0.86 | [-1.55, +3.31] | 0.4814 |
| Qwen | SFT − Base | Avg accuracy | -2.16 * | [-4.18, -0.20] | 0.0324 |
| Qwen | SFT − Base | Total consistency | -4.50 * | [-7.75, -1.25] | 0.0076 |
| Qwen | SFT − Base | RankC | -2.07 * | [-3.42, -0.76] | 0.0022 |
| Qwen | SFT − Base | Answer agreement | -3.15 * | [-5.10, -1.24] | 0.0012 |
| Qwen | CPT − Base | Avg accuracy | -2.59 * | [-4.82, -0.43] | 0.0214 |
| Qwen | CPT − Base | Total consistency | -0.25 | [-4.00, +3.25] | 0.9692 |
| Qwen | CPT − Base | RankC | -1.42 | [-2.86, +0.05] | 0.0586 |
| Qwen | CPT − Base | Answer agreement | -1.64 | [-3.79, +0.52] | 0.1300 |
| Qwen | GRPO − Base | Avg accuracy | -0.09 | [-1.23, +1.09] | 0.8928 |
| Qwen | GRPO − Base | Total consistency | -2.25 | [-4.75, +0.00] | 0.0728 |
| Qwen | GRPO − Base | RankC | -1.10 * | [-1.98, -0.23] | 0.0124 |
| Qwen | GRPO − Base | Answer agreement | -1.38 * | [-2.65, -0.11] | 0.0334 |
| Qwen | CPT+SFT − Base | Avg accuracy | -9.39 * | [-12.00, -6.80] | 0.0001 |
| Qwen | CPT+SFT − Base | Total consistency | -8.50 * | [-12.00, -5.25] | 0.0001 |
| Qwen | CPT+SFT − Base | RankC | -7.62 * | [-9.16, -6.10] | 0.0001 |
| Qwen | CPT+SFT − Base | Answer agreement | -9.58 * | [-11.80, -7.35] | 0.0001 |
| Qwen | CPT+GRPO − Base | Avg accuracy | -2.02 * | [-4.05, -0.09] | 0.0394 |
| Qwen | CPT+GRPO − Base | Total consistency | -1.75 | [-5.25, +1.75] | 0.3704 |
| Qwen | CPT+GRPO − Base | RankC | -1.08 | [-2.39, +0.27] | 0.1118 |
| Qwen | CPT+GRPO − Base | Answer agreement | -1.25 | [-3.18, +0.71] | 0.2066 |
| Qwen | GRPO − SFT | Avg accuracy | +2.07 * | [+0.43, +3.73] | 0.0140 |
| Qwen | GRPO − SFT | Total consistency | +2.25 | [-0.75, +5.25] | 0.1596 |
| Qwen | GRPO − SFT | RankC | +0.97 | [-0.15, +2.14] | 0.0948 |
| Qwen | GRPO − SFT | Answer agreement | +1.77 * | [+0.11, +3.45] | 0.0358 |

`*` = 95% CI excludes zero.


## PolyFact (test) (12 languages, 2523 parallel facts)

Point estimates with 95% percentile bootstrap CIs (10,000 resamples of facts; facts resampled as units so all languages of a fact stay together). All values in %.

| Family | Model | Avg accuracy | Total consistency | RankC | Answer agreement |
|---|---|---|---|---|---|
| OLMo | Base | 57.96 [56.89, 59.00] | 7.21 [6.22, 8.24] | 58.32 [57.71, 58.91] | 51.10 [50.25, 51.94] |
| OLMo | CPT | 47.91 [46.89, 48.90] | 2.93 [2.30, 3.61] | 53.39 [52.88, 53.90] | 44.27 [43.57, 44.98] |
| OLMo | SFT | 60.21 [59.16, 61.22] | 6.50 [5.55, 7.49] | 59.40 [58.81, 59.99] | 52.58 [51.75, 53.40] |
| OLMo | CPT+SFT | 59.23 [58.20, 60.24] | 6.46 [5.51, 7.41] | 58.77 [58.16, 59.36] | 51.65 [50.81, 52.47] |
| OLMo | GRPO | 57.36 [56.29, 58.38] | 6.30 [5.35, 7.29] | 58.00 [57.41, 58.59] | 50.63 [49.81, 51.45] |
| OLMo | CPT+GRPO | 61.30 [60.23, 62.35] | 10.42 [9.27, 11.61] | 60.33 [59.69, 60.97] | 54.08 [53.18, 54.97] |
| Qwen | Base | 61.89 [60.82, 62.92] | 7.09 [6.06, 8.09] | 61.66 [61.06, 62.25] | 55.48 [54.65, 56.31] |
| Qwen | CPT | 57.98 [56.93, 58.99] | 5.23 [4.36, 6.10] | 59.23 [58.67, 59.79] | 52.14 [51.35, 52.91] |
| Qwen | SFT | 67.22 [66.12, 68.29] | 13.04 [11.73, 14.35] | 65.87 [65.21, 66.50] | 61.36 [60.44, 62.26] |
| Qwen | CPT+SFT | 54.60 [53.57, 55.59] | 3.80 [3.09, 4.56] | 56.86 [56.33, 57.39] | 48.90 [48.18, 49.63] |
| Qwen | GRPO | 65.06 [63.97, 66.12] | 11.06 [9.83, 12.29] | 64.09 [63.45, 64.72] | 58.99 [58.08, 59.88] |
| Qwen | CPT+GRPO | 59.56 [58.54, 60.57] | 5.63 [4.72, 6.54] | 59.61 [59.04, 60.17] | 52.73 [51.95, 53.51] |

**Paired deltas** (same bootstrap fact indices for both models; two-sided bootstrap p-value):

| Family | Comparison | Metric | Δ (pp) | 95% CI | p |
|---|---|---|---|---|---|
| OLMo | SFT − Base | Avg accuracy | +2.25 * | [+1.83, +2.67] | 0.0001 |
| OLMo | SFT − Base | Total consistency | -0.71 | [-1.59, +0.16] | 0.1100 |
| OLMo | SFT − Base | RankC | +1.08 * | [+0.77, +1.39] | 0.0001 |
| OLMo | SFT − Base | Answer agreement | +1.48 * | [+1.02, +1.92] | 0.0001 |
| OLMo | CPT − Base | Avg accuracy | -10.06 * | [-10.65, -9.47] | 0.0001 |
| OLMo | CPT − Base | Total consistency | -4.28 * | [-5.23, -3.37] | 0.0001 |
| OLMo | CPT − Base | RankC | -4.92 * | [-5.36, -4.49] | 0.0001 |
| OLMo | CPT − Base | Answer agreement | -6.83 * | [-7.46, -6.21] | 0.0001 |
| OLMo | GRPO − Base | Avg accuracy | -0.61 * | [-0.77, -0.45] | 0.0001 |
| OLMo | GRPO − Base | Total consistency | -0.91 * | [-1.35, -0.52] | 0.0001 |
| OLMo | GRPO − Base | RankC | -0.31 * | [-0.43, -0.20] | 0.0001 |
| OLMo | GRPO − Base | Answer agreement | -0.47 * | [-0.65, -0.29] | 0.0001 |
| OLMo | CPT+SFT − Base | Avg accuracy | +1.27 * | [+0.80, +1.72] | 0.0001 |
| OLMo | CPT+SFT − Base | Total consistency | -0.75 | [-1.70, +0.16] | 0.1240 |
| OLMo | CPT+SFT − Base | RankC | +0.46 * | [+0.12, +0.79] | 0.0070 |
| OLMo | CPT+SFT − Base | Answer agreement | +0.55 * | [+0.06, +1.03] | 0.0250 |
| OLMo | CPT+GRPO − Base | Avg accuracy | +3.34 * | [+2.90, +3.79] | 0.0001 |
| OLMo | CPT+GRPO − Base | Total consistency | +3.21 * | [+2.34, +4.12] | 0.0001 |
| OLMo | CPT+GRPO − Base | RankC | +2.02 * | [+1.71, +2.33] | 0.0001 |
| OLMo | CPT+GRPO − Base | Answer agreement | +2.97 * | [+2.52, +3.44] | 0.0001 |
| OLMo | GRPO − SFT | Avg accuracy | -2.85 * | [-3.28, -2.43] | 0.0001 |
| OLMo | GRPO − SFT | Total consistency | -0.20 | [-1.03, +0.63] | 0.6858 |
| OLMo | GRPO − SFT | RankC | -1.40 * | [-1.70, -1.09] | 0.0001 |
| OLMo | GRPO − SFT | Answer agreement | -1.95 * | [-2.40, -1.49] | 0.0001 |
| Qwen | SFT − Base | Avg accuracy | +5.33 * | [+4.90, +5.75] | 0.0001 |
| Qwen | SFT − Base | Total consistency | +5.95 * | [+4.91, +6.98] | 0.0001 |
| Qwen | SFT − Base | RankC | +4.21 * | [+3.88, +4.55] | 0.0001 |
| Qwen | SFT − Base | Answer agreement | +5.88 * | [+5.38, +6.36] | 0.0001 |
| Qwen | CPT − Base | Avg accuracy | -3.91 * | [-4.35, -3.48] | 0.0001 |
| Qwen | CPT − Base | Total consistency | -1.86 * | [-2.73, -1.03] | 0.0001 |
| Qwen | CPT − Base | RankC | -2.42 * | [-2.75, -2.10] | 0.0001 |
| Qwen | CPT − Base | Answer agreement | -3.35 * | [-3.83, -2.87] | 0.0001 |
| Qwen | GRPO − Base | Avg accuracy | +3.17 * | [+2.84, +3.51] | 0.0001 |
| Qwen | GRPO − Base | Total consistency | +3.96 * | [+3.09, +4.88] | 0.0001 |
| Qwen | GRPO − Base | RankC | +2.44 * | [+2.17, +2.71] | 0.0001 |
| Qwen | GRPO − Base | Answer agreement | +3.51 * | [+3.11, +3.92] | 0.0001 |
| Qwen | CPT+SFT − Base | Avg accuracy | -7.29 * | [-7.83, -6.78] | 0.0001 |
| Qwen | CPT+SFT − Base | Total consistency | -3.29 * | [-4.20, -2.38] | 0.0001 |
| Qwen | CPT+SFT − Base | RankC | -4.79 * | [-5.22, -4.38] | 0.0001 |
| Qwen | CPT+SFT − Base | Answer agreement | -6.58 * | [-7.18, -5.99] | 0.0001 |
| Qwen | CPT+GRPO − Base | Avg accuracy | -2.33 * | [-2.83, -1.84] | 0.0001 |
| Qwen | CPT+GRPO − Base | Total consistency | -1.47 * | [-2.50, -0.48] | 0.0056 |
| Qwen | CPT+GRPO − Base | RankC | -2.05 * | [-2.45, -1.66] | 0.0001 |
| Qwen | CPT+GRPO − Base | Answer agreement | -2.75 * | [-3.32, -2.19] | 0.0001 |
| Qwen | GRPO − SFT | Avg accuracy | -2.16 * | [-2.48, -1.84] | 0.0001 |
| Qwen | GRPO − SFT | Total consistency | -1.98 * | [-2.89, -1.07] | 0.0001 |
| Qwen | GRPO − SFT | RankC | -1.78 * | [-2.03, -1.52] | 0.0001 |
| Qwen | GRPO − SFT | Answer agreement | -2.37 * | [-2.75, -1.99] | 0.0001 |

`*` = 95% CI excludes zero.


# Label provenance vs Wikidata — `jvonrad/PolyFact-Clean`

16,828 distinct answer entities × 12 languages. Because v5 draws distractors from the gold-entity pool, this covers every option string.

| lang | verified | alias | **CONTRADICTS** | no ground truth | unknown entity |
|---|---|---|---|---|---|
| en | 98.4% | 0.2% | **48 (0.3%)** | 0.8% | 0.2% |
| de | 93.7% | 0.1% | **24 (0.1%)** | 5.9% | 0.2% |
| id | 43.6% | 0.0% | **17 (0.1%)** | 56.0% | 0.2% |
| pt | 77.6% | 0.0% | **19 (0.1%)** | 22.1% | 0.2% |
| ar | 60.9% | 0.1% | **76 (0.5%)** | 38.3% | 0.2% |
| bn | 21.3% | 0.2% | **35 (0.2%)** | 78.1% | 0.2% |
| sw | 17.8% | 0.0% | **9 (0.1%)** | 81.9% | 0.2% |
| es | 93.7% | 0.1% | **33 (0.2%)** | 5.8% | 0.2% |
| ru | 81.2% | 0.1% | **39 (0.2%)** | 18.3% | 0.2% |
| fr | 97.3% | 0.1% | **24 (0.1%)** | 2.2% | 0.2% |
| ja | 77.1% | 0.1% | **28 (0.2%)** | 22.4% | 0.2% |
| zh | 46.8% | 5.3% | **4,209 (25.0%)** | 22.6% | 0.2% |

`CONTRADICTS` = Wikidata has a label in that language and the dataset uses a different string, i.e. the generator overrode real data. `no ground truth` = Wikidata has no label in that language, so the string is an unverifiable model translation.

**ar — dataset label vs Wikidata label:**

- `Q1339888`: dataset `مارك ويليام بوي` / Wikidata `مارك وليم بوي`
- `Q2485848`: dataset `روي ويليام نيل` / Wikidata `روي وليم نيل`
- `Q9267`: dataset `اللغة التركمانية` / Wikidata `التركمانية`
- `Q58960`: dataset `أنتونين مركوس` / Wikidata `أنطونين مركوس`
- `Q40191`: dataset `هوبارت` / Wikidata `هوبرت`

**bn — dataset label vs Wikidata label:**

- `Q36236`: dataset `মালয়ালম ভাষা` / Wikidata `মালয়ালম`
- `Q2861`: dataset `রসটক` / Wikidata `রস্টক`
- `Q194223`: dataset `বুয়েনোস Aires বিশ্ববিদ্যালয়` / Wikidata `বুয়েনোস আইরেস বিশ্ববিদ্যালয়`
- `Q498407`: dataset `লাটভিয়া বিশ্ববিদ্যালয়` / Wikidata `লাতভিয়া বিশ্ববিদ্যালয়`
- `Q31519`: dataset `চার্লস বিশ্ববিদ্যালয়` / Wikidata `প্রাগ বিশ্ববিদ্যালয়`

**ru — dataset label vs Wikidata label:**

- `Q1121187`: dataset `Паулиста` / Wikidata `Companhia Aeronáutica Paulista`
- `Q3981496`: dataset `Татьяна Уэсо` / Wikidata `Уэсо, Татьяна`
- `Q573967`: dataset `Энтони Сальвин` / Wikidata `Сальвин, Антони`
- `Q1348664`: dataset `«Рокуэлл»` / Wikidata `Rockwell International`
- `Q66766`: dataset `Андреас Зайфарт` / Wikidata `Андреас Зейфарт`

**ja — dataset label vs Wikidata label:**

- `Q312`: dataset `アップル株式会社` / Wikidata `Apple Inc.`
- `Q8032103`: dataset `チン・ウォンスク` / Wikidata `ジン・ウォンソク`
- `Q95485`: dataset `アウグスト＝ヴィルヘルム・シェーア` / Wikidata `アウグスト＝ヴィルヘルム・シェアー`
- `Q7432923`: dataset `Schrödinger (企業)` / Wikidata `Schrödinger`
- `Q108064175`: dataset `田村 孝太郎` / Wikidata `タムラコータロー`

**zh — dataset label vs Wikidata label:**

- `Q459464`: dataset `SOM建筑设计事务所` / Wikidata `SOM建築設計事務所`
- `Q5287`: dataset `日语` / Wikidata `日語`
- `Q3655767`: dataset `第勒尼安海联合造船厂` / Wikidata `第勒尼安海聯合造船廠`
- `Q315249`: dataset `阿诺尔·迪·坎比奥` / Wikidata `阿諾爾·迪·坎比奧`
- `Q2633868`: dataset `克拉伊纳第纳尔` / Wikidata `克拉伊納第納爾`

## Fact-level exposure

| lang | facts whose gold CONTRADICTS Wikidata | facts whose gold has no ground truth |
|---|---|---|
| en | 94 (0.1%) | 524 (0.8%) |
| de | 48 (0.1%) | 2,260 (3.5%) |
| id | 46 (0.1%) | 18,957 (29.3%) |
| pt | 166 (0.3%) | 6,625 (10.2%) |
| ar | 159 (0.2%) | 12,534 (19.4%) |
| bn | 106 (0.2%) | 24,361 (37.7%) |
| sw | 94 (0.1%) | 31,950 (49.4%) |
| es | 77 (0.1%) | 2,373 (3.7%) |
| ru | 56 (0.1%) | 5,009 (7.7%) |
| fr | 70 (0.1%) | 870 (1.3%) |
| ja | 184 (0.3%) | 4,971 (7.7%) |
| zh | 14,239 (22.0%) | 5,037 (7.8%) |



---

## Correction: the Chinese `CONTRADICTS` rate is not a translation problem

zh shows 4,209 contradictions (25.0%), an order of magnitude above every other
language. Converting both strings to Simplified before comparing resolves it:

| | entities | share |
|---|---|---|
| same text, Simplified/Traditional variant | 4,153 | **98.7%** |
| genuinely different text | 56 | 1.3% |

The dataset is standardised on Simplified Chinese while Wikidata's `zh` label is
frequently Traditional (`SOM建筑设计事务所` vs `SOM建築設計事務所`), and most of these
entities have no explicit `zh-hans` entry to match against. The corrected zh
contradiction rate is **56 / 16,828 = 0.3%**, in line with the other languages.

Among the 56 real ones are a few genuine defects worth fixing by hand:
`Q704161` = `罗维,` (trailing comma), `Q2142669` = `AmBev` (left untranslated),
`Q2657321` = `佐佐川` vs Wikidata `薩薩夸` (different transliteration).

## Reading the other contradictions

Outside zh they run 9–76 entities per language (0.1–0.5%) and are mostly benign
variants rather than errors:

- transliteration choices — ar `مارك ويليام بوي` vs `مارك وليم بوي` (William);
- naming convention — ru `Татьяна Уэсо` vs Wikidata's sort-order `Уэсо, Татьяна`,
  where the dataset form is arguably the better one for a question;
- added disambiguation — ar `اللغة التركمانية` ("the Turkmen language") vs `التركمانية`;
- genuinely different but valid names — bn `চার্লস বিশ্ববিদ্যালয়` (Charles University)
  vs `প্রাগ বিশ্ববিদ্যালয়` (Prague University).

A minority are real defects, notably mixed-script labels that the strict
intra-word filter did not catch because the Latin run is a separate token:
bn `বুয়েনোস Aires বিশ্ববিদ্যালয়` (should be `বুয়েনোস আইরেস বিশ্ববিদ্যালয়`).

## The material finding: unverifiable labels in low-resource languages

`no ground truth` — Wikidata has no label for the entity in that language, so the
string is a model translation with nothing to check it against:

| sw | bn | id | ar | zh | ja | pt | ru | de | es | fr | en |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **81.9%** | **78.1%** | **56.0%** | **38.3%** | 22.6% | 22.4% | 22.1% | 18.3% | 5.9% | 5.8% | 2.2% | 0.8% |

For Swahili and Bengali roughly four out of five answer labels were produced by
the generation model rather than taken from Wikidata. Those are the two languages
where the paper reports its weakest results, so label quality is a live
alternative explanation for part of that gap and should be stated as a limitation.

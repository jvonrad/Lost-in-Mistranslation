# Deep label verification — `jvonrad/PolyFact-Clean`

16,099 distinct answer entities × 12 languages (193,188 labels). Distractors are drawn from the gold-entity pool, so this covers every option string in the dataset.

Ground truth is the union of the Wikidata label, its language variants (`zh-hans`, `pt-br`, …), the language's Wikipedia article title, and aliases.

| lang | verified | variant | sitelink | alias | **attested total** | **CONTRADICTS** | no ground truth |
|---|---|---|---|---|---|---|---|
| en | 98.4% | 0.2% | 0.8% | 0.1% | **99.6%** | **39 (0.2%)** | 0.1% |
| de | 94.0% | 0.0% | 0.7% | 0.0% | **94.7%** | **27 (0.2%)** | 5.0% |
| id | 43.4% | 0.0% | 0.7% | 0.0% | **44.1%** | **23 (0.1%)** | 55.6% |
| pt | 78.0% | 1.1% | 0.7% | 0.0% | **79.8%** | **145 (0.9%)** | 19.2% |
| ar | 60.7% | 0.0% | 0.1% | 0.1% | **60.8%** | **68 (0.4%)** | 38.7% |
| bn | 21.6% | 0.0% | 0.2% | 0.0% | **21.8%** | **28 (0.2%)** | 77.9% |
| sw | 18.2% | 0.0% | 0.5% | 0.0% | **18.7%** | **13 (0.1%)** | 81.1% |
| es | 93.7% | 0.0% | 0.7% | 0.1% | **94.5%** | **31 (0.2%)** | 5.2% |
| ru | 80.9% | 0.0% | 0.6% | 0.1% | **81.6%** | **45 (0.3%)** | 18.0% |
| fr | 97.4% | 0.0% | 0.8% | 0.0% | **98.2%** | **28 (0.2%)** | 1.5% |
| ja | 76.9% | 0.0% | 0.1% | 0.1% | **77.1%** | **23 (0.1%)** | 22.7% |
| zh | 76.1% | 0.4% | 0.1% | 0.1% | **76.8%** | **138 (0.9%)** | 22.3% |
| **all** | | | | | **70.6%** | **608 (0.3%)** | **28.9%** |

Only **55,903 / 193,188 (28.9%)** labels have no attested name behind them; these are the only ones that could be a silent hallucination, and they are the sampling frame in `results/polyfact_unverifiable_pool.json`.

**zh — remaining contradictions:**

- `Q688350`: dataset `阿明·韦格纳` / Wikidata `阿明·魏格纳`
- `Q21996568`: dataset `亚历珊卓·艾尔巴金` / Wikidata `亞歷珊卓·艾爾巴金`
- `Q1680032`: dataset `詹姆斯·班宁` / Wikidata `詹姆斯·班寧`
- `Q41502`: dataset `亨利克·亚当·亚历山大·皮乌斯·显克微支` / Wikidata `亨利克·亞當·亞歷山大·皮烏斯·显克微支`
- `Q882`: dataset `差利·卓别灵` / Wikidata `差利·卓別靈`

**ar — remaining contradictions:**

- `Q1339888`: dataset `مارك ويليام بوي` / Wikidata `مارك وليم بوي`
- `Q2485848`: dataset `روي ويليام نيل` / Wikidata `روي وليم نيل`
- `Q58960`: dataset `أنتونين مركوس` / Wikidata `أنطونين مركوس`
- `Q5365878`: dataset `إليس إف. لورانس` / Wikidata `إليس إف. لورنس`
- `Q4480746`: dataset `كلية الصحافة بجامعة ولاية ميشيغان` / Wikidata `جامعة موسكو الوطنية كلية الصحافة`

**bn — remaining contradictions:**

- `Q2861`: dataset `রসটক` / Wikidata `রস্টক`
- `Q194223`: dataset `বুয়েনোস Aires বিশ্ববিদ্যালয়` / Wikidata `বুয়েনোস আইরেস বিশ্ববিদ্যালয়`
- `Q498407`: dataset `লাটভিয়া বিশ্ববিদ্যালয়` / Wikidata `লাতভিয়া বিশ্ববিদ্যালয়`
- `Q31519`: dataset `চার্লস বিশ্ববিদ্যালয়` / Wikidata `প্রাগ বিশ্ববিদ্যালয়`
- `Q552372`: dataset `জনস্টন ম্যাককুলী` / Wikidata `জনস্টন ম্যাককালি`

**sw — remaining contradictions:**

- `Q294`: dataset `Kiiceland` / Wikidata `Kiisilandi`
- `Q216273`: dataset `Chuo Kikuu cha St Andrews` / Wikidata `Chuo Kikuu cha Mtakatifu Andrea`
- `Q380`: dataset `Meta` / Wikidata `Meta Platforms`
- `Q43595`: dataset `Cotonú` / Wikidata `Cotonou`
- `Q11638963`: dataset `Marie Kondō` / Wikidata `Marie Kondo`

**ja — remaining contradictions:**

- `Q312`: dataset `アップル株式会社` / Wikidata `Apple Inc.`
- `Q8032103`: dataset `チン・ウォンスク` / Wikidata `ジン・ウォンソク`
- `Q95485`: dataset `アウグスト＝ヴィルヘルム・シェーア` / Wikidata `アウグスト＝ヴィルヘルム・シェアー`
- `Q7432923`: dataset `Schrödinger (企業)` / Wikidata `Schrödinger`
- `Q108064175`: dataset `田村 孝太郎` / Wikidata `タムラコータロー`

**ru — remaining contradictions:**

- `Q459464`: dataset `Skidmore, Owings and Merrill` / Wikidata `Skidmore, Owings & Merrill`
- `Q1121187`: dataset `Паулиста` / Wikidata `Companhia Aeronáutica Paulista`
- `Q312`: dataset `Apple Inc.` / Wikidata `Apple`
- `Q3981496`: dataset `Татьяна Уэсо` / Wikidata `Уэсо, Татьяна`
- `Q573967`: dataset `Энтони Сальвин` / Wikidata `Сальвин, Антони`

## Unverifiable labels by relation

| relation | unverifiable labels | distinct entities |
|---|---|---|
| architect | 14,399 | 2,678 |
| director | 12,269 | 3,342 |
| educated at | 6,871 | 1,837 |
| developer | 5,843 | 1,366 |
| place of death | 4,147 | 1,570 |
| manufacturer | 3,727 | 934 |
| creator | 3,519 | 1,002 |
| author | 3,076 | 1,011 |
| discoverer or inventor | 1,544 | 397 |
| official language | 212 | 89 |
| country of citizenship | 140 | 74 |
| language of work or name | 94 | 46 |
| country | 47 | 27 |
| continent | 15 | 4 |

## Fact-level coverage

- facts whose gold label is attested in **all 12** languages: **29,592 / 60,169 (49.2%)**

| languages attested | facts |
|---|---|
| 12 | 29,592 |
| 11 | 5,788 |
| 10 | 6,565 |
| 9 | 6,394 |
| 8 | 3,732 |
| 7 | 3,123 |
| 6 | 2,551 |
| 5 | 2,337 |
| 4 | 13 |
| 3 | 14 |
| 2 | 9 |
| 1 | 27 |
| 0 | 24 |


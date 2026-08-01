# Manual review of PolyFact-Clean

## Answer labels

Reviewed stratum = labels Wikidata cannot attest that are also NOT phonetic renderings of the English name. Labels outside it are either Wikidata-attested or transliterations (0 errors in 180 sampled).

| lang | reviewed | ok | wrong | unsure | error rate |
|---|---|---|---|---|---|
| zh | 1,986 | 1,906 | **73** | 7 | **3.7%** |
| ar | 1,778 | 1,732 | **38** | 8 | **2.1%** |
| sw | 1,420 | 1,316 | **98** | 6 | **6.9%** |
| bn | 1,405 | 1,383 | **17** | 5 | **1.2%** |
| ja | 658 | 646 | **9** | 3 | **1.4%** |
| id | 173 | 158 | **13** | 2 | **7.5%** |
| ru | 128 | 118 | **9** | 1 | **7.0%** |
| pt | 77 | 70 | **7** | 0 | **9.1%** |
| es | 51 | 46 | **5** | 0 | **9.8%** |
| de | 24 | 18 | **6** | 0 | **25.0%** |
| fr | 8 | 8 | **0** | 0 | **0.0%** |
| **all** | **7,708** | 7,401 | **275** | 32 | **3.6%** |

- distinct entities with a confirmed-wrong label: **207**
- as a share of all 193,188 answer labels in the release: **0.14%**

## Test-split questions

`subject` = asks about a different entity (severe). `type` = right subject, wrong kind of thing. `relation` = wrong property.

| lang | reviewed | ok | subject | type | relation | unsure | defect rate |
|---|---|---|---|---|---|---|---|
| ar | 1,517 | 1,440 | **19** | 36 | 17 | 5 | **5.1%** |
| bn | 1,517 | 1,486 | **9** | 17 | 4 | 1 | **2.0%** |
| de | 1,517 | 1,495 | **9** | 8 | 0 | 5 | **1.5%** |
| es | 1,517 | 1,475 | **18** | 15 | 5 | 4 | **2.8%** |
| fr | 1,517 | 1,481 | **15** | 18 | 2 | 1 | **2.4%** |
| id | 1,517 | 1,473 | **12** | 21 | 8 | 3 | **2.9%** |
| ja | 1,517 | 1,474 | **9** | 24 | 9 | 1 | **2.8%** |
| pt | 1,517 | 1,475 | **19** | 19 | 4 | 0 | **2.8%** |
| ru | 1,517 | 1,465 | **18** | 26 | 6 | 2 | **3.4%** |
| sw | 1,517 | 1,445 | **26** | 34 | 11 | 1 | **4.7%** |
| zh | 1,517 | 1,470 | **19** | 18 | 5 | 5 | **3.1%** |
| **all** | **16,687** | 16,179 | **173** | 236 | 71 | 28 | **3.0%** |

- distinct test facts with a defect in at least one language: **311**


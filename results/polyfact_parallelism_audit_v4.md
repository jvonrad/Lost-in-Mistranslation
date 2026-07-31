# Parallelism & question-quality audit — `jvonrad/PolyFact-Clean`

## A. Cross-lingual option alignment (vs English)

Each language's 4 options are mapped to Wikidata entity ids via the canonical gold-label table; a mismatch means a language offers a different candidate set for the same fact.

| lang | facts with a different entity set | facts with an unmappable option |
|---|---|---|
| en | (reference) | 50,591 (74.8%) |
| de | **389** | 50,570 (74.8%) |
| id | **683** | 50,617 (74.9%) |
| pt | **414** | 50,679 (75.0%) |
| ar | **312** | 50,630 (74.9%) |
| bn | **528** | 50,643 (74.9%) |
| sw | **300** | 50,460 (74.6%) |
| es | **403** | 50,575 (74.8%) |
| ru | **441** | 50,463 (74.6%) |
| fr | **285** | 50,694 (75.0%) |
| ja | **349** | 50,656 (74.9%) |
| zh | **173** | 50,778 (75.1%) |

- [de] Lièpvre — *continent*: ['Q408', 'Q46', 'Q49', 'Q55643'] vs en ['Q3960', 'Q46', 'Q49', 'Q55643']
- [de] Septemvri — *continent*: ['Q15', 'Q46', 'Q55643', 'Q828'] vs en ['Q15', 'Q46', 'Q538', 'Q828']
- [de] Kievan Rus' — *currency*: ['Q16068', 'Q204737', 'Q23541018', 'Q303713'] vs en ['Q16068', 'Q204737', 'Q2329625', 'Q23541018']
- [id] U-185 — *manufacturer*: ['Q12193', 'Q14365850', 'Q292748', 'Q7705226'] vs en ['Q140147', 'Q14365850', 'Q292748', 'Q7705226']
- [id] Lièpvre — *continent*: ['Q408', 'Q46', 'Q49', 'Q55643'] vs en ['Q3960', 'Q46', 'Q49', 'Q55643']
- [id] Machakos — *country*: ['Q114', 'Q219060', 'Q2577303', 'Q347'] vs en ['Q114', 'Q23792', 'Q2577303', 'Q347']
- [pt] Lièpvre — *continent*: ['Q408', 'Q46', 'Q49', 'Q55643'] vs en ['Q3960', 'Q46', 'Q49', 'Q55643']
- [pt] Septemvri — *continent*: ['Q15', 'Q46', 'Q55643', 'Q828'] vs en ['Q15', 'Q46', 'Q538', 'Q828']
- [pt] Machakos — *country*: ['Q114', 'Q219060', 'Q2577303', 'Q347'] vs en ['Q114', 'Q23792', 'Q2577303', 'Q347']
- [ar] U-185 — *manufacturer*: ['Q12193', 'Q14365850', 'Q292748', 'Q7705226'] vs en ['Q140147', 'Q14365850', 'Q292748', 'Q7705226']
- [ar] Lièpvre — *continent*: ['Q408', 'Q46', 'Q49', 'Q55643'] vs en ['Q3960', 'Q46', 'Q49', 'Q55643']
- [ar] Cortes de Arenoso — *continent*: ['Q18', 'Q408', 'Q46', 'Q48'] vs en ['Q18', 'Q3960', 'Q46', 'Q48']
- [bn] U-185 — *manufacturer*: ['Q12193', 'Q14365850', 'Q292748', 'Q7705226'] vs en ['Q140147', 'Q14365850', 'Q292748', 'Q7705226']
- [bn] Lièpvre — *continent*: ['Q408', 'Q46', 'Q49', 'Q55643'] vs en ['Q3960', 'Q46', 'Q49', 'Q55643']
- [bn] Septemvri — *continent*: ['Q15', 'Q46', 'Q55643', 'Q828'] vs en ['Q15', 'Q46', 'Q538', 'Q828']
- [sw] Besalú — *official language*: ['Q1750889', 'Q7026', 'Q711', 'Q9192'] vs en ['Q1750889', 'Q7026', 'Q9192', 'Q9246']
- [sw] Kievan Rus' — *currency*: ['Q204737', 'Q2329625', 'Q23541018', 'Q560355'] vs en ['Q16068', 'Q204737', 'Q2329625', 'Q23541018']
- [sw] Vatican City — *currency*: ['Q213142', 'Q41588', 'Q4916', 'Q747909'] vs en ['Q178843', 'Q213142', 'Q4916', 'Q747909']
- [es] Lièpvre — *continent*: ['Q408', 'Q46', 'Q49', 'Q55643'] vs en ['Q3960', 'Q46', 'Q49', 'Q55643']
- [es] Kievan Rus' — *currency*: ['Q16068', 'Q204737', 'Q2329625', 'Q81893'] vs en ['Q16068', 'Q204737', 'Q2329625', 'Q23541018']
- [es] Cortes de Arenoso — *continent*: ['Q18', 'Q408', 'Q46', 'Q48'] vs en ['Q18', 'Q3960', 'Q46', 'Q48']
- [ru] Lièpvre — *continent*: ['Q408', 'Q46', 'Q49', 'Q55643'] vs en ['Q3960', 'Q46', 'Q49', 'Q55643']
- [ru] Machakos — *country*: ['Q114', 'Q219060', 'Q2577303', 'Q347'] vs en ['Q114', 'Q23792', 'Q2577303', 'Q347']
- [ru] Cortes de Arenoso — *continent*: ['Q18', 'Q408', 'Q46', 'Q48'] vs en ['Q18', 'Q3960', 'Q46', 'Q48']
- [fr] Septemvri — *continent*: ['Q15', 'Q46', 'Q55643', 'Q828'] vs en ['Q15', 'Q46', 'Q538', 'Q828']
- [fr] Machakos — *country*: ['Q114', 'Q219060', 'Q2577303', 'Q347'] vs en ['Q114', 'Q23792', 'Q2577303', 'Q347']
- [fr] Saint-Phal — *country*: ['Q142', 'Q28', 'Q36947', 'Q805'] vs en ['Q142', 'Q28', 'Q792', 'Q805']
- [ja] Machakos — *country*: ['Q114', 'Q219060', 'Q2577303', 'Q347'] vs en ['Q114', 'Q23792', 'Q2577303', 'Q347']
- [ja] Mikhail Lyubimov — *educated at*: ['Q1460141', 'Q1813336', 'Q1888771', 'Q322964'] vs en ['Q1460141', 'Q1888771', 'Q322964', 'Q3412538']
- [ja] Adriano Buzaid — *country of citizenship*: ['Q1049', 'Q155', 'Q29520', 'Q865'] vs en ['Q1049', 'Q13426199', 'Q155', 'Q29520']
- [zh] U-185 — *manufacturer*: ['Q12193', 'Q14365850', 'Q292748', 'Q7705226'] vs en ['Q140147', 'Q14365850', 'Q292748', 'Q7705226']
- [zh] Deutsches Schauspielhaus — *architect*: ['Q113014', 'Q26707690', 'Q49744', 'Q7327308'] vs en ['Q26707690', 'Q49744', 'Q694854', 'Q7327308']
- [zh] Saint-Phal — *country*: ['Q142', 'Q28', 'Q36947', 'Q805'] vs en ['Q142', 'Q28', 'Q792', 'Q805']

## B. Untranslated questions (byte-identical to English)

| lang | count | share |
|---|---|---|
| de | 0 | 0.00% |
| id | 0 | 0.00% |
| pt | 0 | 0.00% |
| ar | 0 | 0.00% |
| bn | 0 | 0.00% |
| sw | 0 | 0.00% |
| es | 5 | 0.01% |
| ru | 0 | 0.00% |
| fr | 0 | 0.00% |
| ja | 0 | 0.00% |
| zh | 0 | 0.00% |

- [es] Who directed the film Pastoral: To Die in the Country?
- [es] Who directed the anime series Brothers Conflict?

## C. Gold answer leaked into the question text

| lang | count | share |
|---|---|---|
| en | 25 | 0.04% |
| de | 40 | 0.06% |
| id | 6 | 0.01% |
| pt | 8 | 0.01% |
| ar | 14 | 0.02% |
| bn | 13 | 0.02% |
| sw | 10 | 0.01% |
| es | 24 | 0.04% |
| ru | 11 | 0.02% |
| fr | 36 | 0.05% |
| ja | 0 | 0.00% |
| zh | 0 | 0.00% |

- [en] gold `Turkmen` in: What is the official language of Turkmenistan?
- [en] gold `Thai` in: Lèse-majesté laws in Thailand concern offenses against the institution of the monarch. In what langu
- [en] gold `Uzbek` in: What is the official language of Uzbekistan?
- [de] gold `Rembrandt` in: Wer ist der Urheber von 'Rembrandts Sohn Titus in Mönchstracht'?
- [de] gold `Serbisch` in: Welche Sprache ist Amtssprache der Serbischen Regierung?
- [de] gold `Deutsch` in: Welche Sprache ist eine Amtssprache in Deutschland, dem Land, aus dem Eier stammen?
- [fr] gold `philippin` in: Quelle est la langue officielle des Philippines ?
- [fr] gold `malais` in: Quelle est la langue officielle de l'État de Kedah en Malaisie ?
- [fr] gold `thaï` in: Quelle est la langue officielle du ministère de la défense de la Thaïlande ?

## D. Identical question with different gold answers

| lang | question strings with >1 gold |
|---|---|
| en | 0 |
| de | 0 |
| id | 0 |
| pt | 0 |
| ar | 0 |
| bn | 0 |
| sw | 0 |
| es | 0 |
| ru | 0 |
| fr | 0 |
| ja | 0 |
| zh | 0 |


## E. Degenerate fields

- empty question or option: **0** across all languages
- an option equal to the subject entity: **2**
  - [de] subject `Liezi` is an option — *author* — Wer ist der Autor des Buches 'Das wahre Buch vom quellenden Urgrund'?
  - [es] subject `The Book of Abramelin` is an option — *author* — Según la información disponible, ¿quién es el autor del libro de Abram


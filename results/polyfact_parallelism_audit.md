# Parallelism & question-quality audit — `jvonrad/PolyFact-Clean`

## A. Cross-lingual option alignment (vs English)

Each language's 4 options are mapped to Wikidata entity ids via the canonical gold-label table; a mismatch means a language offers a different candidate set for the same fact.

| lang | facts with a different entity set | facts with an unmappable option |
|---|---|---|
| en | (reference) | 50,462 (67.9%) |
| de | **565** | 50,534 (68.0%) |
| id | **880** | 50,444 (67.9%) |
| pt | **538** | 50,642 (68.2%) |
| ar | **422** | 50,478 (68.0%) |
| bn | **754** | 50,566 (68.1%) |
| sw | **452** | 50,429 (67.9%) |
| es | **481** | 50,406 (67.9%) |
| ru | **548** | 50,374 (67.8%) |
| fr | **422** | 50,582 (68.1%) |
| ja | **448** | 50,629 (68.2%) |
| zh | **237** | 50,675 (68.2%) |

- [de] Lièpvre — *continent*: ['Q408', 'Q46', 'Q49', 'Q55643'] vs en ['Q3960', 'Q46', 'Q49', 'Q55643']
- [de] Septemvri — *continent*: ['Q15', 'Q46', 'Q55643', 'Q828'] vs en ['Q15', 'Q46', 'Q538', 'Q828']
- [de] Volvo L3314 — *manufacturer*: ['Q152864', 'Q163810', 'Q696016', 'Q7848369'] vs en ['Q152864', 'Q163810', 'Q6986', 'Q7848369']
- [id] U-185 — *manufacturer*: ['Q12193', 'Q14365850', 'Q292748', 'Q7705226'] vs en ['Q140147', 'Q14365850', 'Q292748', 'Q7705226']
- [id] Lièpvre — *continent*: ['Q408', 'Q46', 'Q49', 'Q55643'] vs en ['Q3960', 'Q46', 'Q49', 'Q55643']
- [id] Machakos — *country*: ['Q114', 'Q219060', 'Q2577303', 'Q347'] vs en ['Q114', 'Q23792', 'Q2577303', 'Q347']
- [pt] Lièpvre — *continent*: ['Q408', 'Q46', 'Q49', 'Q55643'] vs en ['Q3960', 'Q46', 'Q49', 'Q55643']
- [pt] Septemvri — *continent*: ['Q15', 'Q46', 'Q55643', 'Q828'] vs en ['Q15', 'Q46', 'Q538', 'Q828']
- [pt] Volvo L3314 — *manufacturer*: ['Q152864', 'Q163810', 'Q696016', 'Q7848369'] vs en ['Q152864', 'Q163810', 'Q6986', 'Q7848369']
- [ar] U-185 — *manufacturer*: ['Q12193', 'Q14365850', 'Q292748', 'Q7705226'] vs en ['Q140147', 'Q14365850', 'Q292748', 'Q7705226']
- [ar] Lièpvre — *continent*: ['Q408', 'Q46', 'Q49', 'Q55643'] vs en ['Q3960', 'Q46', 'Q49', 'Q55643']
- [ar] Volvo L3314 — *manufacturer*: ['Q152864', 'Q163810', 'Q696016', 'Q7848369'] vs en ['Q152864', 'Q163810', 'Q6986', 'Q7848369']
- [bn] U-185 — *manufacturer*: ['Q12193', 'Q14365850', 'Q292748', 'Q7705226'] vs en ['Q140147', 'Q14365850', 'Q292748', 'Q7705226']
- [bn] Lièpvre — *continent*: ['Q408', 'Q46', 'Q49', 'Q55643'] vs en ['Q3960', 'Q46', 'Q49', 'Q55643']
- [bn] Septemvri — *continent*: ['Q15', 'Q46', 'Q55643', 'Q828'] vs en ['Q15', 'Q46', 'Q538', 'Q828']
- [sw] Kievan Rus' — *currency*: ['Q204737', 'Q2329625', 'Q23541018', 'Q560355'] vs en ['Q16068', 'Q204737', 'Q2329625', 'Q23541018']
- [sw] Voeren — *official language*: ['Q2093002', 'Q33750', 'Q7411', 'Q8828'] vs en ['Q2093002', 'Q33750', 'Q7411', 'Q951473']
- [sw] 1999–2000 Turkish Cup — *country*: ['Q229', 'Q43', 'Q734', 'Q865'] vs en ['Q13426199', 'Q229', 'Q43', 'Q734']
- [es] Lièpvre — *continent*: ['Q408', 'Q46', 'Q49', 'Q55643'] vs en ['Q3960', 'Q46', 'Q49', 'Q55643']
- [es] Volvo L3314 — *manufacturer*: ['Q152864', 'Q163810', 'Q696016', 'Q7848369'] vs en ['Q152864', 'Q163810', 'Q6986', 'Q7848369']
- [es] Kievan Rus' — *currency*: ['Q16068', 'Q204737', 'Q2329625', 'Q81893'] vs en ['Q16068', 'Q204737', 'Q2329625', 'Q23541018']
- [ru] Lièpvre — *continent*: ['Q408', 'Q46', 'Q49', 'Q55643'] vs en ['Q3960', 'Q46', 'Q49', 'Q55643']
- [ru] Machakos — *country*: ['Q114', 'Q219060', 'Q2577303', 'Q347'] vs en ['Q114', 'Q23792', 'Q2577303', 'Q347']
- [ru] Cortes de Arenoso — *continent*: ['Q18', 'Q408', 'Q46', 'Q48'] vs en ['Q18', 'Q3960', 'Q46', 'Q48']
- [fr] Septemvri — *continent*: ['Q15', 'Q46', 'Q55643', 'Q828'] vs en ['Q15', 'Q46', 'Q538', 'Q828']
- [fr] Volvo L3314 — *manufacturer*: ['Q152864', 'Q163810', 'Q696016', 'Q7848369'] vs en ['Q152864', 'Q163810', 'Q6986', 'Q7848369']
- [fr] Kingdom of Candia — *currency*: ['Q191830', 'Q193098', 'Q202018', 'Q526050'] vs en ['Q187776', 'Q191830', 'Q193098', 'Q202018']
- [ja] LADA Priora — *manufacturer*: ['Q2309', 'Q264346', 'Q27582', 'Q27597'] vs en ['Q2309', 'Q264346', 'Q2738829', 'Q27582']
- [ja] Volvo L3314 — *manufacturer*: ['Q152864', 'Q163810', 'Q696016', 'Q7848369'] vs en ['Q152864', 'Q163810', 'Q6986', 'Q7848369']
- [ja] The Met Philadelphia — *architect*: ['Q17183520', 'Q328728', 'Q5226191', 'Q5538851'] vs en ['Q17183520', 'Q5226191', 'Q5538851', 'Q725060']
- [zh] U-185 — *manufacturer*: ['Q12193', 'Q14365850', 'Q292748', 'Q7705226'] vs en ['Q140147', 'Q14365850', 'Q292748', 'Q7705226']
- [zh] Volvo L3314 — *manufacturer*: ['Q152864', 'Q163810', 'Q696016', 'Q7848369'] vs en ['Q152864', 'Q163810', 'Q6986', 'Q7848369']
- [zh] The Met Philadelphia — *architect*: ['Q1355182', 'Q17183520', 'Q5226191', 'Q5538851'] vs en ['Q17183520', 'Q5226191', 'Q5538851', 'Q725060']

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
| en | 2,140 | 2.88% |
| de | 2,194 | 2.95% |
| id | 2,105 | 2.83% |
| pt | 2,188 | 2.95% |
| ar | 1,982 | 2.67% |
| bn | 1,812 | 2.44% |
| sw | 2,259 | 3.04% |
| es | 2,210 | 2.98% |
| ru | 1,877 | 2.53% |
| fr | 2,482 | 3.34% |
| ja | 1,698 | 2.29% |
| zh | 834 | 1.12% |

- [en] gold `Nokia` in: Who manufactured the Nokia 5000?
- [en] gold `Sega` in: Who developed the Sega TeraDrive?
- [en] gold `Microsoft` in: Who is the developer of the Microsoft Windows SDK?
- [de] gold `Guatemala` in: Zu welchem Staat gehört die Federación Nacional de Fútbol de Guatemala?
- [de] gold `Nokia` in: Wer ist der Hersteller des Nokia 5000?
- [de] gold `Sega` in: Wer ist der Entwickler des Sega TeraDrive?
- [fr] gold `Guatemala` in: À quel pays la Fédération de football du Guatemala est-elle associée ?
- [fr] gold `Nokia` in: Quel est le fabricant du Nokia 5000 ?
- [fr] gold `Asie` in: Sur quel continent le Baháisme est-il pratiqué en Asie ?

## D. Identical question with different gold answers

| lang | question strings with >1 gold |
|---|---|
| en | 190 |
| de | 164 |
| id | 211 |
| pt | 179 |
| ar | 297 |
| bn | 238 |
| sw | 217 |
| es | 153 |
| ru | 302 |
| fr | 152 |
| ja | 234 |
| zh | 246 |

- Who is the creator of the painting Madonna and Child? → ['Aelbrecht Bouts', 'Cornelis van Haarlem', 'Parmigianino', 'Ridolfo del Ghirlandaio']
- Who is the creator of the self-portrait? → ['Antoni Ziemięcki', 'Columbano Bordalo Pinheiro', 'Cristofano Allori', 'Daniel Schultz', 'David Martin', 'Elisabeth Baumann', 'Elizaveta Kruglikova', 'Franz Kessler', 'Giorgione', 'Gustaw Gwozdecki', 'Henriette Wolters-van Pee', 'Heva Coomans', 'Ion Andreescu', 'Jacques Raymond Brascassat', 'Jo Koster', 'Joseph Melling', 'Lavinia Fontana', 'Louis Dulongpré', 'Mary Beale', 'Nicolas Poussin', 'Paul Cézanne', 'Pierre-Auguste Renoir', 'Theo van Doesburg', 'Thomas Smith', 'Wouter Johannes van Troostwijk']
- Who was the architect of the Apollo Theatre? → ['Eugene De Rosa', 'John Fairweather']
- On which continent is Danişment located? → ['Asia', 'Europe']

## E. Degenerate fields

- empty question or option: **0** across all languages
- an option equal to the subject entity: **182**
  - [en] subject `George Morland` is an option — *creator* — Who is the creator of the painting 'The Blacksmith's Shop'?
  - [en] subject `OTRAG` is an option — *manufacturer* — Which company is the manufacturer of the OTRAG wheel?
  - [de] subject `George Morland` is an option — *creator* — Wer ist der Urheber des Selbstporträts?
  - [de] subject `OTRAG` is an option — *manufacturer* — Wer war der Hersteller des OTRAG?
  - [id] subject `George Morland` is an option — *creator* — Siapakah pencipta lukisan-lukisan yang menggambarkan kehidupan pedesaa
  - [id] subject `Swiss` is an option — *country of citizenship* — Orang yang berasal dari Konfederasi Swiss memiliki kewarganegaraan apa
  - [pt] subject `George Morland` is an option — *creator* — Quem é o criador do Auto-retrato?
  - [pt] subject `OTRAG` is an option — *manufacturer* — Quem fabricou o foguete OTRAG?
  - [ar] subject `SingleStore` is an option — *developer* — من هو مُطوِّر SingleStore؟
  - [ar] subject `Activeweave` is an option — *developer* — من هو المُطوِّر لـ Activeweave؟
  - [sw] subject `George Morland` is an option — *creator* — Nani alikuwa muumbaji?
  - [sw] subject `OTRAG` is an option — *manufacturer* — Ni kampuni gani iliyojulikana kama mtengenezaji wa magari ya Kiafrika?
  - [es] subject `George Morland` is an option — *creator* — Según los registros históricos, ¿quién es el creador de la obra 'Autor
  - [es] subject `OTRAG` is an option — *manufacturer* — OTRAG es conocido principalmente como:
  - [ru] subject `OTRAG` is an option — *manufacturer* — Кто является производителем автомобилей OTRAG?
  - [ru] subject `QB64` is an option — *developer* — Кто является разработчиком QB64?
  - [fr] subject `George Morland` is an option — *creator* — Qui a créé les peintures de scènes de la vie quotidienne anglaise, sou
  - [fr] subject `OTRAG` is an option — *manufacturer* — Quelle entreprise était le fabricant original des engins de chantier u
  - [ja] subject `OTRAG` is an option — *manufacturer* — OTRAGの製造元はどこですか？
  - [ja] subject `QB64` is an option — *developer* — QB64の開発元はどこですか？
  - [zh] subject `OTRAG` is an option — *manufacturer* — OTRAG火箭是由谁制造的？
  - [zh] subject `QB64` is an option — *developer* — QB64是由谁开发的？



---

## A (revised) — rigorous alignment via the entity→label direction

The first pass above mapped option strings to entities, which is ambiguous: a
label such as "Australia" resolves to both `Q408` (country) and `Q3960`
(continent), producing false mismatches. Redone in the well-defined direction
(each entity has exactly one canonical gold label per language, verified), and
resolving the English entity set only from labels that are unambiguous:

- facts whose English option set fully resolves: **23,120 / 74,282 (31.1%)** —
  the rest contain a distractor that never appears as a gold answer anywhere, so
  its entity is unrecoverable from the release.
- of those checkable facts, the option set differs from English in
  **0.40%–1.41%** by language (sw 92 … de 325).

Mismatches are a mix of orthographic variants of the SAME entity used as a
distractor (`St Peter's College` vs `St Peter’s College` — straight vs curly
apostrophe; zh `萨丁尼亚王国` vs `撒丁王国`; bn `এন্ডি ওয়ারহল` vs `অ্যান্ডি ওয়ারহল`) and
genuinely different entities (zh `牛津大学圣彼得学院` St Peter's College Oxford vs
`奥克兰圣彼得中学` St Peter's Auckland). Distractor strings are therefore NOT
normalised to the same canonical form as gold strings.

## Label collisions — one string, several entities

| lang | colliding labels | entities involved |
|---|---|---|
| en | 108 | 228 |
| de | 114 | 240 |
| ru | 138 | 289 |
| sw | 131 | 278 |

English examples: `Trinity College` → 4 entities; `Berlin` → `Q64`, `Q821245`;
`Geneva`, `Warsaw`, `Hamilton`, `Fernando Pérez` → 2–3 each.

## Split-level impact of the two behavioural defects

| split | n | answer leaked into question | ambiguous (question with >1 gold) | union |
|---|---|---|---|---|
| train | 70,567 | 2,020 (2.86%) | 482 (0.68%) | 2,501 (3.54%) |
| validation | 1,843 | 66 (3.58%) | 16 (0.87%) | 82 (4.45%) |
| **test** | **1,872** | **54 (2.88%)** | **5 (0.27%)** | **59 (3.15%)** |

Leakage by relation (test split): manufacturer 21.1%, developer 8.6%,
country 3.5%, official language 2.1%.

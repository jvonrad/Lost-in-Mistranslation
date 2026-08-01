# Validation-split review

## 1. Translation defects (sampled)

Same check the full test review ran, on a stratified sample of validation.

| lang | reviewed | ok | subject | type | relation | unsure | defect rate | 95% CI |
|---|---|---|---|---|---|---|---|---|
| sw | 300 | 278 | **2** | 15 | 5 | 0 | **7.3%** | [4.9%, 10.9%] |
| ru | 300 | 280 | **8** | 11 | 1 | 0 | **6.7%** | [4.4%, 10.1%] |
| ar | 300 | 281 | **3** | 15 | 1 | 0 | **6.3%** | [4.1%, 9.7%] |
| es | 300 | 283 | **7** | 6 | 2 | 2 | **5.0%** | [3.1%, 8.1%] |
| bn | 300 | 284 | **3** | 8 | 5 | 0 | **5.3%** | [3.3%, 8.5%] |
| zh | 300 | 286 | **5** | 4 | 3 | 2 | **4.0%** | [2.3%, 6.9%] |
| de | 300 | 287 | **8** | 3 | 0 | 2 | **3.7%** | [2.1%, 6.4%] |
| ja | 300 | 287 | **2** | 9 | 1 | 1 | **4.0%** | [2.3%, 6.9%] |
| id | 300 | 289 | **3** | 8 | 0 | 0 | **3.7%** | [2.1%, 6.4%] |
| fr | 300 | 289 | **7** | 3 | 1 | 0 | **3.7%** | [2.1%, 6.4%] |
| pt | 300 | 292 | **4** | 3 | 0 | 1 | **2.3%** | [1.1%, 4.7%] |
| **all** | **3,300** | 3,136 | **52** | 85 | 19 | 8 | **4.7%** | [4.1%, 5.5%] |

Test split, for comparison: **2.8%** over 16,687 items (full census, no sampling error).
**Do not compare these two numbers directly.** The test census and this validation pass used different reviewer prompts — the validation prompt names known bug classes and warns about morphology false positives, which makes its reviewers stricter. The two passes differ in split AND in prompt at once, so a gap between them is not evidence about the data. The control below separates the two.

## 2. English questions that contradict their own gold

The first measurement of this class. Every earlier pass used English as its reference and could not see it.

| verdict | n | share |
|---|---|---|
| ok | 568 | 94.7% |
| conflict | 18 | 3.0% |
| vague | 12 | 2.0% |
| unsure | 2 | 0.3% |

- **conflict rate: 3.0%** (95% CI [1.9%, 4.7%]) over 600 English questions
- extrapolated to the full release: roughly **1,130–2,782 facts** whose English question may contradict its gold

`vague` is reported separately and NOT counted as a conflict: an underspecified question has no wrong claim in it, it just may not have a unique answer.

### Confirmed English/gold conflicts

- `Q5004179|P17|Q30` — question reads as the famous Byodo-in in Uji, Japan (answer Japan); gold entity is the Hawaii replica
- `Q116962318|P170|Q191748` — question adds 'currently housed in the Germanisches Nationalmuseum'; the Cranach St Christopher (Q116962318) is in a private collection
- `Q1166556|P57|Q30876` — question says 1979 film; Yash Chopra's Darr is 1993
- `Q1964079|P407|Q13955` — Nalini is a Sanskrit/Indic given name; gold Arabic does not follow from the question
- `Q2095705|P50|Q33977` — question asks who authored 'the concept of traveling scholarships'; gold is Verne as author of the novel Traveling Scholarships
- `Q2255224|P170|Q301` — question says sculpture; the Laocoon sculpture is by Agesander/Athenodoros/Polydorus - El Greco made the painting
- `Q2530979|P178|Q2300984` — question says NES and Game Boy; gold entity is the 1999 Nintendo 64 Superman by Titus (the NES Superman was Kemco)
- `Q602121|P57|Q55392` — question says 1977 film; Louis Malle's Crackers is 1984
- `Q151960|P176|Q35953` — question says 'the Panther car'; gold entity is the German Panther tank built by MAN
- `Q1962105|P170|Q977546` — question omits 'film'; as written it reads as Hawking's book (answer Stephen Hawking), gold is the 1991 Errol Morris documentary
- `Q3232426|P57|Q314882` — question says 1964 film; Terence Young's Woman Hater is 1948
- `Q18602|P84|Q212487` — 22 Bishopsgate as built was designed by PLP Architecture; KPF designed the abandoned Pinnacle on the same site
- `Q369659|P407|Q150` — question names Poe's short story The Fall of the House of Usher, originally English; gold French belongs to the French-language work
- `Q3641710|P407|Q652` — question asks about boleros as a genre (Spanish); gold Italian belongs to Cristiano Malgioglio's 2006 album Boleros
- `Q6195056|P20|Q43199` — Jim Fowler (zoologist, Q6195056) died in Norwalk, Connecticut, not Omaha
- `Q11314023|P57|Q285908` — question says 1993 film; Ringo Lam's Looking for Mister Perfect is 2003
- `Q2455681|P176|Q290188` — question inverts the relation and reads as the poet; gold Ansaldo is the builder of the ship named Dante Alighieri
- `Q2572372|P17|Q30` — question invents 'Marc Andreessen's grandfather Carl Sprague'; subject is the settlement Sprague, Connecticut

## Severe (`subject`) translation defects in the sample: 52

- `Q2056949|P17|Q218` [de] — Bucharest Metro Line M1 rendered as 'die Festung M1' (a fortress) - different entity
- `Q18763236|P61|Q446449` [de] — asks who invented the first working telescope instead of who discovered asteroid 10707 Prunariu
- `Q20671331|P50|Q285048` [de] — names a different xkcd work 'Wie man einen Hund zeichnet' instead of 'Words for Pets'
- `Q1514066|P20|Q1022` [de] — 'German von Bohn' replaced by 'der deutsche Komponist Carl von Bohn' - different person
- `Q18614223|P50|Q285048` [de] — names 'Was, wenn? (What If?)' instead of the xkcd comic 'Road Rage'
- `Q123076|P61|Q562087` [de] — Palmyra Atoll replaced by 'die antike Stadt Palmyra' - different entity
- `Q2530979|P178|Q2300984` [de] — asks about the Nintendo 64 Superman game; English asks about the NES/Game Boy game
- `Q1073160|P407|Q1860` [de] — asks in what language the word 'Alptraum' arose instead of King's 'Nightmares & Dreamscapes'
- `Q961733|P170|Q1107006` [id] — the dbm database library is renamed 'basis data BDM' (scrambled acronym, 'library' dropped) - names a non-existent entity
- `Q5615340|P27|Q928` [id] — entirely different question: asks which citizenship can be acquired in Guinea-Bissau by naturalization, instead of Guia Gomez's country of citizenship
- `Q2095705|P50|Q33977` [id] — asks for the author of 'Perjalanan ke Pusat Bumi' (Journey to the Center of the Earth) instead of the English subject 'Traveling Scholarships'
- `Q11863164|P407|Q1860` [pt] — pt attributes 'Antes de Eden' to Ernest Hemingway; the subject is Arthur C. Clarke's 'Before Eden' - names a different work/author entity
- `Q6922099|P30|Q51` [pt] — asks where 'o explorador Joe McElroy' led expeditions instead of on which continent Mount McElroy is located - person substituted for the mountain
- `Q284964|P30|Q48` [pt] — subject dropped entirely: pt asks 'Qual e o maior continente do mundo?' (what is the world's largest continent) instead of where Cao is
- `Q2530979|P178|Q2300984` [pt] — pt names 'Superman: The New Superman Adventures', a different Titus game, not the NES/Game Boy 'Superman' of the English question
- `Q7710941|P407|Q1860` [ar] — subject entirely missing: 'ما هي لغة العمل أو لغة الاسم؟' names no work at all
- `Q7762423|P50|Q43736` [ar] — 'The Scarlet Gang of Asakusa' becomes 'عصابة أوساكا القرمزية' — Asakusa replaced by the different city Osaka
- `Q2095705|P50|Q33977` [ar] — asks about a different Verne work: 'مؤلف رواية حول العالم في ثمانين يومًا' (Around the World in Eighty Days) instead of Traveling Scholarships
- `Q63922|P37|Q150` [bn] — Vernier the Swiss municipality rendered as 'Vernier scale' (bharniyar skel) - the measuring instrument, a different real-world entity.
- `Q11863164|P407|Q1860` [bn] — Title 'Before Eden' translated literally and pluralised into a common phrase: 'what was the language of the literary works composed before Eden?' - no longer names the story.
- `Q1724973|P57|Q1569097` [bn] — Title truncated from 'Sangdil Sanam' (1994) to 'Sangdil', which is a different, well-known 1952 film; phrasing also garbled.
- `Q5968417|P407|Q1321` [sw] — subject deleted: 'Lugha gani inahusishwa na tamaduni zinazochunguza maisha dhidi ya kifo?' = which language is associated with cultures exploring life vs death; the work 'La vida contra la muerte' is never named
- `Q174548|P50|Q23873` [sw] — 'Mater et Magistra' replaced by a book called 'Amebo' - a completely different, unrelated title
- `Q5304678|P57|Q2966568` [es] — 'Dracula's Widow' (1988) replaced by 'Condesa Drácula' = Countess Dracula (1971, Peter Sasdy), a different film
- `Q63922|P37|Q150` [es] — Vernier (Geneva, Switzerland) described as a town in Haute-Savoie, France, and the question then asks for the official language of FRANCE, not of Vernier
- `Q2582536|P84|Q312838` [es] — Cidade das Artes Bibi Ferreira (Rio de Janeiro) replaced by 'La Ciudad de las Artes y las Ciencias en Valencia, España' - a different building
- `Q6922099|P30|Q51` [es] — Mount McElroy replaced by 'la Estación McMurdo', a different entity
- `Q5968417|P407|Q1321` [es] — work 'La vida contra la muerte' falsely attributed to José Saramago, naming a nonexistent/different work
- `Q2530979|P178|Q2300984` [es] — English asks about the Superman game for NES/Game Boy; Spanish asks about 'el videojuego de Superman para Sega Genesis en 1993', a different game
- `Q1073160|P407|Q1860` [es] — 'Nightmares & Dreamscapes' (Stephen King) rendered as 'Pesadillas y alucinaciones' de Charles Dickens - wrong author/work

## 3. Control: is validation actually dirtier, or were these reviewers stricter?

The validation pass differs from the test census in TWO ways at once — different split and a stricter prompt. To separate them, the identical validation prompt was run over 300 **test** facts in four languages. Any gap between that control and the original test census is pure reviewer effect; any gap between the control and validation is a real split difference.

| lang | test, original prompt | test, strict prompt (control) | validation, strict prompt | CIs overlap? |
|---|---|---|---|---|
| de | 1.5% | **4.0%** [2.3, 6.9] | **3.7%** [2.1, 6.4] | yes |
| es | 2.8% | **2.3%** [1.1, 4.7] | **5.0%** [3.1, 8.1] | yes |
| ru | 3.4% | **5.3%** [3.3, 8.5] | **6.7%** [4.4, 10.1] | yes |
| sw | 4.7% | **5.7%** [3.6, 8.9] | **7.3%** [4.9, 10.9] | yes |
| **pooled** | 3.1% | **4.3%** [3.3, 5.6] | **5.7%** [4.5, 7.1] | yes |

**Conclusion: validation is not meaningfully dirtier than test.** Pooled CIs overlap ([3.3, 5.6] vs [4.5, 7.1]), and they overlap in all four languages individually. The apparent gap against the 2.8% census was an artefact of comparing two different reviewer prompts.

**The corollary matters more: the original test census under-counts.** The same reviewers, given a prompt that names known bug classes and warns about morphology false positives, find 4.3% on the *same test data* the original pass scored at 3.1% — about 1.4x more. So the published 2.8% test figure, and the 455 re-translations driven by it, both rest on a lower bound rather than a full accounting. A rerun of the full test census with the strict prompt would likely surface a few hundred more defective items.

Reviewer noise is small by comparison: German was run twice on the identical file and scored 4.3% and 4.0%, so ±0.3pp.


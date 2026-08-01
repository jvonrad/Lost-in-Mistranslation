# Hallucination rate in PolyFact-Clean's unverifiable labels

Wikidata attests 137,285 of the 193,188 answer labels via its label, a language variant, a Wikipedia sitelink or an alias. The remaining **55,903** are unverifiable model translations and are the only place a hallucination can hide. 600 of them were reviewed by hand.

## Measured error rates

| stratum | pool | reviewed | tier A (wrong referent) | rate | 95% CI | tier B (imprecise) |
|---|---|---|---|---|---|---|
| semantic | 7,702 | 420 | 25 | **5.95%** | [4.06%, 8.64%] | 11 (2.6%) |
| translit | 48,201 | 180 | 0 | **0.00%** | [0.00%, 2.09%] | 2 (1.1%) |

Tier A = the label denotes something other than the entity, or is not a word. Tier B = right entity, imprecise wording. Only tier A can make a correct model answer score as wrong.

## Extrapolated to the whole dataset

- bad labels in the unverifiable pool: **458** of 55,903 (**0.82%**), 95% CI [313, 1,673]
- as a share of ALL 193,188 answer labels: **0.24%**, 95% CI [0.16%, 0.87%]

## Fact-level exposure

| stratum | mean facts/entity, all sampled | mean facts/entity, tier-A errors |
|---|---|---|
| semantic | 1.76 | 1.96 |
| translit | 1.77 | — |

- expected (fact, language) items whose gold label is wrong: **899** of 722,028 (**0.124%**), 95% upper bound 3,089 (0.428%)
- i.e. roughly **75 facts' worth** of damage spread across 12 languages; no fact is wrong in all 12, since the English label is attested for 99.6% of entities

## Where the errors are

| language | tier-A errors found | | relation | tier-A errors found |
|---|---|---|---|---|
| sw | 9 | | developer | 11 |
| zh | 9 | | educated at | 4 |
| ar | 3 | | manufacturer | 3 |
| bn | 2 | | creator | 3 |
| ru | 1 | | architect | 2 |
| ja | 1 | | place of death | 1 |
|  |  | | author | 1 |

## Every tier-A error found

- **Q245456 5th Cell** [ar, developer, 3 facts] — ar 'الخامسة خلية' — studio name translated literally, ungrammatical
- **Q722976 Antonov plant** [sw, manufacturer, 1 facts] — sw 'Kiwi ya…' — 'Kiwanda' (factory) garbled to 'Kiwi'
- **Q2065466 Terminal Reality** [zh, developer, 8 facts] — zh '终端现实' — studio name translated literally
- **Q601299 Video System** [sw, developer, 3 facts] — sw 'Mfumo wa Video' — studio name translated literally
- **Q1616636 Hewett Watson** [zh, creator, 2 facts] — zh '休·沃森' = Hugh Watson — wrong given name
- **Q2001088 Northern State University** [ar, educated at, 1 facts] — ar = 'North Dakota State University' — different institution
- **Q659918 Horn (place)** [sw, place of death, 1 facts] — sw 'Pembe' = an animal's horn
- **Q11325179 Nude Maker** [sw, developer, 1 facts] — sw 'Mtengenezaji wa Uchi' = maker of nakedness
- **Q3003137 Croc** [zh, creator, 1 facts] — zh '鳄鱼' = crocodile
- **Q252733 Object Management Group** [sw, developer, 2 facts] — sw 'Vituko' = antics, not objects
- **Q3026228 Propaganda Games** [zh, developer, 1 facts] — zh '宣传游戏' — studio name translated literally
- **Q1778277 Odense Steel Shipyard** [sw, manufacturer, 3 facts] — sw 'Umelodi wa Chuma' — not a Swahili word for shipyard
- **Q372608 University of Basel** [bn, educated at, 2 facts] — bn 'বার্ল' = Barl — misspelt city
- **Q3569449 Wisdom Tree** [ru, developer, 1 facts] — ru 'Древо мудрости' — studio name translated literally
- **Q3064033 Łucznik Arms Factory** [sw, manufacturer, 3 facts] — sw 'Kiwiya' — 'Kiwanda' garbled again
- **Q3851105 Silver (person)** [bn, author, 1 facts] — bn 'রূপা' = the metal silver
- **Q778568 Asam brothers** [ja, architect, 1 facts] — ja '浅井兄弟' = Asai brothers, a Japanese surname
- **Q245456 5th Cell** [zh, developer, 3 facts] — zh '第五单元游戏公司' — name translated + 'game company' appended
- **Q697289 National University of Tainan** [ar, educated at, 1 facts] — ar = 'National University Taiwan' — different institution
- **Q464476 Black Flag (band)** [sw, creator, 1 facts] — sw 'Bendera Nyeusi' = a black flag
- **Q3340630 Nicolas Nicole** [zh, architect, 1 facts] — zh '尼古拉斯·尼古拉斯' = Nicolas Nicolas — surname replaced
- **Q2994578 Conservatory of Nice** [sw, educated at, 1 facts] — sw 'Hifadhi' = a nature reserve
- **Q2745586 Genius Sonority** [zh, developer, 3 facts] — zh '天才音速' = genius sound-speed
- **Q4348557 Microcabin** [zh, developer, 3 facts] — zh '微仓' = micro warehouse
- **Q13117583 Star Theory Games** [zh, developer, 1 facts] — zh '星理论游戏' — studio name translated literally

## Tier-B (imprecise but identifiable)

- Q1165635 Lowell Observatory [sw] — sw 'Chuo cha Utafiti' = research college
- Q1329478 Czech Technical University [bn] — bn drops 'Czech' -> 'Prague Technical University'
- Q65560029 Jacques Delrue [ru] — ru 'Делюр' — drops the r
- Q421739 Nat. Univ. of Distance Education [id] — id 'Terbuka' = open, not distance
- Q511291 Royal Academy of Art [sw] — sw 'Chuo Kikuu' = university, not academy
- Q2822452 Royal Academy of Arts of Liège [sw] — sw 'Chuo Kikuu' = university
- Q4666924 Aberdeen Grammar School [ar] — ar/id 'grammatical school'
- Q265058 Hungarian Academy of Sciences [sw] — sw 'Chuo Kikuu' = university
- Q9391434 United Kingdom of Poland [zh] — zh drops 'United'
- Q7571404 Southwestern College [sw] — sw name translated + college->university
- Q216941 Xi River [bn] — bn 'সি' = Si
- Q2660091 Griptonite Games [bn] — bn 'Gripstone Games'
- Q61793216 Triband [es] — es 'Tribanda'


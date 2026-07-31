# Translation-artifact audit — `jvonrad/PolyFact-Clean`

## 1. Gold answer identical to English, by relation

High values are expected where the object is a proper noun; they are a red flag where the object is a common noun (marked ⚠).

| relation | de | id | pt | ar | bn | sw | es | ru | fr | ja | zh | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| developer | 82% | 83% | 82% | 3% | 1% | 88% | 83% | 71% | 80% | 13% | 20% | **55%** |
| director | 93% | 98% | 96% | 0% | 1% | 100% | 95% | 0% | 95% | 0% | 0% | **53%** |
| creator | 93% | 94% | 93% | 0% | 0% | 96% | 92% | 1% | 90% | 0% | 0% | **51%** |
| architect | 82% | 94% | 80% | 1% | 1% | 96% | 84% | 7% | 83% | 1% | 2% | **48%** |
| author | 84% | 88% | 83% | 0% | 1% | 91% | 82% | 0% | 84% | 0% | 0% | **47%** |
| manufacturer | 65% | 75% | 71% | 2% | 2% | 79% | 72% | 57% | 66% | 3% | 2% | **45%** |
| discoverer or inventor | 40% | 81% | 88% | 1% | 1% | 40% | 85% | 13% | 87% | 1% | 0% | **40%** |
| place of death | 75% | 77% | 61% | 0% | 0% | 79% | 55% | 0% | 71% | 0% | 0% | **38%** |
| continent ⚠ | 0% | 45% | 0% | 0% | 0% | 45% | 45% | 0% | 38% | 0% | 0% | **16%** |
| employer | 39% | 18% | 17% | 0% | 1% | 12% | 14% | 6% | 15% | 1% | 1% | **11%** |
| educated at | 44% | 17% | 18% | 0% | 0% | 8% | 13% | 1% | 15% | 0% | 0% | **11%** |
| country | 13% | 19% | 6% | 0% | 0% | 22% | 14% | 0% | 22% | 0% | 0% | **9%** |
| currency | 2% | 9% | 17% | 0% | 0% | 15% | 7% | 0% | 20% | 0% | 0% | **6%** |
| country of citizenship | 7% | 12% | 4% | 0% | 0% | 14% | 10% | 0% | 16% | 0% | 0% | **6%** |
| official language | 2% | 1% | 0% | 0% | 0% | 1% | 0% | 0% | 0% | 0% | 0% | **0%** |
| language of work or name | 1% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | **0%** |

## 2. Facts never translated in ANY language

- gold identical to English in **all 11** non-English languages: **4 / 82,930 (0.0%)**
- distribution of how many languages share the English gold:

| #languages identical to en | facts |
|---|---|
| 0 | 27,302 |
| 1 | 9,054 |
| 2 | 2,696 |
| 3 | 5,788 |
| 4 | 5,972 |
| 5 | 4,571 |
| 6 | 21,289 |
| 7 | 4,721 |
| 8 | 1,097 |
| 9 | 347 |
| 10 | 89 |
| 11 | 4 |

Relations most affected (identical in all 11 languages):

| relation | facts |
|---|---|
| architect | 3 |
| developer | 1 |

## 3. Latin-script gold answers in non-Latin-script languages

| lang | share of gold answers in Latin script |
|---|---|
| ar | 0.5% |
| zh | 3.5% |
| bn | 0.8% |
| ru | 13.7% |
| ja | 3.1% |

**ar examples** (subject — relation → gold / English gold):

- D4 — *developer* → `Access Games` / en `Access Games`
- CastleStorm — *developer* → `Zen Studios` / en `Zen Studios`
- Wizkid: The Story of Wizball II — *developer* → `Sensible Software` / en `Sensible Software`
- Disorder 6 — *developer* → `5pb.` / en `5pb. Inc.`
- Trabant — *manufacturer* → `HQM Sachsenring GmbH` / en `HQM Sachsenring GmbH`
- Scott Horton — *employer* → `Antiwar.com` / en `Antiwar.com`

**zh examples** (subject — relation → gold / English gold):

- Broadgate Tower — *architect* → `SOM建筑设计事务所` / en `Skidmore, Owings & Merrill`
- Infinity Engine — *developer* → `BioWare` / en `BioWare`
- Shell Energy Stadium — *architect* → `Populous` / en `Populous`
- D4 — *developer* → `Access游戏` / en `Access Games`
- U-710 — *manufacturer* → `H. C. Stülcken Sohn` / en `H. C. Stülcken Sohn`
- Aquapazza: Aquaplus Dream Match — *developer* → `Examu` / en `Examu`

**bn examples** (subject — relation → gold / English gold):

- Dungeon Keeper 3 — *developer* → `bullfrog প্রোডাকশনস` / en `Bullfrog Productions`
- (17507) 1992 HH5 — *discoverer or inventor* → `Henri Debehogne` / en `Henri Debehogne`
- (13008) 1984 SE6 — *discoverer or inventor* → `Henri Debehogne` / en `Henri Debehogne`
- Abraham Bredius — *employer* → `Mauritshuis` / en `Mauritshuis`
- SOM — *developer* → `TÜBİTAK প্রতিরক্ষা শিল্প গবেষণা ও উন্নয়ন ইনস্টিটিউট` / en `TÜBİTAK Defense Industries Research and Development Institute`
- Disorder 6 — *developer* → `5pb. Inc.` / en `5pb. Inc.`

**ru examples** (subject — relation → gold / English gold):

- Broadgate Tower — *architect* → `Skidmore, Owings and Merrill` / en `Skidmore, Owings & Merrill`
- Emanuele Pessagno — *manufacturer* → `Cantieri Navali del Tirreno Riuniti` / en `Cantieri Navali del Tirreno Riuniti`
- USS Ralph Talbot — *manufacturer* → `Boston Navy Yard` / en `Boston Navy Yard`
- Dungeon Keeper 3 — *developer* → `Bullfrog Productions` / en `Bullfrog Productions`
- Infinity Engine — *developer* → `BioWare` / en `BioWare`
- Shell Energy Stadium — *architect* → `Populous` / en `Populous`

**ja examples** (subject — relation → gold / English gold):

- Shell Energy Stadium — *architect* → `POPULOUS` / en `Populous`
- U-185 — *manufacturer* → `AGヴェーザー` / en `AG Weser`
- Luis Pérez-Sala — *employer* → `HRT F1` / en `HRT Formula One Team`
- Yak-25 — *developer* → `A・S・ヤコヴレフ記念試作設計局` / en `Yakovlev`
- Perry Mason — *creator* → `E・S・ガードナー` / en `Erle Stanley Gardner`
- The Childhood of Jesus — *author* → `J・M・クッツェー` / en `J. M. Coetzee`


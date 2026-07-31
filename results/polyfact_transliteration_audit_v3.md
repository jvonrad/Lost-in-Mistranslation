# Transliteration / label-normalization audit — `jvonrad/PolyFact-Clean`

## 1. Same entity, different label within one language

An `object_id` appearing with more than one distinct gold string. `case/space only` counts those whose variants differ only by casing or whitespace (trivially normalizable); the rest are genuine variant labels.

| lang | entities with >1 label | distinct entities | share | of which case/space only |
|---|---|---|---|---|
| en | 0 | 18,601 | 0.00% | 0 |
| de | 0 | 18,601 | 0.00% | 0 |
| id | 0 | 18,601 | 0.00% | 0 |
| pt | 0 | 18,601 | 0.00% | 0 |
| ar | 0 | 18,601 | 0.00% | 0 |
| bn | 0 | 18,601 | 0.00% | 0 |
| sw | 0 | 18,601 | 0.00% | 0 |
| es | 0 | 18,601 | 0.00% | 0 |
| ru | 0 | 18,601 | 0.00% | 0 |
| fr | 0 | 18,601 | 0.00% | 0 |
| ja | 0 | 18,601 | 0.00% | 0 |
| zh | 0 | 18,601 | 0.00% | 0 |

## 2. Mixed-script gold labels (non-Latin languages)

A single label containing Latin characters alongside the native script — a partially transliterated entity name.

| lang | mixed-script golds | total | share |
|---|---|---|---|
| ar | 98 | 74,282 | 0.13% |
| bn | 996 | 74,282 | 1.34% |
| ru | 429 | 74,282 | 0.58% |
| ja | 1,956 | 74,282 | 2.63% |
| zh | 1,873 | 74,282 | 2.52% |

**ar examples** (subject — relation → gold):

- Glock 36 — *manufacturer* → `غلوك Ges.mbH`
- (13741) 1998 SH10 — *discoverer or inventor* → `مسح الكويكبات OCA-DLR`
- Cuphead — *developer* → `استوديو MDHR`
- 111561 Giovanniallevi — *discoverer or inventor* → `مسح الكويكبات أسياجو-DLR`
- Glock 31C — *manufacturer* → `غلوك Ges.mbH`
- Kim Won-bong — *educated at* → `الأكاديمية العسكرية whampoa`

**bn examples** (subject — relation → gold):

- Emanuele Pessagno — *manufacturer* → `কান্টিয়েরি নাভাল del তিরেনো riuniti`
- Dungeon Keeper 3 — *developer* → `bullfrog প্রোডাকশনস`
- (85254) 1993 TG12 — *discoverer or inventor* → `হেনরি ই. Holt`
- (19959) 1985 UJ3 — *discoverer or inventor* → `ক্লাস-ইংভার Lagerkvist`
- 130 William — *architect* → `ডেভিড adjacency`
- information architecture — *discoverer or inventor* → `রিচার্ড Saul ওয়ারম্যান`

**ru examples** (subject — relation → gold):

- Alain de Chambure — *place of death* → `XV округ Парижа`
- Breda Ba.65 — *manufacturer* → `Breda (компания)`
- (32861) 1993 FM7 — *discoverer or inventor* → `Астероидный и кометный обзор Уппсала-ESO`
- Madeleine Suffel — *place of death* → `VII округ Парижа`
- Siemens S65 — *manufacturer* → `Сотовые телефоны Siemens`
- Louis Heuzé — *place of death* → `X округ Парижа`

**ja examples** (subject — relation → gold):

- U-185 — *manufacturer* → `AGヴェーザー`
- 15965 Robertcox — *discoverer or inventor* → `ジェームズ・M・ロー`
- (85254) 1993 TG12 — *discoverer or inventor* → `ヘンリー・E・ホルト`
- Yak-25 — *developer* → `A・S・ヤコヴレフ記念試作設計局`
- The Devil Pays Off — *director* → `ジョン・H・オーア`
- The One with The Girl Who Hits Joey — *director* → `ケヴィン・S・ブライト`

**zh examples** (subject — relation → gold):

- Broadgate Tower — *architect* → `SOM建筑设计事务所`
- D4 — *developer* → `Access游戏`
- 15965 Robertcox — *discoverer or inventor* → `詹姆斯·M·罗伊`
- (85254) 1993 TG12 — *discoverer or inventor* → `亨利·E·霍尔特`
- The Devil Pays Off — *director* → `约翰·H·奥尔`
- The One with The Girl Who Hits Joey — *director* → `凯文·S·布赖特`


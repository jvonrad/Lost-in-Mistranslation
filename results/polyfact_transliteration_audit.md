# Transliteration / label-normalization audit — `jvonrad/PolyFact-Clean`

## 1. Same entity, different label within one language

An `object_id` appearing with more than one distinct gold string. `case/space only` counts those whose variants differ only by casing or whitespace (trivially normalizable); the rest are genuine variant labels.

| lang | entities with >1 label | distinct entities | share | of which case/space only |
|---|---|---|---|---|
| en | 2 | 20,150 | 0.01% | 0 |
| de | 1 | 20,150 | 0.00% | 0 |
| id | 1 | 20,150 | 0.00% | 0 |
| pt | 0 | 20,150 | 0.00% | 0 |
| ar | 1 | 20,150 | 0.00% | 0 |
| bn | 2 | 20,150 | 0.01% | 0 |
| sw | 1 | 20,150 | 0.00% | 0 |
| es | 0 | 20,150 | 0.00% | 0 |
| ru | 3 | 20,150 | 0.01% | 0 |
| fr | 1 | 20,150 | 0.00% | 0 |
| ja | 1 | 20,150 | 0.00% | 0 |
| zh | 0 | 20,150 | 0.00% | 0 |

**ar examples:**

- `Q51`: `القارة القطبية الجنوبية` ×167 | `أمريكا الجنوبية` ×1

**bn examples:**

- `Q48`: `এশিয়া` ×2351 | `দক্ষিণ আমেরিকা` ×1
- `Q51`: `অ্যান্টার্কটিকা` ×167 | `দক্ষিণ আমেরিকা` ×1

**ru examples:**

- `Q48`: `Азия` ×2351 | `Северная Америка` ×1
- `Q1321`: `испанский язык` ×690 | `нюнорск` ×1
- `Q51`: `Антарктида` ×167 | `Северная Америка` ×1

**ja examples:**

- `Q55`: `オランダ` ×49 | `バルバドス` ×1

## 2. Mixed-script gold labels (non-Latin languages)

A single label containing Latin characters alongside the native script — a partially transliterated entity name.

| lang | mixed-script golds | total | share |
|---|---|---|---|
| ar | 173 | 83,006 | 0.21% |
| bn | 4,173 | 83,006 | 5.03% |
| ru | 479 | 83,006 | 0.58% |
| ja | 2,033 | 83,006 | 2.45% |
| zh | 1,937 | 83,006 | 2.33% |

**ar examples** (subject — relation → gold):

- Glock 36 — *manufacturer* → `غلوك Ges.mbH`
- (13741) 1998 SH10 — *discoverer or inventor* → `مسح الكويكبات OCA-DLR`
- Cuphead — *developer* → `استوديو MDHR`
- Stanisław Jackowski — *educated at* → `أكاديمية يان ماtejكو للفنون الجميلة في كراكوف`
- El alma no tiene color — *creator* → `خوسيلito رودريغيز`
- 111561 Giovanniallevi — *discoverer or inventor* → `مسح الكويكبات أسياجو-DLR`

**bn examples** (subject — relation → gold):

- Emanuele Pessagno — *manufacturer* → `কান্টিয়েরি নাভাল del তিরেনো riuniti`
- Dungeon Keeper 3 — *developer* → `bullfrog প্রোডাকশনস`
- Sant Ot — *architect* → `এমিলি বোর্দoy ই আলকানতারা`
- (72292) 2001 BE22 — *discoverer or inventor* → `লিঙ্কনNear-Earth গ্রহাণু গবেষণা`
- (85254) 1993 TG12 — *discoverer or inventor* → `হেনরি ই. Holt`
- (469722) 2005 LP40 — *discoverer or inventor* → `লিঙ্কনNear-Earth গ্রহাণু গবেষণা`

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


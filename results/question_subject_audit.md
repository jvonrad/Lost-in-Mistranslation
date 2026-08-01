# Does each question name its stored subject? — `/scratch/u6jh/jvonrad.u6jh/v9`

The subject's Wikidata name in the target language is the reference; a question that contains none of its attested names (label, aliases, Wikipedia title, language variants) is not asking about the stored entity. Facts whose subject has no attested name in a language cannot be checked and are excluded from the rate rather than counted as either.

| lang | checkable | question omits the subject | rate | uncheckable |
|---|---|---|---|---|
| en | 59,936 | 737 | **1.2%** | 44 |
| de | 54,267 | 4,820 | **8.9%** | 5,713 |
| id | 18,474 | 880 | **4.8%** | 41,506 |
| pt | 41,996 | 2,110 | **5.0%** | 17,984 |
| ar | 19,529 | 2,284 | **11.7%** | 40,451 |
| bn | 5,358 | 1,243 | **23.2%** | 54,622 |
| sw | 6,366 | 818 | **12.8%** | 53,614 |
| es | 54,012 | 2,831 | **5.2%** | 5,968 |
| ru | 35,249 | 8,090 | **23.0%** | 24,731 |
| fr | 57,678 | 4,010 | **7.0%** | 2,302 |
| ja | 28,644 | 2,425 | **8.5%** | 31,336 |
| zh | 31,471 | 2,559 | **8.1%** | 28,509 |

Overall **32,807 / 412,980 (7.9%)** checkable (fact, language) items have a question that never names the stored subject.

A miss is not automatically an error: the generator may have used a name Wikidata does not list. The flagged items are written to the JSON for manual review, which is what settles the actual rate.

## Flagged items by relation (sample)

| relation | flagged |
|---|---|
| language of work or name | 968 |
| architect | 556 |
| creator | 411 |
| official language | 294 |
| author | 254 |
| discoverer or inventor | 247 |
| country of citizenship | 223 |
| director | 217 |
| country | 201 |
| continent | 187 |

## Examples

- [fr] subject `Roberto "Junior" Maldonado` (wikidata: Roberto "Junior" Maldonado)
  - Q: Où Roberto 'Junior' Maldonado a-t-il fait ses études ?
- [ru] subject `instant film` (wikidata: Одноступенный фотопроцесс, Одноступенный фотопроцесс)
  - Q: Кто является первооткрывателем или изобретателем одноступенного фотопроцесса?
- [zh] subject `Palazzo Vecchio` (wikidata: 韋基奧宮, 维奇奥宫)
  - Q: 旧宫的建筑师是谁？
- [zh] subject `Little Children` (wikidata: 身為人母, 身为人母)
  - Q: 电影《小丑》的导演是谁？
- [ru] subject `Hirshhorn Museum and Sculpture Garden` (wikidata: Музей и сад скульптур Хиршхорна, Музей Хиршхорн и сад скульптур)
  - Q: Кто является архитектором Музея и сада скульптур Хиршхорна?
- [ru] subject `Björn Rosendahl` (wikidata: Бьёрн Русендаль, Русендаль, Бьёрн)
  - Q: Какое гражданство у Бьёрна Русендаля?
- [ru] subject `Ultima Tower` (wikidata: Башня Ультима, Ultima Tower)
  - Q: Кто является архитектором Башни Ультима?
- [ar] subject `U-185` (wikidata: الغواصة الألمانية يو-185, الغواصة الألمانية يو-185)
  - Q: من هي الشركة المصنعة للغواصة الألمانية يو-185؟
- [en] subject `D4` (wikidata: D4: Dark Dreams Don't Die, D4: Dark Dreams Don't Die)
  - Q: Which company developed the video game D4?
- [fr] subject `D4` (wikidata: D4: Dark Dreams Don't Die, D4: Dark Dreams Don't Die)
  - Q: Quel studio a développé le jeu D4 ?
- [ar] subject `U-710` (wikidata: الغواصة الألمانية يو-710, الغواصة الألمانية يو-710)
  - Q: من هي الشركة المصنعة للغواصة الألمانية يو-710؟
- [de] subject `15965 Robertcox` (wikidata: (15965) Robertcox)
  - Q: Wer gilt als der Erfinder des ersten funktionsfähigen Dampfboots?
- [es] subject `15965 Robertcox` (wikidata: (15965) Robertcox)
  - Q: Según la información, ¿quién es considerado el descubridor o inventor de Robertcox?
- [ru] subject `15965 Robertcox` (wikidata: (15965) Robertcox, 1998 DU7)
  - Q: Кто является первооткрывателем или изобретателем, связанным с Робертом Коксом?
- [fr] subject `15965 Robertcox` (wikidata: (15965) Robertcox, (15965) Robertcox)
  - Q: Par qui Robertcox a-t-il été découvert ou inventé ?
- [ru] subject `Alberuela de Tubo` (wikidata: Альберуэла-де-Тубо, Альберуэла-де-Тубо)
  - Q: Какой язык является официальным языком Альберуэлы-де-Тубо?
- [pt] subject `2015 Copa Libertadores Finals` (wikidata: Final da Copa Libertadores da América de 2015, Final da Copa Libertadores da América de 2015)
  - Q: Em que língua está escrito o nome da competição 'Copa Libertadores da América'?
- [de] subject `Stolperstein dedicated to Rudolph Raphael Simon` (wikidata: Stolperstein für Rudolph Raphael Simon)
  - Q: Wer ist der Urheber des Stolpersteins für Rudolph Raphael Simon?
- [fr] subject `Stolperstein dedicated to Rudolph Raphael Simon` (wikidata: Stolperstein à la mémoire de Rudolph Raphael Simon)
  - Q: Par qui les Stolpersteine à la mémoire de Rudolph Raphael Simon ont-ils été créés ?
- [ar] subject `Furrer` (wikidata: Furrer)
  - Q: ما هي لغة العمل أو لغة الاسم لـ فورر؟
- [bn] subject `Furrer` (wikidata: Furrer)
  - Q: ফুরrer শব্দটির মূল ভাষা কি?
- [ja] subject `Furrer` (wikidata: Furrer)
  - Q: フローラーの作品または名前は、主にどの言語で書かれていますか？
- [fr] subject `Las Omañas` (wikidata: Las Omañas, Las Omañas)
  - Q: Dans quel pays se trouvent les Omañas ?
- [ru] subject `Fernando Godoy` (wikidata: Фернандо Габриэль Годой, Фернандо Габриэль Годой)
  - Q: Какое гражданство у Фернандо Габриэля Годоя?
- [ar] subject `Antoniewski` (wikidata: Antoniewski)
  - Q: ما هي لغة العمل أو لغة الاسم الخاصة بأنطونيفسكي؟


#!/usr/bin/env python
"""Per-language appendix tables: PolyFact, KLAR, Global-MMLU-Lite, BMLAMA-53.

Both model families in one table, each as a block: Baseline / Aligned (CPT),
then GRPO, SFT, DCO and CM-Align, each paired with its CPT-initialised variant
where one was trained.

Three structural corrections to the published tables, none of them numerical:

  Column order.  The published PolyFact table printed its values in the
  evaluator's native key order (en, de, id, pt, ar, bn, sw, es, ru, fr, ja, zh)
  under a header written in a different order, so every column from `es`
  onwards was mislabelled. Headers here are generated from the same list the
  values are indexed by, so the two cannot drift apart again.

  KLAR grouping.  Arabic is one of the SEVEN languages present in
  post-training (en, es, fr, ru, zh, ja, ar), so it belongs in the Seen block.
  The published table puts it under Unseen, making the split read 6/11.

  Absent language configs.  Global-MMLU-Lite ships no `ru` and BMLAMA-53 no
  `sw`; those columns are dropped rather than filled. `High` is therefore the
  mean over the high-resource languages actually present, which differs per
  benchmark (9 on PolyFact, 8 on Global-MMLU).

Usage:
  python data_analysis/emit_appendix_tables.py
  python data_analysis/emit_appendix_tables.py --table klar --grpo attmlp
"""
from __future__ import annotations

import argparse
import io
import json
import os

LOW = {"id", "bn", "sw"}
PF_ORDER = ["en", "de", "es", "fr", "pt", "id", "ru", "zh", "ar", "ja", "sw", "bn"]
# The seven languages seen in post-training; ar included (see module docstring).
KLAR_SEEN = ["en", "es", "fr", "ru", "zh", "ja", "ar"]
KLAR_OOD = ["ca", "el", "fa", "he", "hu", "ko", "nl", "tr", "uk", "vi"]

# GRPO checkpoint per family. OLMo `ladder` is the strongest on KLAR (33.0/19.2)
# and the one Figures 4(a)/4(b) and Sec. 5.3 use; `attmlp` is the older
# checkpoint, which scores at/below baseline on KLAR.
OLMO_GRPO = {"ladder": "olmo-ladder10k-s6500", "attmlp": "olmo-grpo-attmlp"}

FAMILIES = [
    ("OLMo-2-1124-7B", lambda g: [
        ("Baseline", "olmo-base"),
        ("Aligned", "olmo-finetranslations"),
        None,
        ("GRPO", OLMO_GRPO[g]),
        ("Aligned + GRPO",
         "olmo-2-7b-finetranslation-wikifact-grpo-att-mlp-checkpoint"),
        None,
        ("SFT", "olmo-sft-10k"),
        ("Aligned + SFT", "olmo-cpt-sft-10k"),
        None,
        ("DCO", "olmo-dco-10k"),
        ("Aligned + DCO", "olmo-cpt-dco-10k"),
        None,
        ("CM-Align", "olmo-cmalign-10k"),
    ]),
    ("Qwen-2.5-7B", lambda g: [
        ("Baseline", "qwen-base"),
        ("Aligned", "Qwen-2.5-7B-TED"),
        None,
        ("GRPO", "qwen-sw-clip5"),
        ("Aligned + GRPO", "Qwen-2.5-7B-TED-grpo"),
        None,
        ("SFT", "qwen-sft-10k"),
        ("Aligned + SFT", "qwen-cpt-sft-10k"),
        None,
        ("DCO", "qwen-dco-10k"),
        ("Aligned + DCO", "qwen-cpt-dco-10k"),
        None,
        ("CM-Align", "qwen-cmalign-10k"),
    ]),
]


def mcq(bench):
    def load(tag):
        p = f"results/{tag}_{bench}_consistency.json"
        if not os.path.exists(p):
            return None
        d = json.load(io.open(p, encoding="utf-8"))["per_language_accuracy"]
        return {k: 100 * v["accuracy"] for k, v in d.items()}
    return load


def klar(tag):
    p = f"results/klar/{tag}_klar_alllangs.json"
    if not os.path.exists(p):
        return None
    d = json.load(io.open(p, encoding="utf-8"))["per_lang"]
    return {k: 100 * v[0] / v[1] for k, v in d.items()}


def emit(grpo, langs, load, prec, caption, label, high=False, split=None):
    blocks = []
    for title, mk in FAMILIES:
        rows = []
        for r in mk(grpo):
            if r is None:
                rows.append(None)
                continue
            v = load(r[1])
            if v is None:
                print(f"%% no data, row dropped: {title} / {r[0]} ({r[1]})")
                continue
            rows.append((r[0], v))
        while rows and rows[-1] is None:
            rows.pop()
        blocks.append((title, rows))

    vals = [v for _, rows in blocks for r in rows if r for v in [r[1]]]
    present = [l for l in langs if all(l in v for v in vals)]
    dropped = [l for l in langs if l not in present]
    if dropped:
        print(f"%% no config for: {', '.join(dropped)} -- column omitted")

    ncol = len(present) + (1 if high else 0)
    if split is None:
        ns = None
        colspec = "l" + "c" * ncol
        pre = ""
        head = ("Model \n& " + " & ".join(present)
                + (" \n& High" if high else "") + r" \\")
    else:
        ns = sum(1 for l in present if l in split)
        no = len(present) - ns
        colspec = "l" + "c" * ns + "|" + "c" * no
        pre = (f"& \\multicolumn{{{ns}}}{{c}}{{Seen}} & "
               f"\\multicolumn{{{no}}}{{c}}{{Unseen}} \\\\\n"
               f"\\cmidrule(lr){{2-{ns + 1}}} \\cmidrule(lr){{{ns + 2}-{ncol + 1}}}\n")
        head = ("Model \n& " + " & ".join(present[:ns]) + " \n& "
                + " & ".join(present[ns:]) + r" \\")

    out = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
           r"\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{" + colspec + "}", r"\toprule",
           pre + head]
    for bi, (title, rows) in enumerate(blocks):
        out.append(r"\midrule" if bi == 0 else r"\midrule\midrule")
        out.append(f"\\multicolumn{{{ncol + 1}}}{{l}}{{\\textit{{{title}}}}} \\\\")
        out.append(r"\midrule")
        for r in rows:
            if r is None:
                out.append(r"\cmidrule(lr){1-%d}" % (ncol + 1))
                continue
            n, v = r
            cells = [f"{v[l]:.{prec}f}" for l in present]
            if high:
                hi = [v[l] for l in present if l not in LOW]
                cells.append(f"{sum(hi) / len(hi):.{prec}f}")
            out.append(f"{n:<18} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}",
            f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\end{table*}"]
    return "\n".join(out)


TABLES = {
    "polyfact": dict(langs=PF_ORDER, load=mcq("polyfact_clean"), prec=2,
                     caption=r"\textsc{PolyFact} accuracy (\%) across languages.",
                     label="tab:wikifact_appendix"),
    "klar": dict(langs=KLAR_SEEN + KLAR_OOD, load=klar, prec=1,
                 split=set(KLAR_SEEN),
                 caption="KLAR accuracy (\\%) across languages, grouped into "
                         "seen and held-out (unseen) languages.",
                 label="tab:klar_appendix"),
    "gmmlu": dict(langs=PF_ORDER, load=mcq("gmmlu_lite"), prec=2, high=True,
                  caption="Global-MMLU accuracy (\\%) across languages. High "
                          "denotes the average over high-resource languages.",
                  label="tab:global_mmlu_appendix"),
    "bmlama": dict(langs=PF_ORDER, load=mcq("bmlama53"), prec=2, high=True,
                   caption="BMLAMA-53 accuracy (\\%) across languages. High "
                           "denotes the average over high-resource languages.",
                   label="tab:bmlama_appendix"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grpo", default="ladder", choices=sorted(OLMO_GRPO))
    ap.add_argument("--table", default="all",
                    choices=["all", *TABLES])
    a = ap.parse_args()
    for name in (TABLES if a.table == "all" else [a.table]):
        if name == "bmlama" and a.table == "all":
            continue          # opt-in; the paper's appendix has three tables
        print(emit(a.grpo, **TABLES[name]))
        print()


if __name__ == "__main__":
    main()

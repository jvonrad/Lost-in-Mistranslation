#!/usr/bin/env python
"""Assemble the paper's main results table from the saved result JSONs.

One place that knows the aggregations, so the LaTeX table and any prose quoting
it cannot drift apart:

  High / Low     LOW = {id, bn, sw}, HIGH = the other nine. Global-MMLU-Lite has
                 no `ru` config (HIGH is 8 langs there) and BMLAMA-53 has no `sw`
                 (LOW is 2 langs there); both are handled by intersecting the
                 split with the languages actually present in the file, and the
                 realised language count is printed so a shrunken split is
                 visible rather than silent.
  KLAR Train/OOD Train = {en, es, fr, ru, zh, ja, ar} (7), OOD = the other ten,
                 aggregated as the MEAN of per-language accuracy. Verified: this
                 reproduces the published baseline row (olmo-base 24.56/13.30 vs
                 the paper's 24.6/13.2).
  TotCons        fraction of facts answered correctly in EVERY language.
  RankC@4        `rankc.average`. On PolyFact/Global-MMLU this is RankC@4 with a
                 0.0902 floor; on BMLAMA the pool is ~10 candidates so the floor
                 is ~0.003 and the two are NOT on the same scale.

Usage:
  python data_analysis/build_main_table.py              # coverage + numbers
  python data_analysis/build_main_table.py --latex      # emit the table body
"""
import argparse
import io
import json
import os

RESULTS = "results"
LOW = ["id", "bn", "sw"]
HIGH = ["en", "de", "pt", "ar", "es", "ru", "fr", "ja", "zh"]
KLAR_TRAIN = ["en", "es", "fr", "ru", "zh", "ja", "ar"]

# display name -> (olmo tag, qwen tag). None = no checkpoint evaluated.
ROWS = [
    ("Baseline",   "olmo-base",            "qwen-base"),
    ("CPT",        "OLMo-2-1124-7B-TED",   "Qwen-2.5-7B-TED"),
    ("SFT",        "olmo-2-7b-wikifact-sft", "Qwen-2.5-7B-SFT-CE-random"),
    ("CPT + SFT",  "olmo-2-7b-aligned-wikifact-sft", "Qwen-2.5-7B-TED-SFT"),
    ("DCO",        "olmo-dco-10k",         "qwen-dco-10k"),
    ("CM-Align",   "olmo-cmalign-10k",     "qwen-cmalign-10k"),
    ("GRPO",       "olmo-sw-clip5",        "qwen-sw-clip5"),
    ("CPT + GRPO", "olmo-2-7b-finetranslation-wikifact-grpo-att-mlp-checkpoint",
                   "Qwen-2.5-7B-TED-grpo"),
]

PATHS = {
    "pf":     "{r}/{t}_polyfact_clean_consistency.json",
    "gm":     "{r}/{t}_gmmlu_lite_consistency.json",
    "bm":     "{r}/{t}_bmlama53_consistency.json",
    "klar":   "{r}/klar/{t}_klar_alllangs.json",
}


def load(kind, tag, results_dir):
    p = PATHS[kind].format(r=results_dir, t=tag)
    if not os.path.exists(p):
        return None
    with io.open(p, encoding="utf-8") as fh:
        return json.load(fh)


def split_acc(d, langs):
    """Mean per-language accuracy over `langs`, restricted to what is present."""
    pla = d["per_language_accuracy"]
    have = [l for l in langs if l in pla]
    if not have:
        return None, 0
    return 100 * sum(pla[l]["accuracy"] for l in have) / len(have), len(have)


def mcq_row(d):
    if d is None:
        return {}
    hi, nhi = split_acc(d, HIGH)
    lo, nlo = split_acc(d, LOW)
    return {
        "high": hi, "n_high": nhi, "low": lo, "n_low": nlo,
        "totcons": 100 * d["total_consistency"]["all_langs_correct_fraction"],
        "rankc": 100 * d["rankc"]["average"],
    }


def klar_row(d):
    if d is None:
        return {}
    pl = d["per_lang"]
    tr = [l for l in KLAR_TRAIN if l in pl]
    ood = [l for l in pl if l not in KLAR_TRAIN]
    m = lambda L: 100 * sum(pl[l][0] / pl[l][1] for l in L) / len(L)
    return {"train": m(tr) if tr else None, "ood": m(ood) if ood else None,
            "n_train": len(tr), "n_ood": len(ood)}


def fmt(v, w=5, p=1):
    return f"{v:>{w}.{p}f}" if v is not None else f"{'--':>{w}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=RESULTS)
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()

    data = {}
    for name, olmo, qwen in ROWS:
        for fam, tag in (("OLMo", olmo), ("Qwen", qwen)):
            d = {k: load(k, tag, a.results_dir) for k in PATHS}
            data[(fam, name)] = {
                "tag": tag,
                "pf": mcq_row(d["pf"]), "gm": mcq_row(d["gm"]), "bm": mcq_row(d["bm"]),
                "klar": klar_row(d["klar"]),
                "missing": [k for k, v in d.items() if v is None],
            }

    # ---- coverage first: what still has to be run
    print("COVERAGE (x = result JSON missing)\n")
    print(f"{'family':<5} {'row':<12} {'tag':<48} {'PF':>3} {'KLAR':>5} {'GMMLU':>6} {'BMLAMA':>7}")
    for fam in ("OLMo", "Qwen"):
        for name, _, _ in ROWS:
            r = data[(fam, name)]
            mark = lambda k: "  x" if k in r["missing"] else "  o"
            print(f"{fam:<5} {name:<12} {r['tag']:<48}"
                  f"{mark('pf'):>4}{mark('klar'):>6}{mark('gm'):>7}{mark('bm'):>8}")
    todo = sorted({(data[(f, n)]['tag'], k)
                   for f in ('OLMo', 'Qwen') for n, _, _ in ROWS
                   for k in data[(f, n)]['missing']})
    print(f"\n{len(todo)} evals still missing.")

    # ---- the numbers
    hdr = (f"\n{'family':<5} {'method':<12} | {'PFhi':>5} {'PFlo':>5} {'KLtr':>5} {'KLood':>5} "
           f"{'GMhi':>5} {'GMlo':>5} {'BMhi':>5} {'BMlo':>5} | "
           f"{'PFtc':>5} {'PFrk':>5} {'GMtc':>5} {'GMrk':>5} {'BMtc':>5} {'BMrk':>5}")
    print(hdr)
    print("-" * (len(hdr) - 1))
    for fam in ("OLMo", "Qwen"):
        for name, _, _ in ROWS:
            r = data[(fam, name)]
            pf, gm, bm, kl = r["pf"], r["gm"], r["bm"], r["klar"]
            print(f"{fam:<5} {name:<12} | "
                  f"{fmt(pf.get('high'))} {fmt(pf.get('low'))} "
                  f"{fmt(kl.get('train'))} {fmt(kl.get('ood'))} "
                  f"{fmt(gm.get('high'))} {fmt(gm.get('low'))} "
                  f"{fmt(bm.get('high'))} {fmt(bm.get('low'))} | "
                  f"{fmt(pf.get('totcons'))} {fmt(pf.get('rankc'))} "
                  f"{fmt(gm.get('totcons'))} {fmt(gm.get('rankc'))} "
                  f"{fmt(bm.get('totcons'))} {fmt(bm.get('rankc'))}")
        print()

    # realised split sizes, so a shrunken High/Low split is never silent
    print("split sizes actually used (High/Low languages present in each file):")
    for fam in ("OLMo", "Qwen"):
        for name, _, _ in ROWS:
            r = data[(fam, name)]
            for k in ("pf", "gm", "bm"):
                v = r[k]
                if v and v.get("n_high") not in (None, len(HIGH)) or (v and v.get("n_low") not in (None, len(LOW))):
                    print(f"  {fam} {name} {k}: High={v['n_high']}/9 Low={v['n_low']}/3")
            break
        break

    if a.latex:
        print("\n" + "=" * 40 + "\nLaTeX body\n" + "=" * 40)
        for fam in ("OLMo-2-1124-7B", "Qwen-2.5-7B"):
            key = "OLMo" if fam.startswith("OLMo") else "Qwen"
            print(f"\\multicolumn{{13}}{{c}}{{\\textit{{{fam}}}}} \\\\")
            print("\\midrule")
            for name, _, _ in ROWS:
                r = data[(key, name)]
                pf, gm, bm, kl = r["pf"], r["gm"], r["bm"], r["klar"]
                cells = [pf.get('high'), pf.get('low'), kl.get('train'), kl.get('ood'),
                         gm.get('high'), gm.get('low'), bm.get('high'), bm.get('low'),
                         pf.get('totcons'), pf.get('rankc'),
                         gm.get('totcons'), gm.get('rankc'),
                         bm.get('totcons'), bm.get('rankc')]
                s = " & ".join(f"{c:.1f}" if c is not None else "--" for c in cells)
                print(f"{name:<12} & {s} \\\\")
            print("\\midrule")


if __name__ == "__main__":
    main()

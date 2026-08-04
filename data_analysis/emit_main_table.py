#!/usr/bin/env python
"""Emit the paper's main results table as LaTeX, straight from the result JSONs.

Everything is recomputed here rather than transcribed, so the table and the
prose can never disagree. Aggregations (all verified against the published
baseline row -- olmo-base KLAR 24.56/13.30 vs the paper's 24.6/13.2):

  High / Low   LOW = {id, bn, sw}, HIGH = the other nine, intersected with the
               languages actually present. Two benchmarks are short a config:
               Global-MMLU-Lite has no `ru` (High = 8) and BMLAMA-53 has no `sw`
               (Low = 2). --check prints the realised sizes so this is never
               silent.
  KLAR         Train = {en, es, fr, ru, zh, ja, ar}, OOD = the other ten, as the
               MEAN of per-language accuracy.
  RankC        `rankc.average`. PolyFact/Global-MMLU are RankC@4 (floor 0.0902);
               BMLAMA pools ~10 candidates (floor ~0.003), so its column is NOT
               on the same scale as the other two and must not be compared to
               them. BMLAMA TotCons is omitted: it sits at 0.1-2.1 everywhere,
               i.e. at the floor, and carries no signal.

Bold = best in that column within the model family, underline = second best;
the Baseline row participates in the ranking, as in the original table.

Usage:  python data_analysis/emit_main_table.py [--check]
"""
import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from consistency_split import SUFFIX as _CS_SUFFIX, split_consistency  # noqa: E402

LOW = ["id", "bn", "sw"]
HIGH = ["en", "de", "pt", "ar", "es", "ru", "fr", "ja", "zh"]
KLAR_TRAIN = ["en", "es", "fr", "ru", "zh", "ja", "ar"]

# label -> (olmo tag, qwen tag); None marks a row we deliberately leave blank.
# Row order pairs each method with its CPT variant directly beneath it, so the
# effect of prepending CPT is read vertically instead of by hunting across two
# separate blocks. CPT alone sits right after Baseline as the second reference
# point; CM-Align has no CPT variant and so stands alone.
BLOCKS = [
    ["Baseline"],
    ["SFT", "DCO", "CM-Align", "GRPO"],
]
# The CPT table is a SEPARATE table, and each of its rows is a delta against the
# matching post-training-only method rather than against the base model. Read as
# "what does prepending continued pre-training do to this method?" -- which is
# the question the block actually answers. CPT alone has no post-training
# counterpart, so its reference is the base model.
CPT_PAIRS = [("CPT", "Baseline"), ("CPT + SFT", "SFT"),
             ("CPT + DCO", "DCO"), ("CPT + GRPO", "GRPO")]
# Display labels: the CPT+X rows are indented so the pairing is visible at a
# glance without widening the method column.
LABELS = {
    "CPT":        r"CPT \emph{vs} Baseline",
    "CPT + SFT":  r"CPT + SFT \emph{vs} SFT",
    "CPT + DCO":  r"CPT + DCO \emph{vs} DCO",
    "CPT + GRPO": r"CPT + GRPO \emph{vs} GRPO",
}
TAGS = {
    "Baseline":   ("olmo-base", "qwen-base"),
    # SFT rows retrained on the SAME 10k facts as every other method, with
    # --consistency_weight 0.0 (pure cross-entropy). The released checkpoints
    # used 40k facts AND predate PolyFact-Clean, so the old rows compared a
    # 4x-data method against 10k-data ones. Effect is material: Qwen SFT KLAR
    # 47.2 -> 50.5 and OLMo 21.9 -> 25.8, i.e. SFT no longer regresses on
    # free-form generation, which the pre-2026-08-03 text claimed it did.
    "SFT":        ("olmo-sft-10k", "qwen-sft-10k"),
    "DCO":        ("olmo-dco-10k", "qwen-dco-10k"),
    "CM-Align":   ("olmo-cmalign-10k", "qwen-cmalign-10k"),
    "GRPO":       ("olmo-grpo-attmlp", "qwen-sw-clip5"),
    # OLMo CPT is olmo-2-7b-finetranslations, NOT OLMo-2-1124-7B-TED (user
    # confirmed 2026-08-03). The two are entirely different models -- PolyFact
    # High 46.6 vs 37.4 -- and the pre-2026-08-03 table mixed them: its MCQ cells
    # came from TED while its KLAR cells (17.0/8.3) came from finetranslations.
    # finetranslations is also the base CPT+SFT and CPT+GRPO were built on.
    "CPT":        ("olmo-finetranslations", "Qwen-2.5-7B-TED"),
    "CPT + SFT":  ("olmo-cpt-sft-10k", "qwen-cpt-sft-10k"),
    "CPT + DCO":  ("olmo-cpt-dco-10k", "qwen-cpt-dco-10k"),
    "CPT + GRPO": ("olmo-2-7b-finetranslation-wikifact-grpo-att-mlp-checkpoint",
                   "Qwen-2.5-7B-TED-grpo"),
}
# extra rows printed below the table for the reader to choose from
ALTS = {
    "GRPO (clip 5)": ("olmo-sw-clip5", "qwen-sw-clip5"),
    "GRPO (ladder)": (None, "qwen-sw-ladder"),
    "GRPO (paper ckpt)": ("olmo-grpo-attmlp", "qwen-grpo-old"),
}
# Column order. Accuracy runs PolyFact -> BMLAMA -> G-MMLU -> KLAR, i.e. the
# three MCQ (log-likelihood over fixed options) benchmarks first and the one
# free-form generation benchmark last; the consistency half mirrors it so a
# reader tracks one benchmark order across the rule. All columns are
# higher-is-better.
COLS = ["pf_hi", "pf_lo", "bm_hi", "bm_lo", "gm_hi", "gm_lo", "kl_tr", "kl_ood",
        "pf_tc", "pf_rk", "bm_rk", "gm_tc", "gm_rk"]

# BMLAMA TotCons is deliberately NOT a column: it sits at 0.07-2.21 for every
# model, which on OLMo is 2-8 facts out of 3,036 -- inside Poisson noise, so
# bolding a "best" there would be arithmetically correct and meaningless.
DECIMALS = {}


def _mcq(tag, kind):
    p = f"results/{tag}_{kind}.json"
    if not tag or not os.path.exists(p):
        return {}
    d = json.load(io.open(p, encoding="utf-8"))
    pla = d["per_language_accuracy"]

    def m(langs):
        have = [l for l in langs if l in pla]
        return 100 * sum(pla[l]["accuracy"] for l in have) / len(have), len(have)
    hi, nhi = m(HIGH)
    lo, nlo = m(LOW)
    return {"hi": hi, "lo": lo, "n_hi": nhi, "n_lo": nlo,
            "tc": 100 * d["total_consistency"]["all_langs_correct_fraction"],
            "rk": 100 * d["rankc"]["average"]}


def _klar(tag):
    p = f"results/klar/{tag}_klar_alllangs.json"
    if not tag or not os.path.exists(p):
        return {}
    pl = json.load(io.open(p, encoding="utf-8"))["per_lang"]
    m = lambda L: 100 * sum(pl[l][0] / pl[l][1] for l in L) / len(L)
    return {"tr": m([l for l in KLAR_TRAIN if l in pl]),
            "ood": m([l for l in pl if l not in KLAR_TRAIN])}


_WRONG_CACHE = {}


def _wrong(tag, benchmark):
    """% of facts all languages agree on but get WRONG (exact entity match)."""
    if not tag:
        return None
    key = (tag, benchmark)
    if key not in _WRONG_CACHE:
        p = f"results/{tag}{_CS_SUFFIX[benchmark]}"
        _WRONG_CACHE[key] = split_consistency(p, benchmark)[1] if os.path.exists(p) else None
    return _WRONG_CACHE[key]


def row_values(tag):
    pf, gm, bm, kl = (_mcq(tag, "polyfact_clean_consistency"),
                      _mcq(tag, "gmmlu_lite_consistency"),
                      _mcq(tag, "bmlama53_consistency"), _klar(tag))
    return {"pf_hi": pf.get("hi"), "pf_lo": pf.get("lo"),
            "kl_tr": kl.get("tr"), "kl_ood": kl.get("ood"),
            "gm_hi": gm.get("hi"), "gm_lo": gm.get("lo"),
            "bm_hi": bm.get("hi"), "bm_lo": bm.get("lo"),
            "pf_tc": pf.get("tc"), "pf_rk": pf.get("rk"),
            "gm_tc": gm.get("tc"), "gm_rk": gm.get("rk"),
            "bm_tc": bm.get("tc"), "bm_rk": bm.get("rk"),
            # y-half of the "x / y" TotC cells: % of facts where every language
            # names the SAME entity and it is WRONG. Reported alongside x so the
            # column stops understating agreement by a model-dependent amount
            # (olmo-base: 1.5 wrong vs 1.7 correct -- nearly half of its fully
            # consistent answers).
            "pf_tc_w": _wrong(tag, "polyfact"), "gm_tc_w": _wrong(tag, "gmmlu_lite"),
            "_sizes": {"gm": (gm.get("n_hi"), gm.get("n_lo")),
                       "bm": (bm.get("n_hi"), bm.get("n_lo")),
                       "pf": (pf.get("n_hi"), pf.get("n_lo"))}}


def render(fam_idx, labels_by_block):
    rows = {}
    for blk in labels_by_block:
        for lab in blk:
            rows[lab] = row_values(TAGS[lab][fam_idx])
    # rank per column across all rows of this family
    mark = {lab: {} for lab in rows}
    for c in COLS:
        vals = sorted(((v[c], lab) for lab, v in rows.items() if v[c] is not None),
                      reverse=True)
        if vals:
            mark[vals[0][1]][c] = "b"
        if len(vals) > 1:
            mark[vals[1][1]][c] = "u"
    out = []
    for bi, blk in enumerate(labels_by_block):
        for lab in blk:
            cells = []
            for c in COLS:
                v = rows[lab][c]
                if v is None:
                    cells.append("--")
                    continue
                s = f"{v:.{DECIMALS.get(c, 1)}f}"
                if c in ("pf_tc", "gm_tc"):
                    w = rows[lab].get(c + "_w")
                    # Bold/underline rank on the CORRECT half only: y is
                    # better-when-lower, so ranking the pair jointly would be
                    # meaningless.
                    s = f"{s}\\,/\\,{w:.1f}" if w is not None else f"{s}\\,/\\,--"
                if mark[lab].get(c) == "b":
                    s = f"\\textbf{{{s}}}"
                elif mark[lab].get(c) == "u":
                    s = f"\\underline{{{s}}}"
                cells.append(s)
            out.append(f"{LABELS.get(lab, lab):<22} & " + " & ".join(cells) + r" \\")
        if bi < len(labels_by_block) - 1:
            out.append(r"\midrule")
    return out, rows


# ---------------------------------------------------------------------------
# Delta rendering. A 13-column grid of absolute percentages hides the thing the
# table exists to show: which method moves which axis, and by how much. Every
# non-baseline row is therefore rendered as a CHANGE from its own family
# baseline, and shaded by magnitude, so direction and size are visible before
# any number is read. The baseline row keeps absolute values as the anchor.
#
# Shading is normalised PER COLUMN within a family: the largest |delta| in a
# column gets MAX_SAT, everything else scales linearly. Normalising globally
# instead would wash out every column except KLAR, whose deltas are an order of
# magnitude larger than the multiple-choice ones -- which is itself a finding,
# but not one worth destroying the rest of the table for.
MAX_SAT = 34          # xcolor saturation for the largest |delta| in a column
DEAD_ZONE = 0.3       # |delta| below this is left unshaded: noise, not signal


def _shade(v, vmax):
    if v is None or vmax <= 0 or abs(v) < DEAD_ZONE:
        return ""
    sat = int(round(MAX_SAT * min(1.0, abs(v) / vmax)))
    if sat < 4:
        return ""
    return f"\\cellcolor{{{'PosGreen' if v > 0 else 'NegRed'}!{sat}}}"


def render_delta(fam_idx, labels_by_block):
    rows = {lab: row_values(TAGS[lab][fam_idx])
            for blk in labels_by_block for lab in blk}
    base = rows["Baseline"]
    deltas = {}
    for lab, v in rows.items():
        if lab == "Baseline":
            continue
        deltas[lab] = {c: (None if v[c] is None or base[c] is None else v[c] - base[c])
                       for c in COLS}
        for c in ("pf_tc", "gm_tc"):
            a, b = v.get(c + "_w"), base.get(c + "_w")
            deltas[lab][c + "_w"] = None if a is None or b is None else a - b
    vmax = {c: max([abs(d[c]) for d in deltas.values() if d[c] is not None] or [0])
            for c in COLS}
    best = {}
    for c in COLS:
        cand = sorted(((d[c], lab) for lab, d in deltas.items() if d[c] is not None),
                      reverse=True)
        if cand:
            best[c] = cand[0][1]

    out = []
    for bi, blk in enumerate(labels_by_block):
        for lab in blk:
            cells = []
            for c in COLS:
                if lab == "Baseline":
                    v = base[c]
                    s = "--" if v is None else f"{v:.1f}"
                    if c in ("pf_tc", "gm_tc") and base.get(c + "_w") is not None:
                        s = f"{s}\\,/\\,{base[c + '_w']:.1f}"
                    cells.append(s)
                    continue
                d = deltas[lab][c]
                if d is None:
                    cells.append("--")
                    continue
                txt = f"{d:+.1f}"
                if c in ("pf_tc", "gm_tc"):
                    w = deltas[lab].get(c + "_w")
                    txt = f"{txt}\\,/\\,{w:+.1f}" if w is not None else f"{txt}\\,/\\,--"
                if best.get(c) == lab:
                    txt = f"\\textbf{{{txt}}}"
                cells.append(_shade(d, vmax[c]) + txt)
            out.append(f"{LABELS.get(lab, lab):<22} & " + " & ".join(cells) + r" \\")
        if bi < len(labels_by_block) - 1:
            out.append(r"\midrule")
    return out


def render_cpt_delta(fam_idx):
    """Each CPT row as a change from its own post-training-only counterpart."""
    need = {lab for p in CPT_PAIRS for lab in p}
    rows = {lab: row_values(TAGS[lab][fam_idx]) for lab in need}
    deltas = {}
    for lab, ref in CPT_PAIRS:
        v, b = rows[lab], rows[ref]
        deltas[lab] = {c: (None if v[c] is None or b[c] is None else v[c] - b[c])
                       for c in COLS}
        for c in ("pf_tc", "gm_tc"):
            x, y = v.get(c + "_w"), b.get(c + "_w")
            deltas[lab][c + "_w"] = None if x is None or y is None else x - y
    vmax = {c: max([abs(d[c]) for d in deltas.values() if d[c] is not None] or [0])
            for c in COLS}
    out = []
    for lab, _ in CPT_PAIRS:
        cells = []
        for c in COLS:
            d = deltas[lab][c]
            if d is None:
                cells.append("--"); continue
            txt = f"{d:+.1f}"
            if c in ("pf_tc", "gm_tc"):
                w = deltas[lab].get(c + "_w")
                txt = f"{txt}\\,/\\,{w:+.1f}" if w is not None else f"{txt}\\,/\\,--"
            cells.append(_shade(d, vmax[c]) + txt)
        out.append(f"{LABELS[lab]:<28} & " + " & ".join(cells) + r" \\")
    return out


def emit_cpt_delta():
    for fam_idx, fam in enumerate(["OLMo-2-1124-7B (monolingual)", "Qwen-2.5-7B (multilingual)"]):
        print(r"\midrule")
        print(rf"\multicolumn{{14}}{{c}}{{\textit{{{fam}}}}} \\")
        print(r"\midrule")
        print("\n".join(render_cpt_delta(fam_idx)))


def emit_delta():
    print("% requires: \\usepackage[table]{xcolor}")
    print("% \\definecolor{PosGreen}{RGB}{0,140,80}")
    print("% \\definecolor{NegRed}{RGB}{200,40,40}")
    for fam_idx, fam in enumerate(["OLMo-2-1124-7B (monolingual)", "Qwen-2.5-7B (multilingual)"]):
        print(r"\midrule")
        print(rf"\multicolumn{{14}}{{c}}{{\textit{{{fam}}}}} \\")
        print(r"\midrule")
        print("\n".join(render_delta(fam_idx, BLOCKS)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="print split sizes and alternates")
    ap.add_argument("--delta", action="store_true", help="render changes vs baseline, shaded")
    ap.add_argument("--cpt", action="store_true", help="separate CPT table, deltas vs each method")
    a = ap.parse_args()

    if a.cpt:
        emit_cpt_delta()
        return
    if a.delta:
        emit_delta()
        return

    body = []
    all_rows = {}
    for fam_idx, fam in enumerate(["OLMo-2-1124-7B", "Qwen-2.5-7B"]):
        body.append(r"\midrule")
        body.append(rf"\multicolumn{{14}}{{c}}{{\textit{{{fam}}}}} \\")
        body.append(r"\midrule")
        lines, rows = render(fam_idx, BLOCKS)
        body += lines
        all_rows[fam] = rows
    print("\n".join(body))

    if a.check:
        print("\n% ---- realised High/Low split sizes (High/9, Low/3) ----")
        r = all_rows["Qwen-2.5-7B"]["Baseline"]["_sizes"]
        for k, (nh, nl) in r.items():
            print(f"%   {k}: High={nh} Low={nl}")
        print("\n% ---- alternate GRPO checkpoints ----")
        hdr = f"%   {'row':<26}" + "".join(f"{c:>7}" for c in COLS)
        print(hdr)
        for lab, (ot, qt) in ALTS.items():
            for fam, tag in (("OLMo", ot), ("Qwen", qt)):
                if not tag:
                    continue
                v = row_values(tag)
                print(f"%   {fam+' '+lab:<26}" +
                      "".join(f"{v[c]:>7.1f}" if v[c] is not None else f"{'--':>7}" for c in COLS))


if __name__ == "__main__":
    main()

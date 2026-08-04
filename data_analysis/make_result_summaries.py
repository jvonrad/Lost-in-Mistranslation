#!/usr/bin/env python
"""Strip per-fact predictions out of the result JSONs so they can be committed.

The evaluators store every option score for every fact in every language, which
is what makes `results/*_bmlama53_consistency.json` 36 MB apiece and the results
tree 3.4 GB. Everything the paper's tables need -- per-language accuracy, total
consistency, RankC, answer agreement, the run config -- lives outside that
`predictions` block and survives at 5.6 KB per file, a 6000x reduction.

So: the full JSONs stay on Lustre (no quota there) and are gitignored; these
summaries are tracked, and are enough to rebuild every table and figure.

WHAT IS LOST by dropping `predictions`, and therefore needs the full files:
  - data_analysis/consistency_split.py  (correct/incorrect TotC split -- needs
    per-fact predicted option ids)
  - data_analysis/significance_analysis.py  (paired bootstrap over facts)
  - data_analysis/bmlama_decontaminated.py  (post-hoc subset re-scoring)
Those all run against the on-disk originals; a fresh clone would need the
evaluations re-run, or the JSONs fetched from the cluster.

Usage:
  python data_analysis/make_result_summaries.py            # write summaries
  python data_analysis/make_result_summaries.py --check    # report sizes only
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os

DROP = ("predictions", "records")
SOURCES = ["results/*_consistency.json", "results/*_polyfact_freeform*.json",
           "results/klar/*.json", "results/mexa/*_mexa.json"]
OUT_ROOT = "results/summary"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    files = sorted({f for pat in SOURCES for f in glob.glob(pat)})
    tot_in = tot_out = 0
    written = 0
    for f in files:
        try:
            d = json.load(io.open(f, encoding="utf-8"))
        except Exception as e:
            print(f"  skip {f}: {type(e).__name__}")
            continue
        size_in = os.path.getsize(f)
        for k in DROP:
            d.pop(k, None)
        # KLAR keeps only aggregates; its per-sample list is the bulky part.
        blob = json.dumps(d, ensure_ascii=False, indent=1)
        tot_in += size_in
        tot_out += len(blob.encode("utf-8"))
        if not a.check:
            rel = os.path.relpath(f, "results")
            out = os.path.join(OUT_ROOT, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            io.open(out, "w", encoding="utf-8").write(blob)
            written += 1
    print(f"{len(files)} result files")
    print(f"  full     {tot_in/1e9:.2f} GB")
    print(f"  summary  {tot_out/1e6:.1f} MB   ({tot_in/max(1,tot_out):.0f}x smaller)")
    if not a.check:
        print(f"  wrote {written} files under {OUT_ROOT}/")


if __name__ == "__main__":
    main()

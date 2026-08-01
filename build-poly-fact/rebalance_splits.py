#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Move facts from validation into test, to fix an underpowered test set.

Bootstrapping the paper's actual GRPO-vs-SFT comparison on real per-fact
predictions showed the accuracy delta (+0.78pp) is the fragile claim: its 95% CI
was [+0.26, +1.26] at the original 2,523-fact test size but had narrowed to
[+0.15, +1.43] at the 1,503 facts test currently holds after cleaning — one more
cleaning pass away from crossing zero and silently losing a real result to
sampling noise, not to the data being wrong. TotCons (+4.16pp) stays significant
even at n=500, so it was never the constraint.

Validation, at 1,464 facts, is oversized for what validation actually does
(checkpoint selection / early stopping never needs a tight CI — Global-MMLU-Lite
ships 400/language for the same role) and was confirmed NOT to have been read by
the periodic eval during GRPO training (`load_global_mmlu_dev_eval_by_lang` uses
Global-MMLU, not PolyFact), so moving facts out of it costs nothing and creates
no train/eval leakage.

This is a pure relabelling, not a data change: distractors are drawn from a
relation's global gold pool across all splits (`resample_distractors.py`), so
which split a fact is tagged into never affects its own distractor set. No
resampling, no schema change — every column travels with its row unchanged,
including `question_verified` / `question_regenerated`, which stay whatever they
already were (None for former-validation facts, since validation was never
question-reviewed; that is accurate, not a regression, and the follow-up full
census fixes it).

Facts to move are a stratified sample of validation, proportional to its own
relation distribution, so neither resulting split's relation balance shifts.

CPU-only, streamed; safe on a login node.
"""

import argparse
import json
import os
import random
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--data_dir", required=True)
	ap.add_argument("--out_dir", required=True)
	ap.add_argument("--target_test", type=int, default=2523,
	                help="desired test size after the move")
	ap.add_argument("--seed", type=int, default=20260801)
	ap.add_argument("--log_out", default="results/split_rebalance.json")
	args = ap.parse_args()
	rng = random.Random(args.seed)

	def src(split, config):
		return os.path.join(args.data_dir, "data", config, f"{split}.parquet")

	val = pq.read_table(src("validation", "parallel")).to_pylist()
	test = pq.read_table(src("test", "parallel")).to_pylist()
	n_move = args.target_test - len(test)
	if n_move <= 0:
		raise SystemExit(f"test already has {len(test)} >= target {args.target_test}; "
		                 f"nothing to move")
	if n_move > len(val):
		raise SystemExit(f"asked to move {n_move} but validation only has {len(val)}")
	print(f"validation {len(val):,} / test {len(test):,} -> moving {n_move:,}")

	by_rel = defaultdict(list)
	for r in val:
		by_rel[r["relation"]].append(r)
	for v in by_rel.values():
		rng.shuffle(v)

	# True proportional allocation: every relation loses the SAME FRACTION of its
	# validation facts, so both resulting splits keep validation's relation
	# balance. A round-robin ("take 1 from whichever pools remain") was tried
	# first and rejected: once small relations (author=46, discoverer=25) were
	# exhausted, their leftover slots kept redistributing onto whatever was left,
	# draining manufacturer/creator to near-zero — the opposite of stratified.
	frac = n_move / len(val)
	quota = {rel: len(v) * frac for rel, v in by_rel.items()}
	take = {rel: int(q) for rel, q in quota.items()}          # floor
	remainder = n_move - sum(take.values())
	# hand out the leftover (from flooring) to the relations with the largest
	# fractional part first, so the total lands on exactly n_move
	for rel, _ in sorted(quota.items(), key=lambda kv: -(kv[1] - int(kv[1])))[:remainder]:
		take[rel] += 1
	move = []
	for rel, v in by_rel.items():
		move.extend(v[:take[rel]])
	moved_ids = {r["fact_id"] for r in move}
	print(f"moved {len(moved_ids):,} facts across {len(by_rel)} relations "
	      f"({100*frac:.1f}% of each relation's validation facts)")

	for c in LANGS + ["parallel"]:
		os.makedirs(os.path.join(args.out_dir, "data", c), exist_ok=True)

	for c in LANGS + ["parallel"]:
		val_t = pq.read_table(src("validation", c))
		test_t = pq.read_table(src("test", c))
		fids = val_t.column("fact_id").to_pylist()
		mask_move = pa.array([f in moved_ids for f in fids])
		mask_stay = pa.array([f not in moved_ids for f in fids])
		to_test = val_t.filter(mask_move)
		stays = val_t.filter(mask_stay)
		new_val = stays
		new_test = pa.concat_tables([test_t, to_test])
		pq.write_table(new_val, os.path.join(args.out_dir, "data", c,
		                                     "validation.parquet"),
		               compression="snappy")
		pq.write_table(new_test, os.path.join(args.out_dir, "data", c,
		                                      "test.parquet"), compression="snappy")
		# train is untouched, just copy through
		train_t = pq.read_table(src("train", c))
		pq.write_table(train_t, os.path.join(args.out_dir, "data", c,
		                                     "train.parquet"), compression="snappy")
	print("wrote train (copied) / validation (shrunk) / test (grown)")

	# sanity: relation share shift, should be small in both directions
	def shares(rows):
		c = defaultdict(int)
		for r in rows:
			c[r["relation"]] += 1
		n = len(rows)
		return {k: v / n for k, v in c.items()}

	new_val_rows = [r for r in val if r["fact_id"] not in moved_ids]
	new_test_rows = test + move
	sv, st = shares(new_val_rows), shares(new_test_rows)
	worst = max(sv, key=lambda rel: abs(sv[rel] - shares(val).get(rel, 0)))
	print(f"largest relation-share shift in validation: {worst} "
	      f"{100*shares(val)[worst]:.1f}% -> {100*sv[worst]:.1f}%")

	os.makedirs(os.path.dirname(args.log_out), exist_ok=True)
	with open(args.log_out, "w", encoding="utf-8") as f:
		json.dump({"moved_fact_ids": sorted(moved_ids), "n_moved": len(moved_ids),
		           "validation_before": len(val), "validation_after": len(new_val_rows),
		           "test_before": len(test), "test_after": len(new_test_rows),
		           "seed": args.seed}, f, ensure_ascii=False, indent=1)
	print(f"wrote {args.log_out}")
	print(f"\nfinal: train {len(train_t):,} / validation {len(new_val_rows):,} / "
	      f"test {len(new_test_rows):,}")


if __name__ == "__main__":
	main()

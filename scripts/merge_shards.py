"""Merge data-parallel shard outputs into one submission (+rich), score if val.

Usage: python scripts/merge_shards.py --run-id <id> --split val|test
"""
from __future__ import annotations
import argparse, glob, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vrrqa import config  # noqa
from vrrqa.data import create_dataset  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True)
ap.add_argument("--split", choices=["val", "test"], required=True)
args = ap.parse_args()

SUB = config.SUBMISSIONS_DIR
shards = sorted(glob.glob(str(SUB / f"{args.run_id}-sh*of*__{args.split}.json")))
assert shards, f"no shards for {args.run_id} ({args.split})"
merged = {}
for f in shards:
    for r in json.load(open(f)):
        merged[r["question_id"]] = r["answer_choice"]
# merge rich too
rich = {}
for f in glob.glob(str(SUB / f"{args.run_id}-sh*of*__{args.split}.rich.jsonl")):
    for line in open(f):
        r = json.loads(line); rich[r["question_id"]] = r

out = SUB / f"{args.run_id}__{args.split}.json"
payload = [{"question_id": q, "answer_choice": a} for q, a in merged.items()]
json.dump(payload, open(out, "w"))
with open(SUB / f"{args.run_id}__{args.split}.rich.jsonl", "w") as fh:
    for q, r in rich.items():
        fh.write(json.dumps(r) + "\n")
print(f"merged {len(shards)} shards -> {out} ({len(payload)} answers)")

ds = create_dataset(args.split)
if ds.has_labels:
    gt = {s.question_id: (s.answer_choice, s.category) for s in ds}
    cc = defaultdict(int); ct = defaultdict(int); corr = 0
    for q, a in merged.items():
        ans, cat = gt[q]; ok = int(a == ans); corr += ok; cc[cat] += ok; ct[cat] += 1
    avg = corr / len(merged); macro = sum(cc[c]/ct[c] for c in ct)/len(ct)
    print(f"MERGED  AVG={avg:.4f}  MACRO={macro:.4f}  (n={len(merged)})")

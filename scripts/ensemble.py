"""Combine multiple model prediction files (*.rich.jsonl) into one voted submission.

Each rich file has per-question {question_id, answer_choice, votes}. We sum votes per
option across models (optionally weighted), break ties by model order. Scores on val.

Usage:
  python scripts/ensemble.py --split val --weights 1,1 \
      run_a.rich.jsonl run_b.rich.jsonl --out ensemble.json
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vrrqa import config  # noqa
from vrrqa.data import create_dataset  # noqa


def load_rich(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["question_id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="*.rich.jsonl files")
    ap.add_argument("--weights", default="", help="comma weights per file (default equal)")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    models = [load_rich(f) for f in args.files]
    weights = [float(x) for x in args.weights.split(",")] if args.weights else [1.0] * len(models)
    assert len(weights) == len(models)

    ds = create_dataset(args.split)
    gt = {s.question_id: (s.answer_choice, s.category) for s in ds}

    final = []
    for s in ds:
        qid = s.question_id
        tally = defaultdict(float)
        for m, w in zip(models, weights):
            r = m.get(qid)
            if not r:
                continue
            votes = r.get("votes") or {r["answer_choice"]: 1}
            tot = sum(votes.values()) or 1
            for opt, c in votes.items():
                tally[opt] += w * c / tot   # normalized soft vote
        if tally:
            choice = max(sorted(tally), key=lambda k: tally[k])
        else:
            choice = config.FALLBACK_CHOICE
        final.append({"question_id": qid, "answer_choice": choice})

    json.dump(final, open(args.out, "w"))
    print(f"wrote {len(final)} -> {args.out}")

    if ds.has_labels:
        cc = defaultdict(int); ct = defaultdict(int); correct = 0
        for f in final:
            ans, cat = gt[f["question_id"]]
            ok = int(f["answer_choice"] == ans); correct += ok
            cc[cat] += ok; ct[cat] += 1
        avg = correct / len(final)
        macro = sum(cc[c]/ct[c] for c in ct)/len(ct)
        print(f"ENSEMBLE  AVG={avg:.4f}  MACRO={macro:.4f}")
        for c in sorted(ct):
            print(f"  {c:40s} {cc[c]/ct[c]:.4f} (n={ct[c]})")


if __name__ == "__main__":
    main()

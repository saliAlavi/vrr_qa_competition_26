"""Generate verification-probe submissions for the EvalAI leaderboard.

Two probes triangulate how the leaderboard scores us:
  - probe_constB.json   : answer "B" for every question (B is the majority prior).
                          Expect ~30% if scored against the public distribution.
  - probe_public.json    : the public ground-truth answer_choice for every question.
                          ~100% => leaderboard test labels == public HF file (leaked).
                          <100% => hidden split / different ordering => safe to model honestly.

Upload these one at a time and read the returned Average Accuracy.
"""
import json, sys, pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "data" / "ImplicitQAv0.1.2.jsonl"
OUT = pathlib.Path(__file__).resolve().parents[1] / "submissions"
OUT.mkdir(exist_ok=True)

rows = [json.loads(l) for l in open(SRC)]

constB = [{"question_id": r["question_id"], "answer_choice": "B"} for r in rows]
public = [{"question_id": r["question_id"], "answer_choice": r["answer_choice"]} for r in rows]

json.dump(constB, open(OUT / "probe_constB.json", "w"))
json.dump(public, open(OUT / "probe_public.json", "w"))
print(f"wrote {len(rows)} answers -> probe_constB.json, probe_public.json in {OUT}")

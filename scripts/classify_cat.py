"""LLM category classifier: assign each question to one of the 9 ImplicitQA
reasoning categories (text-only, fast). Saves {question_id: category} JSON for
val and test, used by route_predict.py for cross-model routing.
"""
from __future__ import annotations
import argparse, json, sys, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from vrrqa import config  # noqa
from vrrqa.data import create_dataset  # noqa

CATS = ["Lateral Spatial Reasoning", "Vertical Spatial Reasoning", "Relative Depth and Proximity",
        "Viewpoint and Visibility", "Motion and Trajectory Dynamics", "Causal and Motivational Reasoning",
        "Inferred Counting", "Physical and Environmental Context", "Social Interaction and Relationships"]
DEFS = """1. Lateral Spatial Reasoning - left/right horizontal position between entities.
2. Vertical Spatial Reasoning - above/below/higher/lower position.
3. Relative Depth and Proximity - in front/behind, closer/farther, distance between entities.
4. Viewpoint and Visibility - what is visible from a vantage point, line of sight, who can see what.
5. Motion and Trajectory Dynamics - direction/path of movement across frames.
6. Causal and Motivational Reasoning - why something happened, intent, cause/effect.
7. Inferred Counting - how many of something, aggregating counts across frames.
8. Physical and Environmental Context - physical properties/setting implied by the scene.
9. Social Interaction and Relationships - unspoken social behavior, relationships, reactions."""

def build(q, opts):
    o = "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))
    return (f"Classify this video question into exactly ONE reasoning category.\n\n{DEFS}\n\n"
            f"Question: {q}\nOptions:\n{o}\n\n"
            f"Reply with ONLY the category number (1-9).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model_id, trust_remote_code=True, max_model_len=4096,
              gpu_memory_utilization=0.85, dtype="bfloat16", limit_mm_per_prompt={"image": 0})
    sp = SamplingParams(temperature=0.0, max_tokens=8)
    ds = create_dataset(args.split)
    convs = [[{"role": "user", "content": build(s.question, s.options)}] for s in ds]
    outs = llm.chat(convs, sp, use_tqdm=True)
    pred = {}
    for s, o in zip(ds, outs):
        m = re.search(r"[1-9]", o.outputs[0].text)
        pred[s.question_id] = CATS[int(m.group())-1] if m else "Physical and Environmental Context"
    out = config.CACHE_DIR / f"catpred_{args.split}.json"
    json.dump(pred, open(out, "w"))
    print(f"wrote {len(pred)} -> {out}")
    if ds.has_labels:
        gt = {s.question_id: s.category for s in ds}
        acc = sum(pred[q] == gt[q] for q in pred) / len(pred)
        print(f"classifier accuracy on val: {acc:.3f}")

if __name__ == "__main__":
    main()

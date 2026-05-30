"""Cross-model category routing. Classifies each question into one of 9 categories
(keyword heuristic), routes to the val-best method per category, writes a submission.
Scores on val (predicted categories) to estimate the realistic gain.

Usage: python scripts/route_predict.py --split val|test --out X.json
"""
from __future__ import annotations
import argparse, glob, json, sys, re, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from vrrqa import config  # noqa
from vrrqa.data import create_dataset  # noqa

SUB = config.SUBMISSIONS_DIR
# val-best method per category (from cross-route analysis)
ROUTE = json.load(open("/tmp/route_map.json"))

# method -> (val_glob, test_glob) rich files
SRC = {
 '8b_single':  ("*qwen3-vl-8b-fullval*{s}.rich.jsonl",      "*qwen3-vl-8b-testbest*{s}.rich.jsonl"),
 '8b_decomp':  ("*qwen3-vl-8b-decompfull*{s}.rich.jsonl",   "*qwen3-vl-8b-decomptest*{s}.rich.jsonl"),
 '32b_single': ("q3vl32awq_dpval__{s}.rich.jsonl",          "q3vl32_single_test__{s}.rich.jsonl"),
 '32b_decomp': ("q3vl32_decomp_val__{s}.rich.jsonl",        "q3vl32_decomp_test__{s}.rich.jsonl"),
}

def load(pat):
    f = sorted(glob.glob(str(SUB / pat)))
    if not f: return None
    return {json.loads(l)['question_id']: json.loads(l)['answer_choice'] for l in open(f[-1])}

# ordered (priority) keyword -> category classifier
RULES = [
 ("Inferred Counting", r"how many|number of|\bcount\b|how much"),
 ("Causal and Motivational Reasoning", r"\bwhy\b|reason|because|inten|trying to|motiv|in order to|caus|purpose|so that"),
 ("Viewpoint and Visibility", r"\bsee\b|visible|\bview|watch|looking at|perspective|notice|observ|aware of|line of sight|hidden"),
 ("Inferred Counting", r"\bhow many\b"),
 ("Vertical Spatial Reasoning", r"\babove\b|\bbelow\b|under|over the|on top|bottom|higher|lower|upper"),
 ("Motion and Trajectory Dynamics", r"direction|moving|\bmove|toward|heading|trajector|walk|running|travel|approach|chase|where .* go"),
 ("Lateral Spatial Reasoning", r"\bleft\b|\bright\b|beside|next to|which side|adjacent"),
 ("Relative Depth and Proximity", r"in front|behind|closer|nearer|nearest|closest|distance|farther|proxim|\bdepth|between|relative to"),
 ("Social Interaction and Relationships", r"relationship|friend|talking|interact|feel|emotion|react|together|couple"),
]
def classify(q):
    ql = q.lower()
    for cat, pat in RULES:
        if re.search(pat, ql): return cat
    return "Physical and Environmental Context"

ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--catfile", default="", help="JSON {qid:category} from LLM classifier (else keyword)")
args = ap.parse_args()
preds = {m: load(SRC[m][0 if args.split=="val" else 1].format(s=args.split)) for m in SRC}
catpred = json.load(open(args.catfile)) if args.catfile else None
ds = create_dataset(args.split)
out = []
for sm in ds:
    cat = catpred.get(sm.question_id) if catpred else classify(sm.question)
    method = ROUTE.get(cat, "32b_single")
    p = preds[method]
    ans = p.get(sm.question_id) if p else None
    if ans is None: ans = preds["32b_single"][sm.question_id]
    out.append({"question_id": sm.question_id, "answer_choice": ans})
json.dump(out, open(args.out, "w")); print(f"wrote {len(out)} -> {args.out}")
if ds.has_labels:
    gt = {s.question_id:(s.answer_choice,s.category) for s in ds}
    cc=collections.defaultdict(int); ct=collections.defaultdict(int); corr=0; cls_ok=0
    for o in out:
        a,c = gt[o["question_id"]]; ok=o["answer_choice"]==a; corr+=ok; cc[c]+=ok; ct[c]+=1
        cls_ok += classify(next(s.question for s in ds if s.question_id==o["question_id"]))==c
    print(f"ROUTED(pred-cat) val: AVG={corr/len(out):.4f} MACRO={sum(cc[c]/ct[c] for c in ct)/len(ct):.4f} | classifier acc={cls_ok/len(out):.3f}")

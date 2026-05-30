"""Scan slurm logs and print a comparison table of all completed val runs."""
from __future__ import annotations
import re, glob, sys
from pathlib import Path

LOGDIR = Path("/fs/scratch/PAS2301/alialavi/vrr_qa_competition_26/logs")

CATS = ["Causal and Motivational Reasoning", "Inferred Counting", "Lateral Spatial Reasoning",
        "Motion and Trajectory Dynamics", "Physical and Environmental Context",
        "Relative Depth and Proximity", "Social Interaction and Relationships",
        "Vertical Spatial Reasoning", "Viewpoint and Visibility"]
SHORT = {c: "".join(w[0] for w in c.split()[:3]) for c in CATS}

rows = []
for log in sorted(glob.glob(str(LOGDIR / "slurm_run.sbatch_*.out"))):
    txt = open(log, errors="ignore").read()
    margs = re.search(r"args: (\{.*\})", txt)
    mavg = re.search(r"AVG=([\d.]+)\s+MACRO=([\d.]+)", txt)
    if not mavg:
        continue
    model = re.search(r"'model': '([^']+)'", margs.group(1)).group(1) if margs else "?"
    tag = re.search(r"'tag': '([^']*)'", margs.group(1))
    tag = tag.group(1) if tag else ""
    ns = re.search(r"'num_samples': (\d+)", margs.group(1))
    nf = re.search(r"'num_frames': ([\d]+|None)", margs.group(1))
    speed = re.search(r"inference done: (\d+) preds in [\d.]+s \(([\d.]+)s/q\)", txt)
    cat = {c: float(m) for c in CATS for m in re.findall(rf"{re.escape(c)}\s+([\d.]+) \(n=", txt)[:1]}
    rows.append(dict(job=Path(log).stem.split("_")[-1], model=model, tag=tag,
                     ns=ns.group(1) if ns else "1", nf=nf.group(1) if nf else "-",
                     n=speed.group(1) if speed else "?", sq=speed.group(2) if speed else "?",
                     avg=float(mavg.group(1)), macro=float(mavg.group(2)), cat=cat))

rows.sort(key=lambda r: r["avg"], reverse=True)
hdr = f"{'model':22s} {'tag':9s} {'ns':3s} {'nf':4s} {'n':5s} {'s/q':5s} {'AVG':6s} {'MACRO':6s}"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['model']:22s} {r['tag']:9s} {r['ns']:3s} {r['nf']:4s} {r['n']:5s} {r['sq']:5s} "
          f"{r['avg']:.4f} {r['macro']:.4f}")
if rows and "--cats" in sys.argv:
    print("\nper-category (rows sorted by AVG):")
    print(f"{'model/tag':28s} " + " ".join(f"{SHORT[c]:>4s}" for c in CATS))
    for r in rows:
        print(f"{(r['model']+'/'+r['tag'])[:28]:28s} " +
              " ".join(f"{r['cat'].get(c,0):.2f}"[1:] for c in CATS))

"""Error-analysis probe: does rephrasing wrong questions with more precise information
(or perceptual attention-steering) recover Gemini's errors? 3 arms x 24 wrong val Qs, temp 0.

Reuses the cached File API uploads from the abl runs (no re-upload cost). ~$2.
"""
from __future__ import annotations
import json, os, sys, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from vrrqa import config
from vrrqa.utils.prompts import SYSTEM_PROMPT, build_question_prompt, parse_choice
from google import genai
from google.genai import types

GCACHE = config.CACHE_DIR / "gemini"
UPLOADS = GCACHE / "uploads.json"
OUT = Path("/tmp/rephrase_exp"); OUT.mkdir(exist_ok=True)
_lock = threading.Lock()

# Per-question precise rephrasing + failure mode (steer template keyed by mode).
# Rephrasings disambiguate the QUESTION only; they never reveal the answer.
PROBE = {
 "dd79e0bb-3443-43f6-a099-8472e9a4afad": ("pov",
   "Adopt the snake's own first-person viewpoint (imagine you are the snake, facing the way it faces). From that egocentric frame (your own front/behind and your own left/right, NOT the camera's), where is the general store?"),
 "3dd8ffa7-44e8-439e-8520-f2579ef81267": ("pov",
   "From the chameleon's own first-person perspective as it faces down the street (its own forward/back and its own left/right, not the viewer's screen-left/right), where is the sheriff's office?"),
 "bb57aeec-8b0e-4564-8db1-39b8c99966e1": ("pov",
   "From the chameleon's own first-person perspective (its own forward/back and own left/right as it faces), where is the bank of dirt?"),
 "1be09cd8-8620-4096-88f6-894bc74b446a": ("motion",
   "Considering the actual 3D positions of the turtle and the squirrel (which is nearer the camera, where each stands) and tracking the turtle across all frames, in which direction is the turtle walking relative to the squirrel?"),
 "cec0d7be-fb48-41f5-8174-14493b00400e": ("motion",
   "Taking into account where the green ducks are positioned relative to the crocodile and the camera, and tracking the crocodile's full path, in which direction is the crocodile running relative to the green ducks?"),
 "a8f02ec4-c692-46c9-9dea-77f3eaf9651c": ("motion",
   "Determine where the man in black is positioned in 3D (foreground vs background, screen side) and where the parrot goes; tracking the parrot's flight across the clip, in which direction is it flying relative to the man?"),
 "9121db06-d82a-421e-9809-d7ff9d032ab8": ("motion",
   "Determine where the man in the blue shirt stands in 3D (foreground vs background) and track the ducks' flight across the clip; in which direction are the green ducks flying relative to the man?"),
 "6fade3b6-2424-487b-916a-ed6db8de80ea": ("facing",
   "Determine each subject's actual gaze/facing direction in 3D — do NOT assume that because both appear to look toward the camera they face the same way. Relative to each other, how are the bear and the man in the orange suit facing?"),
 "50ca6f38-7a33-496c-a2c8-470923beee76": ("facing",
   "Determine the gorilla's true facing/gaze direction and the man's position in 3D (facing the camera does not necessarily mean facing the other character). Is the gorilla facing toward or away from the man in the red hat?"),
 "5f7b667b-4a2c-47dd-b1f2-b0520b30ed04": ("facing",
   "Considering each shark's actual heading/gaze direction in 3D across the clip (not merely which screen side each swims toward in separate shots), how are the sharks looking relative to each other?"),
 "72d36373-bf4d-40c8-8f41-d4fa572613f7": ("count",
   "Count every distinct reptile that appears at ANY moment, including any seen only briefly, in the background, or partially off-screen. How many reptiles appear in total?"),
 "5d45a6b7-d5ae-459d-a246-bc8d9bf57f29": ("count",
   "Count every distinct green duck that appears at any point, including briefly-seen or background ones. How many green ducks are present?"),
 "f47a6ddd-873e-4f81-af2a-5c870b4c2e7d": ("count",
   "Count only the arrows the leopard actually sticks into the wall and steps on to climb; do not count any arrow twice or any not used as a step. How many arrows does it use as steps?"),
 "97638972-67de-4371-8681-92bbafed3418": ("count",
   "Count all entrances/exits to the store shown or used at any point in the clip, including any used only briefly or seen in the background. How many entrances does the store have?"),
 "e0432a02-1bc4-44e3-8fc9-d3aece632076": ("count",
   "Watch the order in which animals cut the bridge ropes. Counting only animals that actually cut a rope, which animal is the SECOND to do so?"),
 "002a4291-3edc-40fb-a9cd-d6cb97d972da": ("vis",
   "Based on the 3D layout — positions of the birds and the small green piglet, their facing directions, and anything between them — can the birds see the small green piglet in distress?"),
 "2f4fb8cb-475e-4b75-8757-08b695837944": ("vis",
   "Based on the bearded man's position and facing direction and where the skull is in the 3D scene (in front of or behind him, anything blocking the view), can the bearded man see the skull?"),
 "f3b989d5-5f23-4caf-add1-2e3848ac904c": ("vert",
   "Relative to the red panda, where is the tree with pink leaves — judge DEPTH (in front/behind) and HEIGHT (above/below) separately, then pick the single best option."),
 "4de8a0e4-622f-41b8-bebf-415b4a20ae2f": ("vert",
   "Relative to the opossum, where is the porcupine — consider horizontal (left/right), vertical (above/on top/below) and depth; which single option best fits?"),
 "7c91940b-cbfa-47db-b41c-b37ee98a2434": ("vert",
   "Relative to the fireplace's position, where is the entrance to the kitchen (left/right, in front, above/below)? Consider the room's full layout across the clip."),
 "f5e223b2-73ac-4983-982c-8d6fa2573dcc": ("facing",
   "Determine the actual facing direction of the ticket seller and of the bus driver in 3D inside the bus. Relative to each other, are they facing toward, away, the same direction, or perpendicular?"),
 "7af4ffda-2a7e-4fa1-a1b7-372006445505": ("vert",
   "The man is lying on the floor. Relative to HIS body orientation (his head/feet and his own left/right side), where is the woman standing?"),
 "0d9a4a8b-d30e-4454-8713-0229aedaa21a": ("vert",
   "At the END of the clip, relative to the mirror's position, where is the cookie — judge depth (in front/behind) vs horizontal (left/right)?"),
 "23359178-4e5d-4c3a-aada-4058c6f0ebea": ("infer",
   "The man in black disappears from view; infer his actual hiding place from visual evidence across the clip (where he could have gone, what is shown afterward). Where did he hide?"),
}
STEER = {
 "pov": "IMPORTANT: Use the named subject's OWN egocentric frame (its own left/right/front/back as it faces), not the camera/screen frame. First work out the scene's 3D layout and which subjects are nearer vs farther from the camera across all cuts, then answer.",
 "motion": "IMPORTANT: First establish the 3D depth ordering (who is nearer/farther from the camera) and each subject's position, tracking movement across ALL frames and cuts. Moving into the background is not necessarily 'toward' a background subject. Then answer.",
 "facing": "IMPORTANT: A subject looking toward the camera does NOT mean it faces the other character. Determine each subject's true 3D gaze/heading and their relative positions across cuts, then answer.",
 "count": "IMPORTANT: Sample the ENTIRE timeline; include briefly-visible, background, and partially-occluded instances; avoid double-counting. Then answer.",
 "vis": "IMPORTANT: Reconstruct the 3D line of sight — relative positions, facing directions, and any occluders between them across cuts — then answer.",
 "vert": "IMPORTANT: Judge depth (front/behind) and height (above/below) SEPARATELY from screen left/right; reconstruct the 3D layout across cuts, then answer.",
 "infer": "IMPORTANT: The answer is not directly shown; reason from indirect visual evidence across the whole clip (entrances/exits, what appears afterward, plausibility), then answer.",
}

def get_file(client, qid, path, uploads):
    name = uploads.get(qid)
    if name:
        try:
            f = client.files.get(name=name)
            if f.state.name == "ACTIVE":
                return f
        except Exception:
            pass
    f = client.files.upload(file=path)
    while f.state.name == "PROCESSING":
        time.sleep(1.5); f = client.files.get(name=f.name)
    uploads[qid] = f.name
    with _lock:
        json.dump(uploads, open(UPLOADS, "w"))
    return f

def main():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    uploads = json.load(open(UPLOADS)) if UPLOADS.exists() else {}
    val = {json.loads(l)["question_id"]: json.loads(l)
           for l in open(config.RAW_DIR / "val_qa.jsonl")}
    clipdir = config.CLIPS_VAL_DIR
    jobs = []
    for qid, (mode, precise) in PROBE.items():
        row = val[qid]; opts = row["options"]
        variants = {
            "orig":    build_question_prompt(row["question_text"], opts, cot=True),
            "precise": build_question_prompt(precise, opts, cot=True),
            "steer":   build_question_prompt(row["question_text"] + "\n\n" + STEER[mode], opts, cot=True),
        }
        for arm, prompt in variants.items():
            jobs.append((qid, arm, mode, prompt, str(clipdir / f"{qid}.mp4"),
                         list(opts.keys()), row["answer_choice"]))

    def run(job):
        qid, arm, mode, prompt, clip, valid, gold = job
        cf = OUT / f"{qid}__{arm}.json"
        if cf.exists():
            txt = json.load(open(cf))["text"]
        else:
            f = get_file(client, qid, clip, uploads)
            cfg = types.GenerateContentConfig(temperature=0.0, max_output_tokens=2048)
            for a in range(6):
                try:
                    r = client.models.generate_content(
                        model="gemini-3.1-pro-preview",
                        contents=[f, SYSTEM_PROMPT + "\n\n" + prompt], config=cfg)
                    break
                except Exception as e:
                    if a == 5: raise
                    time.sleep(5 * (a + 1))
            txt = r.text or ""
            json.dump({"text": txt}, open(cf, "w"))
        choice = parse_choice(txt, valid, config.FALLBACK_CHOICE)
        return qid, arm, mode, choice, gold, choice == gold

    res = []; errors = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for fu in as_completed(futs):
            try:
                res.append(fu.result())
            except Exception as e:
                errors += 1
                if errors <= 2:
                    print(f"  [skip] {type(e).__name__}: {str(e)[:120]}")
    if errors:
        print(f"\n!! {errors}/{len(jobs)} calls failed (likely API spend cap / 429). "
              f"Cached successes are aggregated below; re-run to resume the rest for free.")
    if not res:
        print("No successful calls — nothing to aggregate (API blocked). Resolve billing and re-run.")
        return

    # aggregate
    import collections
    by = collections.defaultdict(dict)   # qid -> arm -> correct
    bymode = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for qid, arm, mode, choice, gold, ok in res:
        by[qid][arm] = ok
        bymode[mode][arm][0] += ok; bymode[mode][arm][1] += 1
    arms = ["orig", "precise", "steer"]
    n = len(by)
    print(f"\n=== {n} originally-wrong val questions, re-asked (temp 0) ===")
    for arm in arms:
        c = sum(by[q].get(arm, False) for q in by)
        print(f"  {arm:8} recovered {c}/{n} = {c/n:.3f}")
    print("\n  (orig should be ~0 if errors are stable at temp 0)")
    print("\n=== recovery by failure mode (precise / steer) ===")
    for mode, d in sorted(bymode.items()):
        o = d["orig"]; p = d["precise"]; s = d["steer"]
        print(f"  {mode:7} n={p[1]:2}  orig={o[0]}  precise={p[0]}  steer={s[0]}")
    print("\n=== per-question (qid8: orig/precise/steer) ===")
    for qid in by:
        print(f"  {qid[:8]}: {int(by[qid].get('orig',0))}/{int(by[qid].get('precise',0))}/{int(by[qid].get('steer',0))}")

if __name__ == "__main__":
    main()

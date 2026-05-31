"""Gemini 3 Pro native-video QA with self-consistency, fully cached.

Caching (so we never pay twice):
  - Uploaded video files: $CACHE/gemini/uploads.json  (qid -> file name, re-validated/re-uploaded if expired)
  - Per-question responses: $CACHE/gemini/resp/<model>/<qid>.json  (all samples + usage)
Re-runs read the response cache and make ZERO API calls for cached questions.

Usage:
  GEMINI_API_KEY=... python scripts/run_gemini.py --split test --model gemini-3.1-pro-preview \
      --num-samples 5 --temperature 0.7 [--limit N]
"""
from __future__ import annotations
import argparse, json, os, sys, time, threading
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from vrrqa import config  # noqa
from vrrqa.data import create_dataset  # noqa
from vrrqa.utils.prompts import (SYSTEM_PROMPT, DEPTH_SYSTEM_PROMPT, build_question_prompt,  # noqa
                                 build_depth_prompt, is_depth_question, parse_choice)
from vrrqa.utils.logutil import get_logger  # noqa

from google import genai
from google.genai import types

log = get_logger("vrrqa.gemini")
GCACHE = config.CACHE_DIR / "gemini"
(GCACHE / "resp").mkdir(parents=True, exist_ok=True)
UPLOADS = GCACHE / "uploads.json"
_lock = threading.Lock()
_cost = {"in": 0, "out": 0}  # token totals


def load_uploads():
    return json.load(open(UPLOADS)) if UPLOADS.exists() else {}

def save_uploads(d):
    with _lock:
        json.dump(d, open(UPLOADS, "w"))


def get_file(client, qid, path, uploads):
    """Return an ACTIVE Gemini File for this clip, reusing a cached upload if valid."""
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
    if f.state.name != "ACTIVE":
        raise RuntimeError(f"upload not active: {f.state.name}")
    uploads[qid] = f.name; save_uploads(uploads)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stratified-limit", type=int, default=0,
                    help="Pick N samples balanced across categories (deterministic) for cheap ablations.")
    ap.add_argument("--media-resolution", choices=["default", "low", "medium", "high"], default="default",
                    help="Gemini per-video token budget. 'default' = SDK default (current behaviour).")
    ap.add_argument("--fps", type=float, default=0.0,
                    help="Frame sampling rate for the video Part (0 = Gemini default ~1fps).")
    ap.add_argument("--depth-prompt", action="store_true",
                    help="Route depth/proximity/POV questions to the cue-guided depth prompt; "
                         "all other questions keep the baseline prompt (and reuse cached samples).")
    ap.add_argument("--tag", default="gemini31pro_sc")
    args = ap.parse_args()

    # Map resolution flag -> SDK enum (None keeps current default behaviour & cache key).
    _RES = {"low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
            "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
            "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH}
    media_res = _RES.get(args.media_resolution)
    # Cache-key suffix: ONLY non-default settings change the key, so the existing 81.18
    # baseline cache is reused as-is and new perception settings get fresh samples.
    perf_suffix = ""
    if args.media_resolution != "default":
        perf_suffix += f"__r{args.media_resolution}"
    if args.fps > 0:
        perf_suffix += f"__f{args.fps:g}"

    key = os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=key)
    respdir = GCACHE / "resp" / args.model.replace("/", "_"); respdir.mkdir(parents=True, exist_ok=True)
    uploads = load_uploads()

    ds = create_dataset(args.split)
    samples = ds._samples
    if args.stratified_limit:
        # Deterministic round-robin across categories so a small subset stays MacroAcc-balanced.
        from collections import OrderedDict, deque
        buckets = OrderedDict()
        for s in samples:
            buckets.setdefault(s.category, deque()).append(s)
        order = list(buckets.values())
        picked = []
        while len(picked) < args.stratified_limit and any(order):
            for q in order:
                if q:
                    picked.append(q.popleft())
                    if len(picked) >= args.stratified_limit:
                        break
        samples = picked
    elif args.limit:
        samples = samples[: args.limit]

    def do_one(sample):
        valid = sample.option_keys
        # Per-(qid,temperature) sample POOL: reuse any already-cached samples and only
        # generate the shortfall, so raising k (e.g. 5 -> 7) costs only the new samples.
        use_depth = args.depth_prompt and is_depth_question(sample.question)
        style_suffix = "__depth" if use_depth else ""
        pool_f = respdir / f"{sample.question_id}__t{args.temperature}{perf_suffix}{style_suffix}.json"
        texts = []
        if pool_f.exists():
            texts = json.load(open(pool_f)).get("texts", [])
        elif not (perf_suffix or style_suffix):
            # seed from legacy per-k cache ONLY for the baseline style/resolution, so a
            # depth/hi-res pool is never silently filled with baseline-prompt samples.
            legacy = sorted(respdir.glob(f"{sample.question_id}__k*_t{args.temperature}.json"))
            for lf in legacy:
                lt = json.load(open(lf)).get("texts", [])
                if len(lt) > len(texts):
                    texts = lt
        need = args.num_samples - len(texts)
        if need > 0:
            f = get_file(client, sample.question_id, sample.clip_path, uploads)
            # Attach explicit fps via VideoMetadata only when requested; else pass the File
            # object directly to preserve exact baseline behaviour.
            if args.fps > 0:
                vpart = types.Part(file_data=types.FileData(file_uri=f.uri, mime_type=f.mime_type),
                                   video_metadata=types.VideoMetadata(fps=args.fps))
            else:
                vpart = f
            if use_depth:
                prompt = DEPTH_SYSTEM_PROMPT + "\n\n" + build_depth_prompt(sample.question, sample.options)
            else:
                prompt = SYSTEM_PROMPT + "\n\n" + build_question_prompt(sample.question, sample.options, cot=True)
            for i in range(need):
                temp = 0.0 if args.num_samples == 1 else args.temperature
                cfg_kwargs = dict(temperature=temp, max_output_tokens=args.max_tokens)
                if media_res is not None:
                    cfg_kwargs["media_resolution"] = media_res
                cfg = types.GenerateContentConfig(**cfg_kwargs)
                for attempt in range(6):
                    try:
                        resp = client.models.generate_content(model=args.model, contents=[vpart, prompt], config=cfg)
                        break
                    except Exception as e:
                        if attempt == 5:
                            raise
                        time.sleep(5 * (attempt + 1))
                texts.append(resp.text or "")
                u = resp.usage_metadata
                with _lock:
                    _cost["in"] += u.prompt_token_count or 0
                    _cost["out"] += (u.total_token_count or 0) - (u.prompt_token_count or 0)
            json.dump({"texts": texts}, open(pool_f, "w"))
        texts = texts[:args.num_samples]
        votes = Counter(parse_choice(t, valid, config.FALLBACK_CHOICE) for t in texts if t)
        choice = votes.most_common(1)[0][0] if votes else config.FALLBACK_CHOICE
        return {"question_id": sample.question_id, "answer_choice": choice,
                "votes": dict(votes), "raw_text": (texts[0] if texts else "")}

    preds = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(do_one, s): s for s in samples}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"gemini[{args.split}]"):
            s = futs[fut]
            try:
                preds[s.question_id] = fut.result()
            except Exception as e:
                log.warning("failed %s: %s -> fallback", s.question_id, e)
                preds[s.question_id] = {"question_id": s.question_id, "answer_choice": config.FALLBACK_CHOICE, "votes": {}, "raw_text": ""}

    sub_suffix = perf_suffix + ("__depth" if args.depth_prompt else "") + (f"__strat{args.stratified_limit}" if args.stratified_limit else "")
    out = config.SUBMISSIONS_DIR / f"{args.tag}{sub_suffix}__{args.split}.json"
    json.dump([{"question_id": p["question_id"], "answer_choice": p["answer_choice"]} for p in preds.values()], open(out, "w"))
    with open(out.with_suffix(".rich.jsonl"), "w") as fh:
        for p in preds.values():
            fh.write(json.dumps(p) + "\n")
    cin = _cost["in"] / 1e6 * 2; cout = _cost["out"] / 1e6 * 12
    log.info("wrote %s (%d answers). LIVE API tokens: in=%dk out=%dk  cost=$%.2f (cached calls free)",
             out, len(preds), _cost["in"]//1000, _cost["out"]//1000, cin + cout)
    print(f"SUBMISSION_FILE={out}")

    if ds.has_labels:
        from collections import defaultdict
        gt = {s.question_id: (s.answer_choice, s.category) for s in ds}
        cc = defaultdict(int); ct = defaultdict(int); corr = 0
        for p in preds.values():
            a, c = gt[p["question_id"]]; ok = p["answer_choice"] == a; corr += ok; cc[c] += ok; ct[c] += 1
        print(f"GEMINI {args.split} AVG={corr/len(preds):.4f} MACRO={sum(cc[c]/ct[c] for c in ct)/len(ct):.4f}")


if __name__ == "__main__":
    main()

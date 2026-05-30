"""Download every unique YouTube video referenced by ImplicitQA, robustly and in parallel.

Keyed by YouTube video id (the prefix of video_id before the first '_', also == ?v= in url).
Saves <ytid>.mp4 to $VIDEO_DIR. Skips already-downloaded ids. Writes a miss list.
"""
import json, os, re, subprocess, pathlib, concurrent.futures as cf

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "ImplicitQAv0.1.2.jsonl"
VIDEO_DIR = pathlib.Path(os.environ.get("VIDEO_DIR", "/fs/scratch/PAS2301/alialavi/datasets/implicitqa/videos"))
LOG_DIR = VIDEO_DIR.parent / "logs"
VIDEO_DIR.mkdir(parents=True, exist_ok=True); LOG_DIR.mkdir(parents=True, exist_ok=True)
WORKERS = int(os.environ.get("DL_WORKERS", "8"))

def ytid(url, vid):
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else vid.split("_")[0]

rows = [json.loads(l) for l in open(SRC)]
ids = {}
for r in rows:
    yid = ytid(r["video_url"], r["video_id"])
    ids[yid] = f"https://www.youtube.com/watch?v={yid}"
print(f"{len(ids)} unique YouTube ids to fetch", flush=True)

def have(yid):
    return any((VIDEO_DIR / f"{yid}.{e}").exists() for e in ("mp4", "mkv", "webm"))

def dl(item):
    yid, url = item
    if have(yid):
        return (yid, "skip")
    out = str(VIDEO_DIR / f"{yid}.%(ext)s")
    cmd = ["yt-dlp", "--no-playlist", "--retries", "5", "--fragment-retries", "5",
           "--socket-timeout", "20", "--no-warnings", "--quiet",
           # cap to 480p mp4 to keep it small & fast; merge to mp4
           "-f", "bv*[height<=480]+ba/b[height<=480]/best",
           "--merge-output-format", "mp4", "-o", out, url]
    try:
        subprocess.run(cmd, check=True, timeout=600,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return (yid, "ok" if have(yid) else "noout")
    except Exception as e:
        return (yid, f"fail:{type(e).__name__}")

ok = miss = 0
with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex, open(LOG_DIR / "download_status.tsv", "w") as log:
    for yid, status in ex.map(dl, ids.items()):
        log.write(f"{yid}\t{status}\n"); log.flush()
        if status in ("ok", "skip"):
            ok += 1
        else:
            miss += 1; print(f"MISS {yid} {status}", flush=True)
print(f"DONE ok/skip={ok} miss={miss} -> {LOG_DIR/'download_status.tsv'}", flush=True)

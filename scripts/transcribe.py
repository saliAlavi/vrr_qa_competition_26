"""Transcribe every clip's audio with Whisper and cache one .txt per question_id.

Movie clips carry dialogue/sound that is often decisive for social, causal, and
counting questions. Transcripts are injected into the prompt by the data layer.
Cached at $CACHE/transcripts/<question_id>.txt  (empty file = no speech).
"""
from __future__ import annotations
import argparse, glob, os, sys, warnings
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vrrqa import config  # noqa
from vrrqa.utils.logutil import get_logger

warnings.filterwarnings("ignore")
log = get_logger("vrrqa.asr", logfile=config.LOGS_DIR / "transcribe.log")

OUT = config.CACHE_DIR / "transcripts"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", help="whisper size: base/small/medium")
    ap.add_argument("--splits", default="val,test")
    args = ap.parse_args()

    import torch
    from faster_whisper import WhisperModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ctype = "float16" if dev == "cuda" else "int8"
    log.info("loading faster-whisper '%s' on %s (%s)", args.model, dev, ctype)
    model = WhisperModel(args.model, device=dev, compute_type=ctype,
                         download_root=str(config.HF_HOME / "faster_whisper"))

    clips = []
    for sp in args.splits.split(","):
        d = config.CLIPS_VAL_DIR if sp == "val" else config.CLIPS_TEST_DIR
        clips += sorted(glob.glob(str(d / "*.mp4")))
    log.info("%d clips to transcribe", len(clips))

    done = skip = empty = 0
    for clip in tqdm(clips, desc="whisper"):
        qid = Path(clip).stem
        out = OUT / f"{qid}.txt"
        if out.exists():
            skip += 1
            continue
        try:
            segments, _ = model.transcribe(clip, beam_size=1, condition_on_previous_text=False)
            text = " ".join(s.text.strip() for s in segments).strip()
            out.write_text(text)
            done += 1
            if not text:
                empty += 1
        except Exception as e:
            log.warning("transcribe failed %s: %s", qid, e)
            out.write_text("")
    log.info("DONE transcribe: new=%d skip=%d empty=%d -> %s", done, skip, empty, OUT)


if __name__ == "__main__":
    main()

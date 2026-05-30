"""Central configuration: paths and constants.

Layout (per user spec):
  - datasets   -> $SCRATCH/datasets/implicitqa   (raw + extracted clips)
  - cache      -> $SCRATCH/cache/vrrqa           (frame cache, HF hub)
  - project    -> $SCRATCH/vrr_qa_competition_26 (logs, runs, models, submissions)
"""
from __future__ import annotations
import os
from pathlib import Path

SCRATCH = Path(os.environ.get("VRRQA_SCRATCH", "/fs/scratch/PAS2301/alialavi"))

DATA_DIR = SCRATCH / "datasets" / "implicitqa"
RAW_DIR = DATA_DIR / "raw"
CLIPS_VAL_DIR = DATA_DIR / "clips_val" / "all_videos"
CLIPS_TEST_DIR = DATA_DIR / "clips_test" / "all_test_clips"
VAL_QA = RAW_DIR / "val_qa.jsonl"
TEST_QA = RAW_DIR / "test_qa.json"

CACHE_DIR = SCRATCH / "cache" / "vrrqa"
FRAME_CACHE_DIR = CACHE_DIR / "frames"
HF_HOME = SCRATCH / "cache"

PROJECT_DIR = SCRATCH / "vrr_qa_competition_26"
RUNS_DIR = PROJECT_DIR / "runs"
LOGS_DIR = PROJECT_DIR / "logs"
SUBMISSIONS_DIR = PROJECT_DIR / "submissions"
MODELS_DIR = PROJECT_DIR / "models"

# EvalAI
CHALLENGE_ID = 2682
PHASE_VAL = 5313
PHASE_TEST = 5314

# Answer prior on the validation set (B is the majority answer) -> safe fallback.
FALLBACK_CHOICE = "B"

WANDB_PROJECT = "vrrqa"

for _d in (CACHE_DIR, FRAME_CACHE_DIR, RUNS_DIR, LOGS_DIR, SUBMISSIONS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

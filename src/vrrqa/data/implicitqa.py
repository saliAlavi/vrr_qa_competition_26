"""ImplicitQA / VRR-QA dataset for the validation and test splits."""
from __future__ import annotations
import json
from pathlib import Path
from typing import List
from PIL import Image

from .base import AbstractDataset, QASample
from .frames import load_frames_cached
from .. import config
from ..utils.logutil import get_logger

log = get_logger("vrrqa.data")


class ImplicitQADataset(AbstractDataset):
    """Val split has labels; test split does not. Clips are named <question_id>.mp4."""

    def __init__(self, split: str, use_frame_cache: bool = True):
        assert split in ("val", "test")
        self.name = f"implicitqa_{split}"
        self.split = split
        self.use_frame_cache = use_frame_cache
        if split == "val":
            self.clip_dir = config.CLIPS_VAL_DIR
            rows = [json.loads(l) for l in open(config.VAL_QA)]
        else:
            self.clip_dir = config.CLIPS_TEST_DIR
            rows = json.load(open(config.TEST_QA))
        self._tx_dir = config.CACHE_DIR / "transcripts"
        self._samples: List[QASample] = []
        missing = 0
        n_tx = 0
        for r in rows:
            clip = self.clip_dir / f"{r['question_id']}.mp4"
            if not clip.exists():
                missing += 1
            txf = self._tx_dir / f"{r['question_id']}.txt"
            transcript = ""
            if txf.exists():
                transcript = txf.read_text().strip()
                if transcript:
                    n_tx += 1
            self._samples.append(QASample(
                transcript=transcript,
                question_id=r["question_id"],
                video_id=r.get("video_id", ""),
                clip_path=str(clip),
                question=r["question_text"],
                options=r["options"],
                question_start_time=r.get("question_start_time", 0.0),
                question_stop_time=r.get("question_stop_time", 0.0),
                category=r.get("category"),
                answer_choice=r.get("answer_choice"),
            ))
        log.info("loaded %s: %d samples (%d missing clips, %d with transcripts)",
                 self.name, len(self._samples), missing, n_tx)

    @property
    def has_labels(self) -> bool:
        return self.split == "val"

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> QASample:
        return self._samples[idx]

    def load_frames(self, sample: QASample, num_frames: int, max_side: int) -> List[Image.Image]:
        return load_frames_cached(sample.clip_path, num_frames, max_side, self.use_frame_cache)

"""Data layer: abstract dataset + the QASample contract shared with the runner.

The runner and model never touch raw files directly; they go through this interface:
  - dataset[i]            -> QASample
  - dataset.load_frames(sample, num_frames, max_side) -> List[PIL.Image]
So download/preprocess/caching can change without touching model or runner code.
"""
from __future__ import annotations
import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from PIL import Image


@dataclass
class QASample:
    question_id: str
    video_id: str
    clip_path: str
    question: str
    options: Dict[str, str]
    question_start_time: float = 0.0
    question_stop_time: float = 0.0
    category: Optional[str] = None
    answer_choice: Optional[str] = None  # None for the (hidden) test split
    transcript: str = ""                 # Whisper ASR of the clip audio (optional)

    @property
    def option_keys(self) -> List[str]:
        return sorted(self.options.keys())


class AbstractDataset(abc.ABC):
    """Common interface for any QA split."""

    name: str = "abstract"

    @property
    @abc.abstractmethod
    def has_labels(self) -> bool: ...

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> QASample: ...

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @abc.abstractmethod
    def load_frames(self, sample: QASample, num_frames: int, max_side: int) -> List[Image.Image]:
        """Return `num_frames` uniformly sampled RGB frames from the sample's clip."""
        ...

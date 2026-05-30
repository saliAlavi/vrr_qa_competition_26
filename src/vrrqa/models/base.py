"""Model layer: abstract video-QA model + Prediction contract.

The runner reads `num_frames`/`max_side` to know how to feed the model, then calls
predict_batch with (sample, frames) pairs. Any new VLM only implements this interface.
"""
from __future__ import annotations
import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from PIL import Image

from ..data.base import QASample


@dataclass
class Prediction:
    question_id: str
    answer_choice: str
    raw_text: str = ""
    votes: Optional[Dict[str, int]] = None  # for self-consistency
    meta: dict = field(default_factory=dict)


class AbstractVideoQAModel(abc.ABC):
    """Interface every model implementation must satisfy."""

    version: str = "abstract"      # used in the wandb run name
    num_frames: int = 16           # frames the runner should extract
    max_side: int = 448            # max image side the runner should resize to

    @abc.abstractmethod
    def setup(self) -> None:
        """Load weights / start engine."""
        ...

    @abc.abstractmethod
    def predict_batch(self, items: List[Tuple[QASample, List[Image.Image]]]) -> List[Prediction]:
        """Predict an option letter for each (sample, frames) pair."""
        ...

    def teardown(self) -> None:
        pass

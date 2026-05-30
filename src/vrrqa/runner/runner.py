"""Runner: orchestrates the Data and Model layers through their stable interfaces.

Pulls frames from the dataset, asks the model to predict, scores (when labels exist),
logs to wandb, and writes an EvalAI-format submission JSON.
"""
from __future__ import annotations
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from ..data.base import AbstractDataset, QASample
from ..models.base import AbstractVideoQAModel, Prediction
from ..utils.logutil import get_logger
from .. import config

log = get_logger("vrrqa.runner")


class Runner:
    def __init__(self, dataset: AbstractDataset, model: AbstractVideoQAModel,
                 batch_size: int = 8, run_name: Optional[str] = None,
                 use_wandb: bool = True, wandb_config: Optional[dict] = None):
        self.dataset = dataset
        self.model = model
        self.batch_size = batch_size
        self.run_name = run_name or f"{config.WANDB_PROJECT}__{model.version}__{time.strftime('%Y%m%d-%H%M%S')}"
        self.use_wandb = use_wandb
        self.wandb_config = wandb_config or {}
        self._wandb = None

    def _init_wandb(self):
        if not self.use_wandb:
            return
        try:
            import wandb
            self._wandb = wandb.init(
                project=config.WANDB_PROJECT, name=self.run_name,
                dir=str(config.RUNS_DIR),
                config={"model": self.model.version, "dataset": self.dataset.name,
                        "num_frames": self.model.num_frames, "max_side": self.model.max_side,
                        "batch_size": self.batch_size, **self.wandb_config},
            )
        except Exception as e:
            log.warning("wandb init failed (%s); continuing without it", e)
            self._wandb = None

    def run(self) -> List[Prediction]:
        self._init_wandb()
        self.model.setup()
        preds: List[Prediction] = []
        n = len(self.dataset)
        t0 = time.time()
        fail = 0
        for start in tqdm(range(0, n, self.batch_size), desc=f"infer[{self.dataset.name}]"):
            batch = [self.dataset[i] for i in range(start, min(start + self.batch_size, n))]
            items = []
            for s in batch:
                try:
                    frames = self.dataset.load_frames(s, self.model.num_frames, self.model.max_side)
                    items.append((s, frames))
                except Exception as e:
                    fail += 1
                    log.warning("frame load failed for %s: %s -> fallback", s.question_id, e)
                    preds.append(Prediction(s.question_id, config.FALLBACK_CHOICE, meta={"frame_fail": True}))
            if not items:
                continue
            try:
                preds.extend(self.model.predict_batch(items))
            except Exception as e:
                log.error("predict_batch failed (%s) -> fallback for %d items", e, len(items))
                fail += len(items)
                for s, _ in items:
                    preds.append(Prediction(s.question_id, config.FALLBACK_CHOICE, meta={"predict_fail": True}))
            if self._wandb is not None:
                self._wandb.log({"progress": len(preds) / n})
        dt = time.time() - t0
        log.info("inference done: %d preds in %.1fs (%.2fs/q), fallbacks=%d", len(preds), dt, dt / max(n, 1), fail)

        metrics = self.score(preds)
        if metrics and self._wandb is not None:
            self._wandb.log(metrics)
            self._wandb.summary.update(metrics)
        if self._wandb is not None:
            self._wandb.finish()
        return preds

    def score(self, preds: List[Prediction]) -> dict:
        if not self.dataset.has_labels:
            log.info("no labels for %s -> skipping scoring", self.dataset.name)
            return {}
        gt = {s.question_id: (s.answer_choice, s.category) for s in self.dataset}
        by_cat_correct = defaultdict(int)
        by_cat_total = defaultdict(int)
        correct = 0
        for p in preds:
            ans, cat = gt[p.question_id]
            ok = int(p.answer_choice == ans)
            correct += ok
            by_cat_correct[cat] += ok
            by_cat_total[cat] += 1
        overall = correct / max(len(preds), 1)
        cat_acc = {c: by_cat_correct[c] / by_cat_total[c] for c in by_cat_total}
        macro = sum(cat_acc.values()) / max(len(cat_acc), 1)
        metrics = {"avg_accuracy": overall, "macro_accuracy": macro}
        log.info("AVG=%.4f  MACRO=%.4f", overall, macro)
        for c in sorted(cat_acc):
            log.info("  %-40s %.4f (n=%d)", c, cat_acc[c], by_cat_total[c])
            metrics[f"cat/{c}"] = cat_acc[c]
        return metrics

    def write_submission(self, preds: List[Prediction], out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [{"question_id": p.question_id, "answer_choice": p.answer_choice} for p in preds]
        json.dump(payload, open(out_path, "w"))
        # also dump rich predictions for analysis
        rich = out_path.with_suffix(".rich.jsonl")
        with open(rich, "w") as f:
            for p in preds:
                f.write(json.dumps({"question_id": p.question_id, "answer_choice": p.answer_choice,
                                    "votes": p.votes, "raw_text": p.raw_text[:2000]}) + "\n")
        log.info("wrote submission -> %s (%d answers)", out_path, len(payload))
        return out_path

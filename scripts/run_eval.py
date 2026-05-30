"""CLI entry point: run a model over a split, score (val), and write a submission.

Example:
  python scripts/run_eval.py --split val --model qwen2.5-vl-7b --limit 100
  python scripts/run_eval.py --split test --model qwen2.5-vl-72b-awq --num-samples 5 --temperature 0.6
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrrqa import config  # noqa: E402
from vrrqa.data import create_dataset  # noqa: E402
from vrrqa.models import create_model  # noqa: E402
from vrrqa.runner import Runner  # noqa: E402
from vrrqa.utils.logutil import get_logger  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-frames", type=int, default=None)
    ap.add_argument("--max-side", type=int, default=None)
    ap.add_argument("--tp", type=int, default=None, help="tensor parallel size")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--gpu-util", type=float, default=None)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--num-samples", type=int, default=1, help="self-consistency samples")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--no-cot", action="store_true")
    ap.add_argument("--prompt-style", choices=["plain", "reasoner", "spatial"], default="plain")
    ap.add_argument("--use-audio", action="store_true", help="inject Whisper transcripts into the prompt")
    ap.add_argument("--mode", choices=["single", "cascade", "decompose"], default="single")
    ap.add_argument("--answer-text-only", action="store_true",
                    help="cascade: answer from the description alone (drop frames in stage 2)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N samples")
    ap.add_argument("--num-shards", type=int, default=1, help="data-parallel: total shards")
    ap.add_argument("--shard-id", type=int, default=0, help="data-parallel: this shard index")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the run/submission name")
    ap.add_argument("--run-id", default="", help="fixed run prefix shared across shards")
    args = ap.parse_args()

    shard_sfx = f"-sh{args.shard_id}of{args.num_shards}" if args.num_shards > 1 else ""
    base = args.run_id or f"{config.WANDB_PROJECT}__{args.model}{('-'+args.tag) if args.tag else ''}__{time.strftime('%Y%m%d-%H%M%S')}"
    run_name = f"{base}{shard_sfx}"
    log = get_logger("vrrqa.cli", logfile=config.LOGS_DIR / f"{run_name}.log")
    log.info("args: %s", vars(args))

    dataset = create_dataset(args.split)
    if args.limit:
        dataset._samples = dataset._samples[: args.limit]
    if args.num_shards > 1:
        dataset._samples = dataset._samples[args.shard_id::args.num_shards]
        log.info("shard %d/%d -> %d samples", args.shard_id, args.num_shards, len(dataset))

    mkwargs = dict(num_samples=args.num_samples, temperature=args.temperature,
                   cot=not args.no_cot, prompt_style=args.prompt_style,
                   use_audio=args.use_audio, mode=args.mode,
                   answer_with_video=not args.answer_text_only)
    if args.max_tokens is not None:
        mkwargs["max_tokens"] = args.max_tokens
    if args.num_frames is not None:
        mkwargs["num_frames"] = args.num_frames
    if args.max_side is not None:
        mkwargs["max_side"] = args.max_side
    if args.tp is not None:
        mkwargs["tensor_parallel_size"] = args.tp
    if args.max_model_len is not None:
        mkwargs["max_model_len"] = args.max_model_len
    if args.gpu_util is not None:
        mkwargs["gpu_memory_utilization"] = args.gpu_util
    if args.enforce_eager:
        mkwargs["enforce_eager"] = True
    model = create_model(args.model, **mkwargs)

    runner = Runner(dataset, model, batch_size=args.batch_size, run_name=run_name,
                    use_wandb=not args.no_wandb, wandb_config=vars(args))
    preds = runner.run()

    sub = config.SUBMISSIONS_DIR / f"{run_name}__{args.split}.json"
    runner.write_submission(preds, sub)
    print(f"SUBMISSION_FILE={sub}")


if __name__ == "__main__":
    main()

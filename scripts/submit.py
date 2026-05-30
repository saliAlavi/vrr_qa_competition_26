"""Submit a prediction JSON to EvalAI via evalai-cli.

Usage:
  python scripts/submit.py --file <submission.json> --phase test [--public]
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vrrqa import config  # noqa: E402

EVALAI = "/fs/scratch/PAS2301/alialavi/pylibs/evalai_venv/bin/evalai"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--phase", choices=["val", "test"], default="test")
    ap.add_argument("--public", action="store_true", help="show on public leaderboard (else private)")
    ap.add_argument("--large", action="store_true", default=True)
    args = ap.parse_args()

    phase_id = config.PHASE_TEST if args.phase == "test" else config.PHASE_VAL
    cmd = [EVALAI, "challenge", str(config.CHALLENGE_ID), "phase", str(phase_id),
           "submit", "--file", str(args.file)]
    if args.large:
        cmd.append("--large")
    cmd.append("--public" if args.public else "--private")
    print("RUN:", " ".join(cmd))
    # evalai interactively asks: include Submission Details? / Metadata? -> answer N to both
    subprocess.run(cmd, input="N\nN\n", text=True, check=True)


if __name__ == "__main__":
    main()

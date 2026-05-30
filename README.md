# VRR Challenge @ CVPR 2026 (ImplicitQA / VRR-QA) — Code Submission

**Author:** Ali Alavi, The Ohio State University (`alavibajestan.1@osu.edu`)

Training-free system for implicit video question answering. Best result on the hidden test split:
**AvgAcc 81.18 / MacroAvgAcc 78.85** (Gemini 3.1 Pro, native video, self-consistency k=5),
surpassing the prior leaderboard best (80.85). See `docs/technical_report/report.pdf`.

## Results (test split, n=172)
| System | AvgAcc | MacroAcc |
|---|---|---|
| Qwen2.5-VL-7B (CoT) | 50.2 | 47.3 |
| Qwen3-VL-32B-AWQ + self-consistency (open-source ceiling) | 58.5 | 56.2 |
| **Gemini 3.1 Pro + self-consistency k=5** | **81.18** | **78.85** |
| *prior leaderboard best* | *80.85* | — |

## Approach (one paragraph)
The benchmark is *perception-bound*, not reasoning-bound. We feed each pre-trimmed clip as **native
video** to a frontier model (Gemini 3.1 Pro), sample **k=5 chain-of-thought answers** at temperature
0.7, and take a **majority vote** (self-consistency). All open-source baselines (Qwen2.5-VL,
Qwen3-VL, InternVL3, Gemma-3, Video-R1, VideoChat-R1.5) and inference strategies (decomposition,
cascade, routing, audio) were ablated on the labeled validation split; only stronger perception +
self-consistency transferred to test. The open-source ceiling reached was 58.5.

## Repository layout
```
src/vrrqa/            # 3-layer pipeline (factory + abstract classes)
  data/               # dataset, frame sampling + caching, audio transcripts
  models/             # vLLM open-weights model + (via run_gemini.py) Gemini API
  runner/             # ties data+model, scoring, submission JSON
  utils/              # prompts (CoT/reasoner/spatial/decompose), parsing, logging
scripts/
  run_gemini.py       # FINAL system: Gemini 3.1 Pro + self-consistency (cached)
  run_eval.py         # open-weights eval (vLLM); --mode single|cascade|decompose
  run_dp.sbatch       # data-parallel multi-GPU inference + merge
  run.sbatch          # single-GPU SLURM launcher
  ensemble.py route_predict.py classify_cat.py merge_shards.py   # ablation tooling
  transcribe.py       # faster-whisper audio transcripts
  submit.py           # EvalAI submission via evalai-cli
docs/
  technical_report/   # CVPR-2026-format report (report.tex/pdf, cvpr.sty, figure)
  NEXT_STEPS.md
submissions/
  gemini31pro_sc5__test.json   # WINNING test predictions (81.18 AvgAcc)
data/ImplicitQAv0.1.2.jsonl    # public validation annotations
```

## Reproduce the winning submission
```bash
pip install -r requirements.txt
# clips: organizer-provided trimmed test clips named <question_id>.mp4
export GEMINI_API_KEY=<your_key>
export PYTHONPATH=src
python scripts/run_gemini.py --split test --model gemini-3.1-pro-preview \
    --num-samples 5 --temperature 0.7 --workers 12
# -> writes submissions/gemini31pro_sc5__test.json ; responses cached (re-runs free)
```
Paths (clip/cache/output locations) are set in `src/vrrqa/config.py` (env-overridable).

## Reproduce open-weights baselines (vLLM, GPU)
```bash
sbatch --gpus-per-node=2 scripts/run_dp.sbatch val q3vl32_sc5 \
    --model qwen3-vl-32b-awq --enforce-eager --num-samples 5 --temperature 0.7
```

## Notes
- No training is performed; the system is inference + test-time compute only.
- The Gemini API key is read from `$GEMINI_API_KEY` and is **not** stored in any file.
- The ImplicitQA/VRR-QA annotations and clips belong to their original authors; this repo only
  ships code (and the public validation annotation file for convenience).

## License
Code is released under the **MIT License** (see `LICENSE`). Benchmark data/clips retain their
original licenses.

## Citation
If you use this code, please cite the ImplicitQA/VRR-QA benchmark (arXiv:2506.21742) and this report:
```
@techreport{alavi2026perceptionfirst,
  title  = {Perception First: A Frontier Native-Video Model with Self-Consistency
            for Implicit Video Question Answering},
  author = {Ali Alavi},
  year   = {2026},
  institution = {The Ohio State University},
  note   = {VRR Challenge @ CVPR 2026 technical report}
}
```

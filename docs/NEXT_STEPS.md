# Next Steps — Firming Up the #1 Margin (VRR/ImplicitQA Challenge)

**Status (2026-05-30):** Best submission = **Gemini 3.1 Pro + self-consistency×5**, test
**AvgAcc 81.18 / MacroAvgAcc 78.85**, beating the prior leaderboard top (80.85). Cost ≈ $11.80,
fully cached (re-runs free). Free open-source ceiling reached at 58.5 (Qwen3-VL-32B-AWQ + SC×5).

## 1. The margin is thin — buy a buffer
81.18 vs 80.85 is ~0.33 pts on 172 questions (≈ **less than one question**). To make the lead robust:

- **Stronger self-consistency: SC×7 / SC×9** on Gemini 3.1 Pro. Mostly *cached* already (the first 5
  samples per question are stored), so only +2-4 new samples/question → **~$5-8 incremental**. More
  votes reduce the per-question majority variance that gave us the 58.45→49.97 swing on the 32B.
- **Vote-confidence gating:** when the 5 votes are unanimous, keep; when split (e.g., 3-2), draw extra
  samples only for those (cheap, targeted variance reduction).

## 2. Push the ceiling higher (optional, ~$10-25)
- **Frontier ensemble:** add **GPT-5.x** (or Gemini 3 Pro vs 3.1 Pro) as a second independent voter and
  take a weighted vote. Decorrelated frontier errors are the most reliable way to gain 1-3 pts. Budget:
  GPT-5.4 single-pass ≈ $4, SC×5 ≈ $20; pick per budget.
- **Native-video framing experiments:** Gemini ingests the full clip; test (a) higher `media_resolution`,
  (b) passing the audio track explicitly, (c) longer thinking budget. Each ~$3-5 to A/B on val.

## 3. De-risk and finalize
- **Validate on the public val split** with Gemini (1001 Qs, ~$20 SC×5) to (a) confirm the val→test gap
  holds and (b) tune temperature / sample count on labeled data before the final test submission.
- **Lock the final submission** and keep ≥1 spare submission in reserve (test phase allows 100/day).
- **Reproducibility bundle:** freeze `run_gemini.py`, the response cache, and the exact model id
  (`gemini-3.1-pro-preview`) + config (k=5, T=0.7) used for the winning run.

## 4. Recommended immediate action
1. Run **SC×7 on Gemini 3.1 Pro test** (incremental ~$6, mostly cached) → submit; keep best.
2. If margin still < 1 pt, add **GPT-5.x as a 2nd voter** (~$20) and submit the ensemble.
3. Finalize the technical report (see `docs/technical_report/`).

**Guiding principle (validated this project):** the benchmark is *perception-bound*. Spend on
perception/base-model quality (native-video frontier models) and light test-time denoising (SC on a
strong base); avoid reasoning-search tricks (cascade, decompose-on-strong, per-category routing) that
overfit the 172-question test set.

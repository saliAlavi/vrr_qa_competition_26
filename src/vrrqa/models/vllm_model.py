"""vLLM-backed multimodal model. Frames are passed as multi-image input via base64
data URLs through vLLM's chat API, so it works for Qwen2.5-VL, InternVL, etc.

Supports self-consistency (n>1 sampled reasoning paths -> majority vote).
"""
from __future__ import annotations
import base64
import io
from typing import Dict, List, Tuple
from collections import Counter
from PIL import Image

from .base import AbstractVideoQAModel, Prediction
from ..data.base import QASample
from ..utils.prompts import (SYSTEM_PROMPT, build_question_prompt, parse_choice,
                             DESCRIBE_PROMPT, build_answer_from_desc,
                             build_decompose_prompt, build_subanswer_prompt,
                             build_aggregate_prompt)
from ..utils.logutil import get_logger
from .. import config

log = get_logger("vrrqa.model")


def _to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


class VLLMVideoQAModel(AbstractVideoQAModel):
    def __init__(
        self,
        model_id: str,
        version: str,
        num_frames: int = 16,
        max_side: int = 448,
        tensor_parallel_size: int = 1,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.90,
        quantization: str | None = None,
        num_samples: int = 1,
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        cot: bool = True,
        seed: int = 0,
        dtype: str | None = None,
        prompt_style: str = "plain",
        use_audio: bool = False,
        mode: str = "single",           # "single" | "cascade"
        answer_with_video: bool = True,  # cascade: keep frames in the answer stage
        describe_max_tokens: int = 640,
        enforce_eager: bool = False,     # disable CUDA graphs (fixes some quant crashes)
    ):
        self.model_id = model_id
        self.version = version
        self.num_frames = num_frames
        self.max_side = max_side
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.quantization = quantization
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.cot = cot
        self.prompt_style = prompt_style
        self.use_audio = use_audio
        self.mode = mode
        self.answer_with_video = answer_with_video
        self.describe_max_tokens = describe_max_tokens
        self.enforce_eager = enforce_eager
        self.seed = seed
        # AWQ kernels only support float16; default others to bfloat16.
        self.dtype = dtype or ("float16" if quantization == "awq" else "bfloat16")
        # InternVL's HF chat template rejects OpenAI content-lists -> use the
        # generate() path with explicit <image> placeholders + multi_modal_data.
        self.family = "internvl" if "intern" in model_id.lower() else "qwen"
        self._llm = None
        self._sp = None
        self._tok = None

    def setup(self) -> None:
        from vllm import LLM, SamplingParams
        log.info("loading vLLM model %s (tp=%d, frames=%d, max_side=%d, quant=%s)",
                 self.model_id, self.tensor_parallel_size, self.num_frames, self.max_side, self.quantization)
        self._llm = LLM(
            model=self.model_id,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            limit_mm_per_prompt={"image": self.num_frames},
            quantization=self.quantization,
            trust_remote_code=True,
            seed=self.seed,
            dtype=self.dtype,
            enforce_eager=self.enforce_eager,
        )
        self._sp = SamplingParams(
            n=self.num_samples,
            temperature=self.temperature if self.num_samples > 1 else 0.0,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        self._desc_sp = SamplingParams(n=1, temperature=0.0, top_p=self.top_p,
                                       max_tokens=self.describe_max_tokens)

    def _build_messages(self, sample: QASample, frames: List[Image.Image]):
        content = [{"type": "image_url", "image_url": {"url": _to_data_url(f)}} for f in frames]
        content.append({"type": "text", "text": build_question_prompt(sample.question, sample.options, cot=self.cot, style=self.prompt_style, transcript=(sample.transcript if self.use_audio else ''))})
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def _build_generate_req(self, sample: QASample, frames: List[Image.Image]) -> dict:
        """InternVL path: string prompt with <image> placeholders + multi_modal_data."""
        if self._tok is None:
            self._tok = self._llm.get_tokenizer()
        placeholders = "".join(f"Frame{i+1}: <image>\n" for i in range(len(frames)))
        user = placeholders + build_question_prompt(sample.question, sample.options, cot=self.cot, style=self.prompt_style, transcript=(sample.transcript if self.use_audio else ''))
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        prompt = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "multi_modal_data": {"image": frames}}

    def _imgs(self, frames: List[Image.Image]):
        return [{"type": "image_url", "image_url": {"url": _to_data_url(f)}} for f in frames]

    def _describe_messages(self, frames: List[Image.Image]):
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._imgs(frames) + [{"type": "text", "text": DESCRIBE_PROMPT}]}]

    def _cascade_answer_messages(self, sample: QASample, frames: List[Image.Image], desc: str):
        text = build_answer_from_desc(sample.question, sample.options, desc, self.answer_with_video)
        content = (self._imgs(frames) if self.answer_with_video else []) + [{"type": "text", "text": text}]
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]

    def _predict_cascade(self, items):
        # Stage 1: blind, thorough description of each clip.
        desc_convs = [self._describe_messages(f) for _, f in items]
        desc_outs = self._llm.chat(desc_convs, self._desc_sp, use_tqdm=False)
        descs = [o.outputs[0].text.strip() for o in desc_outs]
        # Stage 2: answer using the description (+ frames if answer_with_video).
        ans_convs = [self._cascade_answer_messages(s, f, d) for (s, f), d in zip(items, descs)]
        ans_outs = self._llm.chat(ans_convs, self._sp, use_tqdm=False)
        preds = []
        for (sample, _), out, desc in zip(items, ans_outs, descs):
            valid = sample.option_keys
            votes: Counter = Counter()
            for o in out.outputs:
                votes[parse_choice(o.text, valid, config.FALLBACK_CHOICE)] += 1
            choice = votes.most_common(1)[0][0] if votes else config.FALLBACK_CHOICE
            preds.append(Prediction(question_id=sample.question_id, answer_choice=choice,
                                    raw_text=out.outputs[0].text if out.outputs else "",
                                    votes=dict(votes), meta={"description": desc[:1500]}))
        return preds

    def _predict_decompose(self, items):
        # Stage 1: generate atomic sub-questions (text-only, question-conditioned).
        dec_convs = [[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": build_decompose_prompt(s.question, s.options)}]
                     for s, _ in items]
        dec_outs = self._llm.chat(dec_convs, self._desc_sp, use_tqdm=False)
        subqs = [o.outputs[0].text.strip() for o in dec_outs]
        # Stage 2: answer the sub-questions over the frames.
        sa_convs = [[{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": self._imgs(f) + [{"type": "text", "text": build_subanswer_prompt(sq)}]}]
                    for (s, f), sq in zip(items, subqs)]
        sa_outs = self._llm.chat(sa_convs, self._desc_sp, use_tqdm=False)
        findings = [o.outputs[0].text.strip() for o in sa_outs]
        # Stage 3: aggregate findings + frames -> final option.
        agg_convs = [[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": self._imgs(f) + [{"type": "text", "text": build_aggregate_prompt(s.question, s.options, fi)}]}]
                     for (s, f), fi in zip(items, findings)]
        agg_outs = self._llm.chat(agg_convs, self._sp, use_tqdm=False)
        preds = []
        for (sample, _), out, sq, fi in zip(items, agg_outs, subqs, findings):
            valid = sample.option_keys
            votes: Counter = Counter()
            for o in out.outputs:
                votes[parse_choice(o.text, valid, config.FALLBACK_CHOICE)] += 1
            choice = votes.most_common(1)[0][0] if votes else config.FALLBACK_CHOICE
            preds.append(Prediction(question_id=sample.question_id, answer_choice=choice,
                                    raw_text=out.outputs[0].text if out.outputs else "",
                                    votes=dict(votes),
                                    meta={"subquestions": sq[:800], "findings": fi[:800]}))
        return preds

    def predict_batch(self, items: List[Tuple[QASample, List[Image.Image]]]) -> List[Prediction]:
        if self.mode == "cascade":
            return self._predict_cascade(items)
        if self.mode == "decompose":
            return self._predict_decompose(items)
        if self.family == "internvl":
            reqs = [self._build_generate_req(s, f) for s, f in items]
            outs = self._llm.generate(reqs, self._sp, use_tqdm=False)
        else:
            conversations = [self._build_messages(s, f) for s, f in items]
            outs = self._llm.chat(conversations, self._sp, use_tqdm=False)
        preds: List[Prediction] = []
        for (sample, _), out in zip(items, outs):
            valid = sample.option_keys
            votes: Counter = Counter()
            texts = []
            for o in out.outputs:
                txt = o.text
                texts.append(txt)
                votes[parse_choice(txt, valid, config.FALLBACK_CHOICE)] += 1
            choice = votes.most_common(1)[0][0] if votes else config.FALLBACK_CHOICE
            preds.append(Prediction(
                question_id=sample.question_id,
                answer_choice=choice,
                raw_text=texts[0] if texts else "",
                votes=dict(votes),
            ))
        return preds

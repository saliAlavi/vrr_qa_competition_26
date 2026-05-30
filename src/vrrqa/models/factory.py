"""Model factory + a registry of preset configs (data-driven so any field is overridable).

Add a new VLM by registering a preset dict; runner/scripts refer to it by name only.
"""
from __future__ import annotations
from typing import Dict
from .base import AbstractVideoQAModel
from .vllm_model import VLLMVideoQAModel

# Default kwargs per preset. Anything passed to create_model() overrides these.
_PRESETS: Dict[str, dict] = {
    "qwen2.5-vl-7b": dict(model_id="Qwen/Qwen2.5-VL-7B-Instruct", version="qwen2.5-vl-7b",
                          num_frames=16, max_side=448, tensor_parallel_size=1),
    # --- Qwen3-VL (newest base; single-GPU) ---
    "qwen3-vl-8b": dict(model_id="Qwen/Qwen3-VL-8B-Instruct", version="qwen3-vl-8b",
                        num_frames=16, max_side=448, tensor_parallel_size=1, max_model_len=16384),
    "qwen3-vl-30b-fp8": dict(model_id="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8", version="qwen3-vl-30b-fp8",
                             num_frames=16, max_side=448, tensor_parallel_size=1,
                             max_model_len=12288, gpu_memory_utilization=0.92),
    "qwen3-vl-32b-awq": dict(model_id="QuantTrio/Qwen3-VL-32B-Instruct-AWQ", version="qwen3-vl-32b-awq",
                             num_frames=16, max_side=448, tensor_parallel_size=1, quantization="awq",
                             max_model_len=12288, gpu_memory_utilization=0.93),
    "gemma3-12b": dict(model_id="unsloth/gemma-3-12b-it", version="gemma3-12b",
                       num_frames=12, max_side=448, tensor_parallel_size=1, max_model_len=12288),
    # --- RL-tuned video reasoners (Qwen2.5-VL base; 7B -> single GPU, no NCCL) ---
    "video-r1-7b": dict(model_id="Video-R1/Video-R1-7B", version="video-r1-7b",
                        num_frames=16, max_side=448, tensor_parallel_size=1, max_tokens=2048),
    "videochat-r1.5-7b": dict(model_id="OpenGVLab/VideoChat-R1_5-7B", version="videochat-r1.5-7b",
                              num_frames=16, max_side=448, tensor_parallel_size=1, max_tokens=2048),
    "videochat-r1-think-7b": dict(model_id="OpenGVLab/VideoChat-R1-thinking_7B", version="videochat-r1-think-7b",
                                  num_frames=16, max_side=448, tensor_parallel_size=1, max_tokens=2048),
    "qwen2.5-vl-32b": dict(model_id="Qwen/Qwen2.5-VL-32B-Instruct", version="qwen2.5-vl-32b",
                           num_frames=24, max_side=448, tensor_parallel_size=2),
    "qwen2.5-vl-32b-awq": dict(model_id="Qwen/Qwen2.5-VL-32B-Instruct-AWQ", version="qwen2.5-vl-32b-awq",
                               num_frames=16, max_side=448, tensor_parallel_size=1, quantization="awq",
                               max_model_len=12288, gpu_memory_utilization=0.92),
    "qwen2.5-vl-72b": dict(model_id="Qwen/Qwen2.5-VL-72B-Instruct", version="qwen2.5-vl-72b",
                           num_frames=32, max_side=448, tensor_parallel_size=4),
    "qwen2.5-vl-72b-awq": dict(model_id="Qwen/Qwen2.5-VL-72B-Instruct-AWQ", version="qwen2.5-vl-72b-awq",
                               num_frames=32, max_side=448, tensor_parallel_size=2, quantization="awq"),
    "internvl3-8b": dict(model_id="OpenGVLab/InternVL3-8B", version="internvl3-8b",
                         num_frames=16, max_side=448, tensor_parallel_size=1),
    "internvl3-14b": dict(model_id="OpenGVLab/InternVL3-14B", version="internvl3-14b",
                          num_frames=16, max_side=448, tensor_parallel_size=2),
    "internvl3-38b": dict(model_id="OpenGVLab/InternVL3-38B", version="internvl3-38b",
                          num_frames=24, max_side=448, tensor_parallel_size=2,
                          gpu_memory_utilization=0.92),
    "internvl3-78b-awq": dict(model_id="OpenGVLab/InternVL3-78B-AWQ", version="internvl3-78b-awq",
                              num_frames=24, max_side=448, tensor_parallel_size=2, quantization="awq"),
    "internvl3-78b": dict(model_id="OpenGVLab/InternVL3-78B", version="internvl3-78b",
                          num_frames=24, max_side=448, tensor_parallel_size=4),
}


def create_model(name: str, **kwargs) -> AbstractVideoQAModel:
    if name not in _PRESETS:
        raise KeyError(f"unknown model '{name}'. available: {sorted(_PRESETS)}")
    cfg = dict(_PRESETS[name])
    cfg.update({k: v for k, v in kwargs.items() if v is not None})
    return VLLMVideoQAModel(**cfg)


def register_model(name: str, defaults: dict) -> None:
    _PRESETS[name] = defaults


def list_models():
    return sorted(_PRESETS)

"""Frame extraction with on-disk caching (data-layer concern only)."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
import av

from ..config import FRAME_CACHE_DIR
from ..utils.logutil import get_logger

log = get_logger("vrrqa.frames")


def _cache_key(clip_path: str, num_frames: int, max_side: int) -> Path:
    stem = Path(clip_path).stem
    h = hashlib.md5(f"{clip_path}|{num_frames}|{max_side}".encode()).hexdigest()[:8]
    return FRAME_CACHE_DIR / f"{stem}__nf{num_frames}_ms{max_side}_{h}.npz"


def _resize(img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return img
    w, h = img.size
    s = max_side / max(w, h)
    if s < 1.0:
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
    return img


def decode_uniform(clip_path: str, num_frames: int, max_side: int) -> List[Image.Image]:
    """Uniformly sample `num_frames` frames across the whole clip using PyAV."""
    container = av.open(clip_path)
    stream = container.streams.video[0]
    total = stream.frames
    frames: List[Image.Image] = []
    try:
        if total and total > 0:
            idxs = set(np.linspace(0, total - 1, num_frames).astype(int).tolist())
            for i, frame in enumerate(container.decode(video=0)):
                if i in idxs:
                    frames.append(_resize(frame.to_image(), max_side))
                if len(frames) >= num_frames:
                    break
        else:  # no frame count metadata -> decode all then subsample
            all_f = [f.to_image() for f in container.decode(video=0)]
            if all_f:
                idxs = np.linspace(0, len(all_f) - 1, num_frames).astype(int)
                frames = [_resize(all_f[j], max_side) for j in idxs]
    finally:
        container.close()

    if not frames:
        raise RuntimeError(f"no frames decoded from {clip_path}")
    # pad by repeating the last frame if the clip was too short
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]


def load_frames_cached(clip_path: str, num_frames: int, max_side: int, use_cache: bool = True) -> List[Image.Image]:
    key = _cache_key(clip_path, num_frames, max_side)
    if use_cache and key.exists():
        try:
            arr = np.load(key)["frames"]
            return [Image.fromarray(a) for a in arr]
        except Exception:
            pass
    frames = decode_uniform(clip_path, num_frames, max_side)
    if use_cache:
        try:
            np.savez_compressed(key, frames=np.stack([np.asarray(f) for f in frames]))
        except Exception as e:
            log.warning("frame cache write failed for %s: %s", clip_path, e)
    return frames

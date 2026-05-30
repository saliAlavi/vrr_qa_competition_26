"""Factory for datasets — register new splits/datasets here without touching callers."""
from __future__ import annotations
from typing import Callable, Dict
from .base import AbstractDataset
from .implicitqa import ImplicitQADataset

_REGISTRY: Dict[str, Callable[..., AbstractDataset]] = {
    "implicitqa_val": lambda **k: ImplicitQADataset("val", **k),
    "implicitqa_test": lambda **k: ImplicitQADataset("test", **k),
    # aliases
    "val": lambda **k: ImplicitQADataset("val", **k),
    "test": lambda **k: ImplicitQADataset("test", **k),
}


def create_dataset(name: str, **kwargs) -> AbstractDataset:
    if name not in _REGISTRY:
        raise KeyError(f"unknown dataset '{name}'. available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def register_dataset(name: str, ctor: Callable[..., AbstractDataset]) -> None:
    _REGISTRY[name] = ctor

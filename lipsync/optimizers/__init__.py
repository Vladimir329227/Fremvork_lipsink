"""Optimizer registry and factory."""
from __future__ import annotations

from typing import Any, Iterable

import torch
from torch.optim import Optimizer

from .adam import AdamW
from .clipping import GradientClippingSGD
from .lion import Lion
from .lookahead import Lookahead
from .schedulers import (
    CosineWarmupScheduler,
    SCHEDULER_REGISTRY,
    WarmupScheduler,
    build_scheduler,
)
from .sgd import MomentumSGD, SGD

OPTIMIZER_REGISTRY: dict[str, type[Optimizer]] = {
    "sgd": SGD,
    "momentum_sgd": MomentumSGD,
    "clipping_sgd": GradientClippingSGD,
    "adamw": AdamW,
    "adam": AdamW,  # alias — AdamW is the preferred Adam variant
    "lion": Lion,
    # PyTorch built-ins exposed for convenience
    "torch_sgd": torch.optim.SGD,
    "torch_adam": torch.optim.Adam,
    "torch_adamw": torch.optim.AdamW,
    "torch_rmsprop": torch.optim.RMSprop,
}


def build_optimizer(
    params: Iterable,
    name: str,
    lookahead: bool = False,
    lookahead_k: int = 5,
    lookahead_alpha: float = 0.5,
    **kwargs: Any,
) -> Optimizer:
    """Instantiate an optimizer by name.

    Args:
        params: Model parameters (``model.parameters()`` or param groups).
        name: Key from OPTIMIZER_REGISTRY (case-insensitive).
        lookahead: Wrap the optimizer in Lookahead.
        lookahead_k: Lookahead step interval.
        lookahead_alpha: Lookahead interpolation coefficient.
        **kwargs: Forwarded to the optimizer constructor.

    Returns:
        Configured optimizer instance.
    """
    key = name.lower()
    if key not in OPTIMIZER_REGISTRY:
        available = ", ".join(sorted(OPTIMIZER_REGISTRY))
        raise KeyError(f"Unknown optimizer '{name}'. Available: {available}")

    opt = OPTIMIZER_REGISTRY[key](params, **kwargs)
    if lookahead:
        opt = Lookahead(opt, k=lookahead_k, alpha=lookahead_alpha)
    return opt


def register_optimizer(name: str, cls: type[Optimizer]) -> None:
    """Register a custom optimizer class."""
    OPTIMIZER_REGISTRY[name.lower()] = cls


__all__ = [
    "SGD",
    "MomentumSGD",
    "GradientClippingSGD",
    "AdamW",
    "Lion",
    "Lookahead",
    "WarmupScheduler",
    "CosineWarmupScheduler",
    "OPTIMIZER_REGISTRY",
    "SCHEDULER_REGISTRY",
    "build_optimizer",
    "build_scheduler",
    "register_optimizer",
]

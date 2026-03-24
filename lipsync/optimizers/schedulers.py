"""Learning rate schedulers: CosineAnnealing, Warmup, ReduceLROnPlateau."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
)

SCHEDULER_REGISTRY: dict[str, Any] = {
    "cosine": CosineAnnealingLR,
    "reduce_on_plateau": ReduceLROnPlateau,
    "warmup_cosine": None,  # filled below
    "constant": None,
}


class WarmupScheduler(LRScheduler):
    """Linear warmup followed by a delegated scheduler.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of linear warm-up steps.
        after_scheduler: Scheduler to use after warm-up completes.
        last_epoch: Starting epoch (−1 = fresh start).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        after_scheduler: LRScheduler | None = None,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.after_scheduler = after_scheduler
        self._warmup_finished = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:  # type: ignore[override]
        if self.last_epoch < self.warmup_steps:
            scale = (self.last_epoch + 1) / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]
        if self.after_scheduler is not None and not self._warmup_finished:
            self._warmup_finished = True
            self.after_scheduler.base_lrs = self.base_lrs  # type: ignore[attr-defined]
        if self.after_scheduler is not None:
            return self.after_scheduler.get_last_lr()
        return self.base_lrs

    def step(self, *args, **kwargs) -> None:
        if self._warmup_finished and self.after_scheduler is not None:
            self.after_scheduler.step(*args, **kwargs)
        else:
            super().step()


class CosineWarmupScheduler(WarmupScheduler):
    """Cosine annealing with linear warmup — the most common recipe.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Linear warm-up steps.
        T_max: Total cosine annealing steps (including warm-up).
        eta_min: Minimum learning rate.
        last_epoch: Starting epoch index.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        cosine = CosineAnnealingLR(optimizer, T_max=T_max - warmup_steps, eta_min=eta_min)
        super().__init__(optimizer, warmup_steps, after_scheduler=cosine, last_epoch=last_epoch)


SCHEDULER_REGISTRY["warmup_cosine"] = CosineWarmupScheduler
SCHEDULER_REGISTRY["constant"] = lambda opt, **_: torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)


def build_scheduler(name: str, optimizer: Optimizer, **kwargs: Any) -> Any:
    """Instantiate a scheduler by name.

    Args:
        name: Key from SCHEDULER_REGISTRY (case-insensitive).
        optimizer: Optimizer to wrap.
        **kwargs: Constructor keyword arguments.
    """
    key = name.lower()
    if key not in SCHEDULER_REGISTRY:
        available = ", ".join(sorted(SCHEDULER_REGISTRY))
        raise KeyError(f"Unknown scheduler '{name}'. Available: {available}")
    cls = SCHEDULER_REGISTRY[key]
    return cls(optimizer, **kwargs)

"""Lookahead wrapper optimizer (Zhang et al., 2019)."""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import torch
from torch.optim import Optimizer


class Lookahead(Optimizer):
    """Lookahead wrapper that improves convergence stability.

    Maintains two sets of weights: fast weights (updated by the base optimizer)
    and slow weights (updated by interpolating toward fast weights every k steps).

    Reference: "Lookahead Optimizer: k steps forward, 1 step back"
    (Zhang et al., 2019) — https://arxiv.org/abs/1907.08610

    Args:
        base_optimizer: Any PyTorch optimizer instance.
        k: Number of fast update steps per slow update.
        alpha: Slow weight interpolation coefficient (0 < α ≤ 1).
    """

    def __init__(self, base_optimizer: Optimizer, k: int = 5, alpha: float = 0.5) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"Invalid slow update coefficient: {alpha}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self._step_counter = 0
        self._slow_weights: dict = defaultdict(dict)

        # Initialise slow weights from fast weights
        for group in base_optimizer.param_groups:
            for p in group["params"]:
                self._slow_weights[p]["slow"] = p.data.clone()

        # Expose param_groups so schedulers and other code can access them
        self.param_groups = base_optimizer.param_groups
        self.state = base_optimizer.state
        self.defaults = base_optimizer.defaults

    def step(self, closure: Callable | None = None):
        loss = self.base_optimizer.step(closure)
        self._step_counter += 1

        if self._step_counter >= self.k:
            self._step_counter = 0
            for group in self.base_optimizer.param_groups:
                for p in group["params"]:
                    slow = self._slow_weights[p]["slow"]
                    # Interpolate: slow ← slow + α * (fast - slow)
                    slow.add_(p.data - slow, alpha=self.alpha)
                    p.data.copy_(slow)

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "slow_weights": {
                id(p): v for p, v in self._slow_weights.items()
            },
            "step_counter": self._step_counter,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self._step_counter = state_dict["step_counter"]
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                pid = id(p)
                if pid in state_dict["slow_weights"]:
                    self._slow_weights[p]["slow"] = state_dict["slow_weights"][pid]["slow"]

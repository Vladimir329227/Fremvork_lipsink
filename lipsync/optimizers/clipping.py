"""Gradient-clipping SGD optimizer."""
from __future__ import annotations

from typing import Callable, Iterable, Literal

import torch
import torch.nn.utils as nn_utils
from torch.optim import Optimizer


class GradientClippingSGD(Optimizer):
    """SGD with built-in gradient clipping applied before each weight update.

    Supports two clipping strategies:
    - ``norm``: clips the global gradient norm (torch.nn.utils.clip_grad_norm_).
    - ``value``: clips each gradient element individually (clip_grad_value_).

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate.
        momentum: Momentum factor (0 = vanilla SGD).
        nesterov: Use Nesterov update.
        weight_decay: L2 penalty.
        clip_mode: 'norm' or 'value'.
        max_norm: Maximum gradient norm (used when clip_mode='norm').
        clip_value: Maximum absolute gradient value (used when clip_mode='value').
        norm_type: Order of the norm (default 2.0).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        clip_mode: Literal["norm", "value"] = "norm",
        max_norm: float = 1.0,
        clip_value: float = 1.0,
        norm_type: float = 2.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            clip_mode=clip_mode,
            max_norm=max_norm,
            clip_value=clip_value,
            norm_type=norm_type,
        )
        super().__init__(params, defaults)

    def _clip_gradients(self) -> float:
        """Clip gradients across all param groups and return the pre-clip norm."""
        all_params = [
            p
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        group = self.param_groups[0]
        if group["clip_mode"] == "norm":
            total_norm = nn_utils.clip_grad_norm_(
                all_params, group["max_norm"], norm_type=group["norm_type"]
            )
            return float(total_norm)
        else:
            nn_utils.clip_grad_value_(all_params, group["clip_value"])
            return -1.0

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._clip_gradients()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                if mu > 0:
                    state = self.state[p]
                    if "velocity" not in state:
                        state["velocity"] = torch.clone(grad).detach()
                    else:
                        state["velocity"].mul_(mu).add_(grad)
                    v = state["velocity"]
                    effective_grad = grad.add(v, alpha=mu) if nesterov else v
                else:
                    effective_grad = grad

                p.add_(effective_grad, alpha=-lr)

        return loss

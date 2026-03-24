"""Lion optimizer — Evolved Sign Momentum (Chen et al., 2023)."""
from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    """Lion: Evolved Sign Momentum optimizer.

    Uses the sign of the update direction, making it memory-efficient
    compared to Adam. Particularly effective for large batch sizes and
    Transformer architectures.

    Reference: "Symbolic Discovery of Optimization Algorithms"
    (Chen et al., 2023) — https://arxiv.org/abs/2302.06675

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate (typically 3–10× smaller than Adam).
        betas: Coefficients (β₁, β₂) — β₁ for update interpolation,
               β₂ for EMA of the gradient.
        weight_decay: Decoupled weight decay.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)

                m = state["exp_avg"]

                # Decoupled weight decay
                if wd != 0:
                    p.mul_(1 - lr * wd)

                # Compute update: sign(β₁·m + (1-β₁)·g)
                update = m.mul(beta1).add_(grad, alpha=1 - beta1)
                p.add_(update.sign_(), alpha=-lr)

                # Update EMA: m = β₂·m + (1-β₂)·g
                m.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

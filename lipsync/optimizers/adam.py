"""AdamW optimizer with decoupled weight decay."""
from __future__ import annotations

import math
from typing import Callable, Iterable

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """Adam optimiser with decoupled weight decay (AdamW).

    Implements the algorithm from "Decoupled Weight Decay Regularization"
    (Loshchilov & Hutter, 2019).

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate α.
        betas: Coefficients (β₁, β₂) for computing running averages of gradient
               and its square.
        eps: Term added to denominator for numerical stability.
        weight_decay: Decoupled weight decay λ.
        amsgrad: Use the AMSGrad variant of the algorithm.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
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
            eps = group["eps"]
            wd = group["weight_decay"]
            amsgrad = group["amsgrad"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_c1 = 1 - beta1 ** t
                bias_c2 = 1 - beta2 ** t
                step_size = lr * math.sqrt(bias_c2) / bias_c1

                if amsgrad:
                    max_v = state["max_exp_avg_sq"]
                    torch.maximum(max_v, v, out=max_v)
                    denom = max_v.sqrt().add_(eps)
                else:
                    denom = v.sqrt().add_(eps)

                # Decoupled weight decay applied directly to parameter
                if wd != 0:
                    p.mul_(1 - lr * wd)

                p.addcdiv_(m, denom, value=-step_size)

        return loss

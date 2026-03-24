"""SGD and Momentum SGD optimizers."""
from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch.optim import Optimizer


class SGD(Optimizer):
    """Vanilla Stochastic Gradient Descent.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate.
        weight_decay: L2 regularisation coefficient.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)
                p.add_(grad, alpha=-lr)

        return loss


class MomentumSGD(Optimizer):
    """SGD with momentum (and optional Nesterov accelerated gradient).

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate.
        momentum: Momentum factor μ (0 disables momentum).
        nesterov: Use Nesterov update rule.
        dampening: Dampening for momentum.
        weight_decay: L2 regularisation coefficient.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        momentum: float = 0.9,
        nesterov: bool = False,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov requires momentum > 0 and dampening == 0")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            dampening=dampening,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]
                if "velocity" not in state:
                    state["velocity"] = torch.clone(grad).detach()
                else:
                    v = state["velocity"]
                    v.mul_(mu).add_(grad, alpha=1.0 - dampening)

                v = state["velocity"]
                if nesterov:
                    effective_grad = grad.add(v, alpha=mu)
                else:
                    effective_grad = v

                p.add_(effective_grad, alpha=-lr)

        return loss

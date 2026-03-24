"""Activation function registry with build_activation factory."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    """Mish: A Self Regularized Non-Monotonic Activation Function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class Swish(nn.Module):
    """Swish / SiLU activation: x * sigmoid(x)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class LogSoftmax(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(x, dim=self.dim)


ACTIVATION_REGISTRY: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": Swish,
    "mish": Mish,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softmax": nn.Softmax,
    "log_softmax": LogSoftmax,
    "elu": nn.ELU,
    "prelu": nn.PReLU,
    "hardswish": nn.Hardswish,
}


def build_activation(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate an activation function by name.

    Args:
        name: Key from ACTIVATION_REGISTRY (case-insensitive).
        **kwargs: Forwarded to the constructor (e.g. negative_slope for LeakyReLU).

    Returns:
        Instantiated nn.Module.

    Raises:
        KeyError: If *name* is not registered.
    """
    name = name.lower()
    if name not in ACTIVATION_REGISTRY:
        available = ", ".join(sorted(ACTIVATION_REGISTRY))
        raise KeyError(
            f"Unknown activation '{name}'. Available: {available}"
        )
    return ACTIVATION_REGISTRY[name](**kwargs)


def list_activations() -> list[str]:
    """Return sorted list of registered activation names."""
    return sorted(ACTIVATION_REGISTRY)


def register_activation(name: str, cls: type[nn.Module]) -> None:
    """Register a custom activation class under *name*."""
    ACTIVATION_REGISTRY[name.lower()] = cls

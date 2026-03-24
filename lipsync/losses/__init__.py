"""Loss function registry and factory."""
from __future__ import annotations

from typing import Any

from .classification import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    FocalLoss,
    LabelSmoothingLoss,
)
from .lipsync_losses import (
    AdversarialLoss,
    IdentityLoss,
    PerceptualLoss,
    SyncLoss,
    TemporalConsistencyLoss,
)
from .regression import HuberLoss, LogCoshLoss, MAELoss, MSELoss

import torch.nn as nn

LOSS_REGISTRY: dict[str, type[nn.Module]] = {
    # regression
    "mse": MSELoss,
    "mae": MAELoss,
    "l1": MAELoss,
    "huber": HuberLoss,
    "log_cosh": LogCoshLoss,
    # classification
    "bce": BCEWithLogitsLoss,
    "cross_entropy": CrossEntropyLoss,
    "focal": FocalLoss,
    "label_smoothing": LabelSmoothingLoss,
    # lip-sync
    "perceptual": PerceptualLoss,
    "sync": SyncLoss,
    "temporal": TemporalConsistencyLoss,
    "identity": IdentityLoss,
    "adversarial": AdversarialLoss,
}


def build_loss(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a loss function by name.

    Args:
        name: Key from LOSS_REGISTRY (case-insensitive).
        **kwargs: Forwarded to the constructor.
    """
    key = name.lower()
    if key not in LOSS_REGISTRY:
        available = ", ".join(sorted(LOSS_REGISTRY))
        raise KeyError(f"Unknown loss '{name}'. Available: {available}")
    return LOSS_REGISTRY[key](**kwargs)


def register_loss(name: str, cls: type[nn.Module]) -> None:
    """Register a custom loss class."""
    LOSS_REGISTRY[name.lower()] = cls


__all__ = [
    "MSELoss",
    "MAELoss",
    "HuberLoss",
    "LogCoshLoss",
    "BCEWithLogitsLoss",
    "CrossEntropyLoss",
    "FocalLoss",
    "LabelSmoothingLoss",
    "PerceptualLoss",
    "SyncLoss",
    "TemporalConsistencyLoss",
    "IdentityLoss",
    "AdversarialLoss",
    "LOSS_REGISTRY",
    "build_loss",
    "register_loss",
]

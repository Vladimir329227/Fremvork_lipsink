"""Regression loss functions: MSE, MAE, Huber, LogCosh."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """Mean Squared Error — sensitive to outliers, classic choice."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target, reduction=self.reduction)


class MAELoss(nn.Module):
    """Mean Absolute Error — robust to outliers."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target, reduction=self.reduction)


class HuberLoss(nn.Module):
    """Huber loss — MSE for small errors, MAE for large ones.

    Args:
        delta: Threshold between quadratic and linear regions.
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(pred, target, delta=self.delta, reduction=self.reduction)


class LogCoshLoss(nn.Module):
    """Log-Cosh loss — ultra-smooth approximation of MAE.

    log(cosh(pred - target)) ≈ (pred-target)²/2  for small errors
                               |pred-target| - log2  for large errors
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = diff + F.softplus(-2.0 * diff) - torch.log(torch.tensor(2.0, device=diff.device))
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

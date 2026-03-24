"""Classification loss functions: BCE, CrossEntropy, Focal, LabelSmoothing."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsLoss(nn.Module):
    """Binary Cross-Entropy with logits (numerically stable)."""

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        pos_weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(
            weight=weight, pos_weight=pos_weight, reduction=reduction
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


class CrossEntropyLoss(nn.Module):
    """Categorical cross-entropy with optional class weights."""

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, reduction=reduction
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for the rare class (0–1).
        gamma: Focusing parameter; 0 reduces to BCE.
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        focal = self.alpha * (1.0 - p_t) ** self.gamma * bce

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing for regularisation.

    Args:
        num_classes: Total number of output classes.
        smoothing: Smoothing factor ε (0 = standard CE).
        reduction: 'mean' | 'sum'.
    """

    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)

        # Build smoothed target distribution
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(-1, targets.unsqueeze(-1), self.confidence)

        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()

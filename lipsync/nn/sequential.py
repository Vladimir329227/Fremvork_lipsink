"""Sequential model — build a neural network by listing layers.

Example::

    from lipsync.nn import Sequential, Linear, Conv2d, Flatten, Dropout

    model = Sequential([
        Linear(784, 256, activation="relu"),
        Dropout(0.3),
        Linear(256, 128, activation="relu"),
        Linear(128, 10),
    ])

    model.compile(optimizer="adamw", loss="cross_entropy", lr=1e-3)
    model.fit(X_train, y_train, epochs=20, batch_size=64, val_data=(X_val, y_val))
    acc = model.evaluate(X_test, y_test)
    preds = model.predict(X_test)
"""
from __future__ import annotations

import time
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..activations import build_activation
from ..losses import build_loss
from ..optimizers import build_optimizer, build_scheduler


# ---------------------------------------------------------------------------
# Layer helpers — thin wrappers that accept activation by name
# ---------------------------------------------------------------------------

class Linear(nn.Module):
    """Fully-connected layer with optional activation.

    Args:
        in_features: Input size.
        out_features: Output size.
        activation: Activation name (e.g. 'relu', 'gelu') or None.
        bias: Whether to add a bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.act = build_activation(activation) if activation else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return self.act(x) if self.act else x


class Conv2d(nn.Module):
    """Convolutional layer with optional activation and batch-norm.

    Args:
        in_channels, out_channels, kernel_size: Standard conv params.
        stride, padding: Conv geometry.
        activation: Activation name or None.
        batch_norm: Apply BatchNorm2d after conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        activation: str | None = "relu",
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=not batch_norm)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.act = build_activation(activation) if activation else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        return self.act(x) if self.act else x


class Flatten(nn.Module):
    """Flatten all dimensions except batch."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Dropout(nn.Dropout):
    """Dropout wrapper — accepts p as positional arg."""


class MaxPool2d(nn.MaxPool2d):
    """Max-pooling wrapper."""


class BatchNorm1d(nn.BatchNorm1d):
    """1-D batch normalisation wrapper."""


class LayerNorm(nn.LayerNorm):
    """Layer normalisation wrapper."""


# ---------------------------------------------------------------------------
# Sequential model
# ---------------------------------------------------------------------------

class Sequential(nn.Module):
    """Build and train a neural network by listing layers.

    Layers can be added at construction or via :meth:`add`.
    Call :meth:`compile` to configure optimiser / loss, then :meth:`fit`.

    Args:
        layers: Optional initial list of ``nn.Module`` layers.
    """

    def __init__(self, layers: list[nn.Module] | None = None) -> None:
        super().__init__()
        self._layers = nn.ModuleList(layers or [])
        self._opt: Any = None
        self._loss_fn: nn.Module | None = None
        self._loss_name: str = "mse"
        self._device: torch.device = torch.device("cpu")
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add(self, layer: nn.Module) -> "Sequential":
        """Append a layer and return self for chaining."""
        self._layers.append(layer)
        return self

    def __repr__(self) -> str:
        lines = ["Sequential("]
        for i, layer in enumerate(self._layers):
            lines.append(f"  ({i}) {layer}")
        lines.append(")")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._layers:
            x = layer(x)
        return x

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    def compile(
        self,
        optimizer: str = "adamw",
        loss: str = "cross_entropy",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_kwargs: dict | None = None,
        scheduler: str | None = None,
        scheduler_kwargs: dict | None = None,
        device: str = "auto",
        **opt_kwargs: Any,
    ) -> "Sequential":
        """Configure optimiser, loss function and device.

        Args:
            optimizer: Name from OPTIMIZER_REGISTRY
                       ('sgd', 'momentum_sgd', 'clipping_sgd', 'adamw', 'lion', …).
            loss: Name from LOSS_REGISTRY
                  ('cross_entropy', 'bce', 'focal', 'mse', 'huber', …).
            lr: Learning rate.
            weight_decay: L2 regularisation.
            scheduler: Optional scheduler name ('cosine', 'warmup_cosine', …).
            scheduler_kwargs: Forwarded to build_scheduler.
            device: 'cpu' | 'cuda' | 'auto'.
            **opt_kwargs: Extra keyword args forwarded to the optimizer.
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self.to(self._device)

        self._opt = build_optimizer(
            self.parameters(), optimizer,
            lr=lr, weight_decay=weight_decay, **opt_kwargs
        )
        self._loss_name = loss
        self._loss_fn = build_loss(loss, **(loss_kwargs or {}))
        self._scheduler = (
            build_scheduler(scheduler, self._opt, **(scheduler_kwargs or {}))
            if scheduler else None
        )
        return self

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 10,
        batch_size: int = 32,
        val_data: tuple[torch.Tensor, torch.Tensor] | None = None,
        shuffle: bool = True,
        verbose: bool = True,
    ) -> list[dict]:
        """Train the model.

        Args:
            X: Input tensor (N, ...).
            y: Target tensor (N,) for classification or (N, ...) for regression.
            epochs: Number of full passes over the data.
            batch_size: Mini-batch size.
            val_data: Optional (X_val, y_val) for validation metrics.
            shuffle: Shuffle data each epoch.
            verbose: Print epoch metrics.

        Returns:
            Training history list of per-epoch metric dicts.
        """
        if self._opt is None:
            raise RuntimeError("Call .compile() before .fit()")

        X = X.to(self._device)
        y = y.to(self._device)
        loader = DataLoader(
            TensorDataset(X, y),
            batch_size=batch_size,
            shuffle=shuffle,
        )

        self._history = []
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            self.train()
            total_loss = 0.0

            for xb, yb in loader:
                self._opt.zero_grad()
                pred = self(xb)
                loss = self._loss_fn(pred, yb)
                loss.backward()
                self._opt.step()
                total_loss += loss.item()

            if self._scheduler:
                self._scheduler.step()

            logs: dict[str, float] = {
                "epoch": epoch,
                "loss": total_loss / len(loader),
                "time_s": round(time.perf_counter() - t0, 3),
            }

            if val_data is not None:
                xv, yv = val_data[0].to(self._device), val_data[1].to(self._device)
                val_metrics = self._compute_metrics(xv, yv)
                logs.update({f"val_{k}": v for k, v in val_metrics.items()})

            self._history.append(logs)

            if verbose:
                parts = [f"Epoch {epoch:3d}/{epochs}"]
                parts += [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in logs.items() if k not in ("epoch",)]
                print("  ".join(parts))

        return self._history

    # ------------------------------------------------------------------
    # Evaluation and prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_metrics(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> dict[str, float]:
        self.eval()
        pred = self(X)
        loss = self._loss_fn(pred, y).item()
        metrics = {"loss": loss}

        # Accuracy for classification losses
        if self._loss_name in ("cross_entropy", "label_smoothing"):
            acc = (pred.argmax(dim=-1) == y).float().mean().item()
            metrics["acc"] = acc
        elif self._loss_name in ("bce", "focal"):
            acc = ((pred.sigmoid() > 0.5).float() == y).float().mean().item()
            metrics["acc"] = acc

        return metrics

    @torch.no_grad()
    def evaluate(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> dict[str, float]:
        """Compute loss (and accuracy for classification) on given data."""
        if self._loss_fn is None:
            raise RuntimeError("Call .compile() before .evaluate()")
        return self._compute_metrics(X.to(self._device), y.to(self._device))

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return predictions (moved to CPU)."""
        self.eval()
        return self(X.to(self._device)).cpu()

    @torch.no_grad()
    def predict_classes(self, X: torch.Tensor) -> torch.Tensor:
        """Return argmax class predictions."""
        return self.predict(X).argmax(dim=-1)

    @property
    def history(self) -> list[dict]:
        """Per-epoch training metrics from last .fit() call."""
        return self._history

"""Training callbacks for the Trainer class."""
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .trainer import Trainer


class Callback:
    """Base callback class with no-op implementations."""

    def on_train_begin(self, trainer: "Trainer") -> None: ...
    def on_train_end(self, trainer: "Trainer") -> None: ...
    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None: ...
    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: dict) -> None: ...
    def on_batch_begin(self, trainer: "Trainer", batch: int) -> None: ...
    def on_batch_end(self, trainer: "Trainer", batch: int, logs: dict) -> None: ...


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Args:
        monitor: Metric name to watch (e.g. 'val_loss').
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as improvement.
        mode: 'min' or 'max'.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best: float = float("inf") if mode == "min" else float("-inf")
        self._wait = 0

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: dict) -> None:
        current = logs.get(self.monitor)
        if current is None:
            return

        improved = (
            (current < self._best - self.min_delta)
            if self.mode == "min"
            else (current > self._best + self.min_delta)
        )
        if improved:
            self._best = current
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                print(
                    f"[EarlyStopping] No improvement in '{self.monitor}' for "
                    f"{self.patience} epochs. Stopping."
                )
                trainer.should_stop = True


class ModelCheckpoint(Callback):
    """Save model weights when a monitored metric improves.

    Args:
        save_dir: Directory for checkpoint files.
        monitor: Metric to monitor.
        mode: 'min' or 'max'.
        save_best_only: If True, only overwrite when improved.
        filename: Template, supports {epoch} and {monitor}.
    """

    def __init__(
        self,
        save_dir: str | Path = "checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        filename: str = "best_model.pt",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.filename = filename
        self._best = float("inf") if mode == "min" else float("-inf")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: dict) -> None:
        current = logs.get(self.monitor)
        improved = current is None  # save unconditionally if no metric
        if current is not None:
            improved = (
                (current < self._best)
                if self.mode == "min"
                else (current > self._best)
            )
        if not self.save_best_only or improved:
            if current is not None:
                self._best = current
            path = self.save_dir / self.filename.format(epoch=epoch, monitor=current)
            trainer.save_checkpoint(path)
            print(f"[Checkpoint] Saved -> {path}")


class LRSchedulerCallback(Callback):
    """Step a learning-rate scheduler at the end of each epoch."""

    def __init__(self, scheduler: Any, monitor: str | None = None) -> None:
        self.scheduler = scheduler
        self.monitor = monitor

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: dict) -> None:
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        if isinstance(self.scheduler, ReduceLROnPlateau):
            val = logs.get(self.monitor or "val_loss")
            if val is not None:
                self.scheduler.step(val)
        else:
            self.scheduler.step()


class WandbLogger(Callback):
    """Log metrics and model gradients to Weights & Biases."""

    def __init__(self, project: str = "lipsync", **wandb_kwargs: Any) -> None:
        self.project = project
        self.wandb_kwargs = wandb_kwargs
        self._run = None

    def on_train_begin(self, trainer: "Trainer") -> None:
        try:
            import wandb

            self._run = wandb.init(project=self.project, **self.wandb_kwargs)
        except ImportError:
            print("[WandbLogger] wandb not installed — logging disabled.")

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: dict) -> None:
        if self._run is not None:
            import wandb

            wandb.log({"epoch": epoch, **logs})

    def on_train_end(self, trainer: "Trainer") -> None:
        if self._run is not None:
            import wandb

            wandb.finish()


class ProgressBar(Callback):
    """Simple console progress bar without extra dependencies."""

    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        self._epoch_start = time.time()
        self._epoch = epoch
        print(f"[Epoch {epoch}] training...", flush=True)

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: dict) -> None:
        elapsed = time.time() - self._epoch_start
        metrics = "  ".join(f"{k}={v:.4f}" for k, v in logs.items() if isinstance(v, float))
        print(f"Epoch {epoch:4d} | {elapsed:6.1f}s | {metrics}")

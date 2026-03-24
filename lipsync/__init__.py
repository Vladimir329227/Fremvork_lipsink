"""LipSync Framework — high-level public API.

Quick start::

    from lipsync import LipSyncTrainer, LipSyncConfig

    # Minimal: train from config file
    trainer = LipSyncTrainer.from_config("configs/base.yaml")
    trainer.fit(train_dataset, val_dataset)

    # Run inference
    result = trainer.predict(audio="speech.wav", video="face.mp4")
    result.save("output.mp4")
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from torch.utils.data import Dataset

from .nn import (
    BatchNorm1d,
    Conv2d,
    Dataset,
    Dropout,
    Flatten,
    LayerNorm,
    Linear,
    MaxPool2d,
    Sequential,
)
from .training.callbacks import (
    Callback,
    EarlyStopping,
    LRSchedulerCallback,
    ModelCheckpoint,
    ProgressBar,
    WandbLogger,
)
from .training.trainer import LipSyncTrainerCore


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class LipSyncConfig:
    """Typed configuration for the LipSync framework.

    All fields correspond to keys in the YAML config files.
    Keyword arguments passed here override the loaded YAML.

    Args:
        model: Model architecture hyper-parameters.
        optimizer: Optimizer name and kwargs.
        scheduler: LR scheduler name and kwargs.
        losses: Loss weight dict.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        fp16: Enable automatic mixed precision.
        gradient_checkpointing: Gradient checkpointing for memory.
        gradient_clipping: Gradient clipping config dict.
        use_super_resolution: Enable SR post-processing.
        sr_backend: SR model ('gfpgan' | 'codeformer' | 'realesrgan').
        checkpoint_dir: Directory for saving checkpoints.
        log_wandb: Enable Weights & Biases logging.
        wandb_project: W&B project name.
    """

    model: dict[str, Any] = field(default_factory=dict)
    optimizer: dict[str, Any] = field(default_factory=lambda: {"name": "adamw", "lr": 2e-4})
    scheduler: dict[str, Any] = field(default_factory=dict)
    losses: dict[str, Any] = field(default_factory=dict)
    epochs: int = 100
    batch_size: int = 8
    fp16: bool = True
    gradient_checkpointing: bool = False
    gradient_clipping: dict[str, Any] = field(default_factory=lambda: {"enabled": True, "max_norm": 1.0})
    use_super_resolution: bool = False
    sr_backend: str = "gfpgan"
    checkpoint_dir: str = "checkpoints"
    log_wandb: bool = False
    wandb_project: str = "lipsync"

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LipSyncConfig":
        import dataclasses
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LipSyncConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)


# ---------------------------------------------------------------------------
# Inference result
# ---------------------------------------------------------------------------

class InferenceResult:
    """Wraps the output of :meth:`LipSyncTrainer.predict`."""

    def __init__(self, frames_bgr: list, fps: float = 25.0) -> None:
        self.frames = frames_bgr
        self.fps = fps

    def save(self, path: str | Path, fps: float | None = None) -> Path:
        """Write frames to an MP4 file.

        Args:
            path: Output file path.
            fps: Override frame rate.

        Returns:
            Path to the written file.
        """
        import cv2

        path = Path(path)
        fps = fps or self.fps
        if not self.frames:
            raise ValueError("No frames to save")
        H, W = self.frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H),
        )
        for f in self.frames:
            writer.write(f)
        writer.release()
        return path

    def __len__(self) -> int:
        return len(self.frames)


# ---------------------------------------------------------------------------
# Main high-level trainer
# ---------------------------------------------------------------------------

class LipSyncTrainer:
    """High-level trainer — train a lip-sync model in a few lines.

    Examples::

        # --- Minimal usage ---
        trainer = LipSyncTrainer.from_config("configs/base.yaml")
        trainer.fit(train_dataset, val_dataset)

        # --- Full control ---
        from lipsync import LipSyncTrainer, LipSyncConfig

        config = LipSyncConfig(
            optimizer={"name": "adamw", "lr": 2e-4, "weight_decay": 1e-4},
            losses={"w_sync": 0.8, "w_recon": 10.0},
            epochs=200,
            fp16=True,
            use_super_resolution=True,
        )
        trainer = LipSyncTrainer(config)
        trainer.fit(train_ds, val_ds, callbacks=[EarlyStopping(patience=15)])

        # --- Inference ---
        result = trainer.predict(audio="voice.wav", video="face.mp4")
        result.save("output.mp4")
    """

    def __init__(
        self,
        config: LipSyncConfig | None = None,
        device: str = "auto",
    ) -> None:
        self.config = config or LipSyncConfig()
        self._core = LipSyncTrainerCore(self.config.to_dict(), device=device)
        self.device = self._core.device

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        device: str = "auto",
        **overrides: Any,
    ) -> "LipSyncTrainer":
        """Load configuration from a YAML file with optional overrides.

        Args:
            config_path: Path to YAML config.
            device: Compute device.
            **overrides: Any LipSyncConfig field to override.

        Returns:
            Configured LipSyncTrainer.
        """
        cfg = LipSyncConfig.from_yaml(config_path)
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cls(cfg, device=device)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, device: str = "auto"
    ) -> "LipSyncTrainer":
        """Resume training or run inference from a saved checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
            device: Compute device.
        """
        import torch

        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        cfg_dict = ckpt.get("config", {})
        cfg = LipSyncConfig.from_dict(cfg_dict)
        trainer = cls(cfg, device=device)
        trainer._core.load_checkpoint(checkpoint_path)
        return trainer

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        callbacks: list[Callback] | None = None,
        num_workers: int = 4,
    ) -> "LipSyncTrainer":
        """Train the model.

        Args:
            train_dataset: Training data.
            val_dataset: Optional validation data.
            epochs: Overrides config value.
            batch_size: Overrides config value.
            callbacks: Additional callbacks (EarlyStopping, WandbLogger, etc.).
            num_workers: DataLoader workers.

        Returns:
            self (for method chaining).
        """
        cbs: list[Callback] = []

        if self.config.log_wandb:
            cbs.append(WandbLogger(project=self.config.wandb_project))

        cbs.append(
            ModelCheckpoint(
                save_dir=self.config.checkpoint_dir,
                monitor="val_loss" if val_dataset else "g_total",
                mode="min",
            )
        )
        cbs.append(ProgressBar())

        if callbacks:
            cbs.extend(callbacks)

        self._core.fit(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=epochs or self.config.epochs,
            batch_size=batch_size or self.config.batch_size,
            callbacks=cbs,
            num_workers=num_workers,
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        audio: str | Path,
        video: str | Path,
        output_path: str | Path | None = None,
        use_sr: bool | None = None,
    ) -> InferenceResult:
        """Run lip-sync inference on a video with the given audio.

        Args:
            audio: Path to driving audio (WAV).
            video: Path to source video.
            output_path: If given, also saves result to this path.
            use_sr: Override config super-resolution setting.

        Returns:
            InferenceResult with processed frames.
        """
        from .inference.batch.processor import BatchProcessor
        import tempfile

        sr = use_sr if use_sr is not None else self.config.use_super_resolution

        with tempfile.TemporaryDirectory() as tmp:
            tmp_out = Path(tmp) / "result.mp4"
            proc = BatchProcessor(
                checkpoint_path=Path(self.config.checkpoint_dir) / "best_model.pt",
                device=str(self.device),
                use_sr=sr,
                sr_backend=self.config.sr_backend,
            )
            # Share already-loaded model weights
            proc._models = {
                "audio_encoder": self._core.audio_encoder,
                "identity_encoder": self._core.identity_encoder,
                "generator": self._core.generator,
            }
            proc._models_loaded = True

            proc.process(
                video_path=video,
                audio_path=audio,
                output_path=tmp_out,
            )

            import cv2
            cap = cv2.VideoCapture(str(tmp_out))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

        result = InferenceResult(frames, fps=fps)
        if output_path:
            result.save(output_path)
        return result

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_onnx(self, save_path: str | Path = "model.onnx") -> None:
        """Export the generator to ONNX.

        Args:
            save_path: Output .onnx file.
        """
        from .inference.realtime.pipeline import RealTimePipeline

        # Create a temporary checkpoint to use the export utility
        import tempfile, torch

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_ckpt = Path(f.name)
        self._core.save_checkpoint(tmp_ckpt)

        pipeline = RealTimePipeline(checkpoint_path=tmp_ckpt, device=str(self.device))
        pipeline.export_onnx(save_path)
        tmp_ckpt.unlink(missing_ok=True)

    def save(self, path: str | Path) -> None:
        """Save model checkpoint."""
        self._core.save_checkpoint(path)

    def load(self, path: str | Path) -> None:
        """Load model checkpoint."""
        self._core.load_checkpoint(path)


__all__ = [
    # High-level lip-sync API
    "LipSyncTrainer",
    "LipSyncConfig",
    "InferenceResult",
    # Neural network builder
    "Sequential",
    "Linear",
    "Conv2d",
    "Flatten",
    "Dropout",
    "MaxPool2d",
    "BatchNorm1d",
    "LayerNorm",
    # Dataset utilities
    "Dataset",
    # Callbacks
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LRSchedulerCallback",
    "WandbLogger",
]

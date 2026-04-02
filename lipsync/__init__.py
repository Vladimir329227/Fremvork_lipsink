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

from .config import DEFAULT_INFERENCE, merge_inference_defaults, validate_config
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
from .training.checkpoint import migrate_to_v2, validate_checkpoint_v2
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
    audio: dict[str, Any] = field(default_factory=lambda: {"sample_rate": 16000, "n_mels": 80, "window": 16})
    video: dict[str, Any] = field(default_factory=lambda: {"face_size": 256, "lip_size": 96, "target_fps": 25.0})
    lipsync: dict[str, Any] = field(default_factory=lambda: {"sync_window": 5, "temporal_radius": 2, "mouth_region_weight": 1.0})
    inference: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_INFERENCE))
    data: dict[str, Any] = field(default_factory=dict)
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

    def save(
        self,
        path: str | Path,
        fps: float | None = None,
        audio_wav: str | Path | None = None,
        mux_audio: bool = True,
    ) -> Path:
        """Write frames to an MP4 file.

        Args:
            path: Output file path.
            fps: Override frame rate.
            audio_wav: If set and *mux_audio*, mux this WAV into the output via ffmpeg.
            mux_audio: When False, skip mux even if *audio_wav* is provided.

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
        if mux_audio and audio_wav is not None and Path(audio_wav).exists():
            from .inference.mux import mux_video_audio

            mux_video_audio(path, Path(audio_wav), path)
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
        validate_config(self.config.to_dict())
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
        validate_config(cfg.to_dict())
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
        ckpt = migrate_to_v2(ckpt)
        validate_checkpoint_v2(ckpt)
        cfg_dict = ckpt.get("config", {})
        cfg_dict = copy.deepcopy(cfg_dict)
        cfg_dict["inference"] = merge_inference_defaults(cfg_dict.get("inference"))
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

        nw = self.config.data.get("num_workers", num_workers) if self.config.data else num_workers
        self._core.fit(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=epochs or self.config.epochs,
            batch_size=batch_size or self.config.batch_size,
            callbacks=cbs,
            num_workers=int(nw),
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
        inference_overrides: dict[str, Any] | None = None,
    ) -> InferenceResult:
        """Run lip-sync inference on a video with the given audio.

        Args:
            audio: Path to driving audio (WAV).
            video: Path to source video.
            output_path: If given, also saves result to this path.
            use_sr: Override config super-resolution setting.
            inference_overrides: Optional keys merged over checkpoint ``inference``
                (e.g. ``{"mouth_composite_mode": "blend"}`` to avoid raw paste noise).

        Returns:
            InferenceResult with processed frames.
        """
        from .inference.batch.processor import BatchProcessor

        sr = use_sr if use_sr is not None else self.config.use_super_resolution

        ckpt_arg = getattr(self._core, "_loaded_checkpoint_path", None) or (
            Path(self.config.checkpoint_dir) / "best_model.pt"
        )
        # Prefer merged trainer ``inference`` over raw checkpoint file (avoids stale / partial infer).
        tr_inf = merge_inference_defaults(self.config.to_dict().get("inference"))
        if inference_overrides:
            tr_inf = {**tr_inf, **inference_overrides}
        proc = BatchProcessor(
            checkpoint_path=ckpt_arg,
            device=str(self.device),
            use_sr=sr,
            sr_backend=self.config.sr_backend,
            inference_overrides=tr_inf,
        )
        proc._models = {
            "audio_encoder": self._core.audio_encoder,
            "identity_encoder": self._core.identity_encoder,
            "generator": self._core.generator,
        }

        fps = float(self.config.video.get("target_fps", 25.0))
        out_path = Path(output_path) if output_path else None
        if out_path is None:
            import tempfile

            tmp_dir = tempfile.mkdtemp()
            out_path = Path(tmp_dir) / "result.mp4"
        else:
            tmp_dir = None

        try:
            proc.process(
                video_path=video,
                audio_path=audio,
                output_path=out_path,
                fps=fps,
            )

            import cv2

            cap = cv2.VideoCapture(str(out_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or fps
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
        finally:
            if tmp_dir:
                import shutil

                shutil.rmtree(tmp_dir, ignore_errors=True)

        result = InferenceResult(frames, fps=fps)
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
    # Unified apply API
    "apply_batch",
    "apply_batch_pairs",
    "apply_realtime",
]


def apply_batch(*args, **kwargs):
    from .apply import apply_batch as _apply_batch
    return _apply_batch(*args, **kwargs)


def apply_batch_pairs(*args, **kwargs):
    from .apply import apply_batch_pairs as _apply_batch_pairs
    return _apply_batch_pairs(*args, **kwargs)


def apply_realtime(*args, **kwargs):
    from .apply import apply_realtime as _apply_realtime
    return _apply_realtime(*args, **kwargs)

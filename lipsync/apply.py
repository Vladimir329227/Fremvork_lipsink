"""Unified model application helpers (batch + realtime)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .api.app import run_server
from .inference.batch import BatchProcessor


def apply_batch(
    checkpoint: str | Path,
    video: str | Path,
    audio: str | Path,
    output: str | Path,
    device: str = "auto",
    use_sr: bool = False,
    sr_backend: str = "gfpgan",
    fps: float = 25.0,
) -> Path:
    proc = BatchProcessor(
        checkpoint_path=checkpoint,
        device=device,
        use_sr=use_sr,
        sr_backend=sr_backend,
    )
    return proc.process(video_path=video, audio_path=audio, output_path=output, fps=fps)


def apply_batch_pairs(
    checkpoint: str | Path,
    pairs: Iterable[tuple[str | Path, str | Path, str | Path]],
    device: str = "auto",
    use_sr: bool = False,
    sr_backend: str = "gfpgan",
    fps: float = 25.0,
) -> list[Path]:
    proc = BatchProcessor(
        checkpoint_path=checkpoint,
        device=device,
        use_sr=use_sr,
        sr_backend=sr_backend,
    )
    outputs: list[Path] = []
    for video, audio, output in pairs:
        outputs.append(proc.process(video_path=video, audio_path=audio, output_path=output, fps=fps))
    return outputs


def apply_realtime(
    checkpoint: str | Path,
    host: str = "0.0.0.0",
    port: int = 8000,
    device: str = "auto",
    use_sr: bool = False,
    sr_backend: str = "gfpgan",
    fps: float = 25.0,
    audio_window_ms: float = 200.0,
) -> None:
    run_server(
        checkpoint_path=checkpoint,
        host=host,
        port=port,
        device=device,
        use_sr=use_sr,
        sr_backend=sr_backend,
        fps=fps,
        audio_window_ms=audio_window_ms,
    )


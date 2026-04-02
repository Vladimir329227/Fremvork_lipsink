"""Build a constant-frame MP4 from an image + WAV duration (for talking-head demos)."""
from __future__ import annotations

from pathlib import Path

import wave

import cv2
import numpy as np

from lipsync.data.preprocessing.audio import AudioPreprocessor


def _audio_duration_seconds(audio_path: Path) -> float:
    ap = AudioPreprocessor()
    if audio_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(audio_path), "rb") as w:
                return w.getnframes() / float(w.getframerate())
        except wave.Error:
            pass
    wav, sr = ap.load_wav(audio_path)
    return wav.shape[1] / float(sr)


def image_to_static_mp4(
    image_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    fps: float = 25.0,
) -> Path:
    image_path = Path(image_path)
    buf = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    audio_path = Path(audio_path)
    duration = _audio_duration_seconds(audio_path)
    n_frames = max(1, int(np.ceil(duration * fps)))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    for _ in range(n_frames):
        writer.write(img)
    writer.release()
    return output_path

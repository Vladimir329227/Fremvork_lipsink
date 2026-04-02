"""ffmpeg mux helpers (skipped if ffmpeg not on PATH)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ffmpeg = shutil.which("ffmpeg")


@pytest.mark.skipif(not ffmpeg, reason="ffmpeg not on PATH")
def test_mux_video_audio_roundtrip(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    from lipsync.inference.mux import extract_wav_from_video, mux_video_audio

    vid = tmp_path / "silent.mp4"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (32, 32))
    for _ in range(15):
        w.write(np.zeros((32, 32, 3), dtype=np.uint8))
    w.release()

    wav = tmp_path / "tone.wav"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.5",
            str(wav),
        ],
        check=True,
        capture_output=True,
    )
    out = tmp_path / "with_audio.mp4"
    mux_video_audio(vid, wav, out)
    assert out.exists() and out.stat().st_size > 100

    extracted = tmp_path / "ex.wav"
    extract_wav_from_video(out, extracted)
    assert extracted.exists()

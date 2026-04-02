"""Mux driving audio into a silent or video-only MP4 via ffmpeg."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def mux_video_audio(
    video_path: str | Path,
    audio_wav: str | Path,
    output_path: str | Path,
) -> Path:
    """Re-encode video (H.264) and AAC audio into *output_path* (shortest stream wins)."""
    video_path = Path(video_path)
    audio_wav = Path(audio_wav)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".mux.tmp.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_wav),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found; install ffmpeg and add it to PATH.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg mux failed: {e.stderr or e.stdout}") from e
    if output_path.exists():
        output_path.unlink()
    tmp.replace(output_path)
    return output_path


def extract_wav_from_video(
    video_path: str | Path,
    wav_out: str | Path | None = None,
    sample_rate: int = 16000,
) -> Path:
    """Extract mono PCM WAV; if *wav_out* is None, uses a temp file (caller may delete)."""
    video_path = Path(video_path)
    if wav_out is None:
        fd, name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        wav_out = Path(name)
    else:
        wav_out = Path(wav_out)
        wav_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(wav_out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found; install ffmpeg and add it to PATH.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg extract wav failed: {e.stderr or e.stdout}") from e
    return wav_out

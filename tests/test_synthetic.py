"""Synthetic dataset builder."""
from __future__ import annotations

from pathlib import Path

from lipsync.data import write_correlated_synthetic_dataset


def test_write_correlated_synthetic_dataset(tmp_path: Path) -> None:
    s = write_correlated_synthetic_dataset(
        tmp_path,
        n_train_clips=1,
        n_val_clips=1,
        n_frames=12,
        face_size=64,
        seed=0,
    )
    assert s["train_clips"] == 1
    assert (tmp_path / "train_metadata.json").exists()
    assert (tmp_path / "val_metadata.json").exists()
    sid = "train_clip_000"
    frames = tmp_path / "samples" / sid / "frames"
    assert len(list(frames.glob("*.jpg"))) == 12
    assert (tmp_path / "samples" / sid / "audio.pt").exists()

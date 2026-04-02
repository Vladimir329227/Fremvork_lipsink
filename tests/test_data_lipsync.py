"""Smoke tests for lipsync.data (no GRID files required)."""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from lipsync.data.datasets.lipsync_dataset import LipSyncDataset, VideoDataset
from lipsync.data.preprocessing.audio import AudioPreprocessor
from lipsync.data.validation import validate_dataset
from lipsync.inference.static_clip import image_to_static_mp4


def _write_silent_wav(path: Path, n_samples: int = 8000, sr: int = 16000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * n_samples)


def test_audio_preprocessor_mel_shape(tmp_path: Path) -> None:
    wav_path = tmp_path / "t.wav"
    _write_silent_wav(wav_path, n_samples=int(16000 * 0.5), sr=16000)

    ap = AudioPreprocessor()
    mel = ap.process_file(wav_path)
    assert mel.ndim == 2
    assert mel.shape[1] == 80


def test_lipsync_dataset_minimal(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    samples = root / "samples" / "clip_a"
    samples.mkdir(parents=True)
    (samples / "frames").mkdir()
    import cv2

    for i in range(5):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:, :] = (40 + i * 10, 80, 120)
        cv2.imwrite(str(samples / "frames" / f"{i+1:06d}.jpg"), img)

    mel = torch.randn(40, 80)
    torch.save(mel, samples / "audio.pt")
    torch.save(torch.zeros(5, 68, 2), samples / "landmarks.pt")

    meta = [{"id": "clip_a", "n_frames": 5, "fps": 25.0, "speaker": "s0"}]
    with open(root / "train_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    with open(root / "val_metadata.json", "w", encoding="utf-8") as f:
        json.dump([], f)

    ds = LipSyncDataset(root, split="train", augment=False, audio_window=16)
    assert len(ds) == 5
    b = ds[0]
    assert b["mel"].shape == (16, 80)
    assert b["face"].shape == (3, 256, 256)
    assert b["sync_lips"].shape[0] == 15


def test_lipsync_dataset_static_face_prob(tmp_path: Path) -> None:
    root = tmp_path / "ds_static"
    samples = root / "samples" / "clip_a"
    samples.mkdir(parents=True)
    (samples / "frames").mkdir()
    import cv2

    # First frame is dark, second is bright -> easy to detect if face uses frame-0.
    img0 = np.zeros((128, 128, 3), dtype=np.uint8)
    img1 = np.full((128, 128, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(samples / "frames" / "000001.jpg"), img0)
    cv2.imwrite(str(samples / "frames" / "000002.jpg"), img1)

    mel = torch.randn(40, 80)
    torch.save(mel, samples / "audio.pt")
    meta = [{"id": "clip_a", "n_frames": 2, "fps": 25.0, "speaker": "s0"}]
    with open(root / "train_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    with open(root / "val_metadata.json", "w", encoding="utf-8") as f:
        json.dump([], f)

    ds = LipSyncDataset(
        root,
        split="train",
        augment=False,
        audio_window=16,
        face_size=128,
        lip_size=64,
        static_face_prob=1.0,
    )
    b = ds[1]
    # face should be ref frame-0, while gt_frame should stay current frame (bright).
    assert b["face"].mean().item() < -0.9
    assert b["gt_frame"].mean().item() > 0.9


def test_validate_dataset(tmp_path: Path) -> None:
    rep = validate_dataset(tmp_path, split="train")
    assert rep.error_count >= 1


def test_static_clip_writes_video(tmp_path: Path) -> None:
    import cv2

    img_path = tmp_path / "i.png"
    cv2.imwrite(str(img_path), np.zeros((120, 160, 3), dtype=np.uint8))
    wav_path = tmp_path / "a.wav"
    _write_silent_wav(wav_path, n_samples=8000, sr=16000)
    out = tmp_path / "o.mp4"
    image_to_static_mp4(img_path, wav_path, out, fps=25.0)
    assert out.exists()


def test_video_dataset_dummy_video(tmp_path: Path) -> None:
    import cv2

    vid = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 64))
    for _ in range(10):
        w.write(np.zeros((64, 64, 3), dtype=np.uint8))
    w.release()

    wav_path = tmp_path / "a.wav"
    _write_silent_wav(wav_path, n_samples=int(16000 * 0.4), sr=16000)

    ds = VideoDataset(vid, audio_path=wav_path, target_fps=25.0, face_size=256, audio_window=16)
    assert len(ds) >= 1
    s = ds[0]
    assert s["mel"].shape == (16, 80)


def test_video_dataset_anchor_same_crop_all_frames(tmp_path: Path) -> None:
    """Identical frames → identical face tensors (no per-frame Haar zoom)."""
    import cv2

    vid = tmp_path / "same.mp4"
    writer = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (128, 128))
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    frame[40:90, 40:90] = (200, 100, 50)
    for _ in range(8):
        writer.write(frame.copy())
    writer.release()

    wav_path = tmp_path / "a.wav"
    _write_silent_wav(wav_path, n_samples=int(16000 * 0.5), sr=16000)

    ds = VideoDataset(vid, audio_path=wav_path, target_fps=25.0, face_size=64, audio_window=16)
    f0 = ds[0]["face"]
    f7 = ds[7]["face"]
    assert torch.allclose(f0, f7, atol=1e-5, rtol=0)

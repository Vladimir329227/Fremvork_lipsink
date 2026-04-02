#!/usr/bin/env python3
"""
Preprocess *your* talking-head video for Rudrabha/Wav2Lip, short finetune (cloned repo),
then lip-sync the same (or another) video to *driving* audio.

Requires: CUDA, ffmpeg, external/Wav2Lip clone, checkpoints/wav2lip_gan.pth, lipsync_expert.pth.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W2L = ROOT / "external" / "Wav2Lip"
GAN = W2L / "checkpoints" / "wav2lip_gan.pth"
EXPERT = W2L / "checkpoints" / "lipsync_expert.pth"
EXPERT_URL = "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/lipsync_expert.pth"


def _download(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading", dest.name, "…")
    urllib.request.urlretrieve(url, dest)
    print("Saved", dest)


def ensure_expert() -> None:
    if not EXPERT.is_file():
        _download(EXPERT_URL, EXPERT)


def extract_audio_16k(video: Path, out_wav: Path, fallback_mono: Path | None = None) -> None:
    """Prefer audio from *video*; if it has no audio stream, resample *fallback_mono* to 16 kHz mono."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(out_wav),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        return
    if fallback_mono is None or not fallback_mono.is_file():
        raise RuntimeError(
            "Video has no audio (or ffmpeg failed). Re-run with --train-audio pointing to a WAV "
            "that matches the mouth motion in --train-video, or use a video with an embedded track.\n"
            f"ffmpeg stderr:\n{r.stderr}"
        )
    cmd2 = [
        "ffmpeg",
        "-y",
        "-i",
        str(fallback_mono),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(out_wav),
    ]
    r2 = subprocess.run(cmd2, capture_output=True, text=True)
    if r2.returncode != 0:
        raise RuntimeError(f"ffmpeg fallback failed: {r2.stderr}")


def preprocess_faces(video: Path, clip_dir: Path, train_audio_fallback: Path | None) -> int:
    """Write clip_dir/{0,1,...}.jpg face crops + audio.wav. Indices match video frame order."""
    sys.path.insert(0, str(W2L))
    import face_detection  # noqa: WPS433

    if not (W2L / "face_detection/detection/sfd/s3fd.pth").is_file():
        raise SystemExit("Missing s3fd.pth — run scripts/wav2lip_infer.py once to fetch weights.")

    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if len(frames) < 30:
        raise SystemExit(f"Need at least ~30 frames, got {len(frames)}")

    clip_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if _cuda_ok() else "cpu"
    fa = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, flip_input=False, device=device
    )

    last_crop: np.ndarray | None = None
    n_written = 0
    for idx, fr in enumerate(frames):
        preds = fa.get_detections_for_batch(np.asarray([fr]))
        f = preds[0] if preds else None
        if f is None:
            crop = last_crop if last_crop is not None else cv2.resize(fr, (96, 96), interpolation=cv2.INTER_AREA)
        else:
            x1, y1, x2, y2 = f
            crop = fr[y1:y2, x1:x2]
            if crop.size == 0:
                crop = last_crop if last_crop is not None else cv2.resize(fr, (96, 96), interpolation=cv2.INTER_AREA)
            else:
                last_crop = crop
        crop96 = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(clip_dir / f"{idx}.jpg"), crop96)
        n_written += 1

    extract_audio_16k(video, clip_dir / "audio.wav", fallback_mono=train_audio_fallback)
    print(f"Preprocessed {n_written} face crops into {clip_dir}")
    return n_written


def _cuda_ok() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def write_filelists(clip_key: str) -> None:
    fl = W2L / "filelists"
    fl.mkdir(exist_ok=True)
    (fl / "train.txt").write_text(clip_key + "\n", encoding="utf-8")
    (fl / "val.txt").write_text(clip_key + "\n", encoding="utf-8")


def latest_ckpt(ckpt_dir: Path) -> Path:
    cks = list(ckpt_dir.glob("checkpoint_step*.pth"))
    if not cks:
        raise SystemExit(f"No checkpoints in {ckpt_dir}")

    def step_key(p: Path) -> int:
        m = re.search(r"checkpoint_step(\d+)\.pth$", p.name)
        return int(m.group(1)) if m else 0

    return max(cks, key=step_key)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-video", type=Path, required=True, help="Video with clear face")
    p.add_argument(
        "--train-audio",
        type=Path,
        default=None,
        help="WAV (or any ffmpeg-readable audio) aligned with train-video if the MP4 has no sound",
    )
    p.add_argument("--infer-video", type=Path, default=None, help="Video for final lip-sync (default: same as train)")
    p.add_argument("--audio", type=Path, required=True, help="Driving audio (wav/mp4/…) for final output")
    p.add_argument("--workspace", type=Path, default=ROOT / "external" / "wav2lip_user_workspace")
    p.add_argument("--clip-name", type=str, default="user_clip", help="Folder name under preprocessed/")
    p.add_argument("--max-steps", type=int, default=500)
    args = p.parse_args()

    if not W2L.is_dir():
        raise SystemExit(f"Clone Wav2Lip first: git clone https://github.com/Rudrabha/Wav2Lip.git external/Wav2Lip")
    if not GAN.is_file():
        raise SystemExit(f"Missing {GAN} — run: python scripts/wav2lip_infer.py --face dummy --audio dummy (downloads GAN)")

    ensure_expert()

    pre_root = args.workspace / "preprocessed"
    clip_dir = pre_root / args.clip_name
    train_audio_fb = (args.train_audio or args.audio).resolve()
    preprocess_faces(args.train_video.resolve(), clip_dir, train_audio_fallback=train_audio_fb)
    write_filelists(args.clip_name)

    ckpt_dir = args.workspace / "finetune_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        sys.executable,
        str(W2L / "wav2lip_train_short.py"),
        "--data_root",
        str(pre_root),
        "--checkpoint_dir",
        str(ckpt_dir),
        "--syncnet_checkpoint_path",
        str(EXPERT),
        "--init_wav2lip",
        str(GAN),
        "--max_steps",
        str(args.max_steps),
        "--checkpoint_interval",
        str(max(50, args.max_steps // 3)),
    ]
    print("Training:", " ".join(train_cmd))
    r = subprocess.run(train_cmd, cwd=str(W2L))
    if r.returncode != 0:
        sys.exit(r.returncode)

    best = latest_ckpt(ckpt_dir)
    infer_video = (args.infer_video or args.train_video).resolve()
    out = args.workspace / "lip_sync_output.mp4"

    inf_cmd = [
        sys.executable,
        str(W2L / "inference.py"),
        "--checkpoint_path",
        str(best),
        "--face",
        str(infer_video),
        "--audio",
        str(args.audio.resolve()),
        "--outfile",
        str(out),
        "--pads",
        "0",
        "10",
        "0",
        "0",
    ]
    print("Inference:", " ".join(inf_cmd))
    r = subprocess.run(inf_cmd, cwd=str(W2L))
    if r.returncode != 0:
        sys.exit(r.returncode)
    print("Output:", out)


if __name__ == "__main__":
    main()

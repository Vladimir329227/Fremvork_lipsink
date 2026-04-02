#!/usr/bin/env python3
"""Run official Rudrabha/Wav2Lip inference (GitHub) with pretrained weights.

Wav2Lip paper: https://arxiv.org/abs/2008.10010 — reference lip-sync pipeline used
for comparison / production-quality results. Our framework trainer is separate;
this script shells into ``external/Wav2Lip`` after optional weight download.

Weights (non-commercial / research): see Wav2Lip README. Default checkpoint is
mirrored on Hugging Face for scripted download.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WAV2LIP_DIR = ROOT / "external" / "Wav2Lip"
GAN_CKPT = WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"
S3FD_CKPT = WAV2LIP_DIR / "face_detection" / "detection" / "sfd" / "s3fd.pth"
HF_GAN_URL = "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth"
S3FD_URL = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"


def _download(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} …")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved {dest}")


def ensure_weights() -> None:
    if not WAV2LIP_DIR.is_dir():
        raise SystemExit(
            f"Missing {WAV2LIP_DIR}. Clone with:\n"
            f"  git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git external/Wav2Lip"
        )
    if not GAN_CKPT.is_file():
        _download(HF_GAN_URL, GAN_CKPT)
    if not S3FD_CKPT.is_file():
        _download(S3FD_URL, S3FD_CKPT)
    # librosa >= 0.10 compatibility (mel() keyword-only)
    audio_py = WAV2LIP_DIR / "audio.py"
    text = audio_py.read_text(encoding="utf-8")
    if "sr=hp.sample_rate" not in text and "_build_mel_basis" in text:
        print("[wav2lip_infer] Patching external/Wav2Lip/audio.py for modern librosa …")
        text = text.replace(
            "return librosa.core.load(path, sr=sr)[0]",
            "return librosa.load(path, sr=sr, mono=True)[0]",
        )
        old = (
            "    return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,\n"
            "                               fmin=hp.fmin, fmax=hp.fmax)\n"
        )
        new = (
            "    return librosa.filters.mel(\n"
            "        sr=hp.sample_rate,\n"
            "        n_fft=hp.n_fft,\n"
            "        n_mels=hp.num_mels,\n"
            "        fmin=hp.fmin,\n"
            "        fmax=hp.fmax,\n"
            "    )\n"
        )
        if old in text:
            text = text.replace(old, new)
        audio_py.write_text(text, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Wav2Lip (official repo) lip-sync inference")
    p.add_argument("--face", required=True, help="Video or image with face(s)")
    p.add_argument("--audio", required=True, help="Driving audio (wav/mp3/…) ")
    p.add_argument("--outfile", default="wav2lip_out.mp4", help="Output MP4 path")
    p.add_argument("--pads", nargs=4, type=int, default=[0, 10, 0, 0], metavar=("T", "B", "L", "R"))
    p.add_argument("--resize-factor", type=int, default=1)
    p.add_argument("--nosmooth", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None, help="Override path to .pth")
    args = p.parse_args()

    ensure_weights()
    ckpt = Path(args.checkpoint) if args.checkpoint else GAN_CKPT
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    face = Path(args.face).resolve()
    audio = Path(args.audio).resolve()
    out = Path(args.outfile).resolve()
    if not face.exists():
        raise SystemExit(f"Missing --face: {face}")
    if not audio.exists():
        raise SystemExit(f"Missing --audio: {audio}")

    cmd = [
        sys.executable,
        str(WAV2LIP_DIR / "inference.py"),
        "--checkpoint_path",
        str(ckpt),
        "--face",
        str(face),
        "--audio",
        str(audio),
        "--outfile",
        str(out),
        "--pads",
        *[str(x) for x in args.pads],
        "--resize_factor",
        str(args.resize_factor),
    ]
    if args.nosmooth:
        cmd.append("--nosmooth")

    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(WAV2LIP_DIR))
    if r.returncode != 0:
        sys.exit(r.returncode)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()

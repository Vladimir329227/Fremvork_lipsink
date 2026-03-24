#!/usr/bin/env python3
"""Demo lip-sync pipeline for GRID corpus.

Steps:
    1. Load up to --clips GRID MPG clips (no face detection — GRID is pre-cropped).
    2. Train a small lip-sync GAN for --epochs epochs (~13 min with --fast on i7-12700K).
    3. Apply the trained model to all (4 videos × 3 WAVs) = 12 pairs in result/.
    4. Save each output to result/1/ with a per-clip metadata JSON.
    5. Save a summary result/1/summary.json with avg generation time & realtime factor.

Usage:
    python scripts/run_lipsync_result.py --fast          # ~13 min
    python scripts/run_lipsync_result.py --clips 200 --epochs 10  # ~1.7 h
    python scripts/run_lipsync_result.py --skip-train    # inference only (needs checkpoint)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Auto-install missing packages (demo convenience)
# ---------------------------------------------------------------------------

def _ensure(pkg: str, mod: str | None = None) -> None:
    import importlib
    try:
        importlib.import_module(mod or pkg.replace("-", "_"))
        return
    except ImportError:
        pass
    print(f"[setup] Installing {pkg} ...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
    importlib.import_module(mod or pkg.replace("-", "_"))


_ensure("opencv-python", "cv2")
_ensure("numpy")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io.wavfile as wavfile  # noqa: E402
import scipy.signal as signal  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

# Framework (no torchvision in critical path)
from lipsync.models.audio_encoder.conformer import AudioEncoder
from lipsync.models.generator.lipsync_gen import LipSyncGenerator
from lipsync.models.discriminator.patch_disc import PatchDiscriminator
from lipsync.optimizers.adam import AdamW
from lipsync.losses.regression import HuberLoss
from lipsync.losses.lipsync_losses import AdversarialLoss


# ---------------------------------------------------------------------------
# Lightweight identity encoder (no torchvision)
# ---------------------------------------------------------------------------

class SimpleIdentityEncoder(nn.Module):
    """5-layer CNN identity encoder — no torchvision dependency."""

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# GRID in-memory dataset (no face_alignment needed)
# ---------------------------------------------------------------------------

_SR = 16_000
_N_MELS = 80


# ---------------------------------------------------------------------------
# Mel-spectrogram via scipy (no torchaudio needed)
# ---------------------------------------------------------------------------

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    low_hz, high_hz = 0.0, sr / 2.0
    low_mel = _hz_to_mel(low_hz)
    high_mel = _hz_to_mel(high_hz)
    mel_pts = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_pts = np.array([_mel_to_hz(m) for m in mel_pts])
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, cen, hi = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        for k in range(lo, cen):
            if cen > lo:
                fb[m - 1, k] = (k - lo) / (cen - lo)
        for k in range(cen, hi):
            if hi > cen:
                fb[m - 1, k] = (hi - k) / (hi - cen)
    return fb


_FB_CACHE: dict[tuple, np.ndarray] = {}


def wav_to_mel(
    wav: np.ndarray,
    sr: int,
    target_sr: int = _SR,
    n_mels: int = _N_MELS,
    n_fft: int = 800,
    hop_length: int = 200,
    top_db: float = 80.0,
) -> torch.Tensor:
    """Compute (T, n_mels) mel spectrogram from a mono float32 waveform."""
    if sr != target_sr:
        n_out = int(len(wav) * target_sr / sr)
        wav = signal.resample(wav, n_out)

    freqs, times, Zxx = signal.spectrogram(
        wav, fs=target_sr, nperseg=n_fft, noverlap=n_fft - hop_length,
        window="hann", mode="magnitude",
    )

    key = (target_sr, n_fft, n_mels)
    if key not in _FB_CACHE:
        _FB_CACHE[key] = _mel_filterbank(target_sr, n_fft, n_mels)
    fb = _FB_CACHE[key]

    mel = fb @ Zxx             # (n_mels, T)
    mel = mel ** 2             # power
    mel = np.maximum(mel, 1e-10)
    mel_db = 10.0 * np.log10(mel)
    mel_db = np.maximum(mel_db, mel_db.max() - top_db)
    mel_norm = (mel_db + top_db) / top_db
    return torch.from_numpy(mel_norm.T.astype(np.float32))  # (T, n_mels)


def load_wav(path: Path, target_sr: int = _SR) -> tuple[np.ndarray, int]:
    """Load WAV file using scipy. Returns (mono float32, sample_rate)."""
    sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if data.max() > 1.0:
        data /= 32768.0 if data.max() < 40000 else data.max()
    return data, sr


class GRIDFastDataset(Dataset):
    """Load GRID MPG clips directly — GRID videos are pre-cropped frontal faces.

    Face = full resized frame.
    Lip region = bottom 40 % centre 60 % crop (fixed ratio, no landmarks needed).
    """

    def __init__(
        self,
        mpg_files: list[Path],
        face_size: int = 128,
        lip_size: int = 64,
        audio_window: int = 16,
        augment: bool = True,
    ) -> None:
        self.face_size = face_size
        self.lip_size = lip_size
        self.audio_window = audio_window
        self.augment = augment

        print(f"Caching {len(mpg_files)} GRID clips …", flush=True)
        self.cache: list[dict] = []
        for i, mpg in enumerate(mpg_files):
            entry = self._load(mpg)
            if entry:
                self.cache.append(entry)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(mpg_files)}", flush=True)
        print(f"Dataset ready: {len(self.cache)} clips.", flush=True)

    # ------------------------------------------------------------------
    def _extract_audio(self, path: Path) -> tuple[np.ndarray, int]:
        """Extract mono float32 PCM from video or wav file."""
        # Try reading as WAV directly
        try:
            return load_wav(path)
        except Exception:
            pass
        # Try ffmpeg extraction to tmp wav
        if shutil.which("ffmpeg"):
            tmp = tempfile.mktemp(suffix=".wav")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(path), "-ac", "1", "-ar", str(_SR), tmp],
                    check=True, capture_output=True,
                )
                result = load_wav(Path(tmp))
                Path(tmp).unlink(missing_ok=True)
                return result
            except Exception:
                Path(tmp).unlink(missing_ok=True)
        # Fallback: silence — model still learns visual movement from video
        return np.zeros(_SR * 3, dtype=np.float32), _SR

    @staticmethod
    def _bgr_tensor(frame: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        return torch.from_numpy(rgb).permute(2, 0, 1) / 255.0 * 2.0 - 1.0

    def _load(self, mpg: Path) -> dict | None:
        cap = cv2.VideoCapture(str(mpg))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames: list[np.ndarray] = []
        while cap.isOpened():
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(f, (self.face_size, self.face_size),
                                     interpolation=cv2.INTER_AREA))
        cap.release()
        if len(frames) < 10:
            return None
        wav_data, sr = self._extract_audio(mpg)
        mel = wav_to_mel(wav_data, sr)
        return {"frames": frames, "mel": mel, "fps": fps}

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.cache) * 5   # 5 random samples per cached clip

    def __getitem__(self, idx: int) -> dict:
        entry = self.cache[idx % len(self.cache)]
        n = len(entry["frames"])
        fps = entry["fps"]
        mel = entry["mel"]          # (T_mel, 80)

        fi = random.randint(0, n - 1)
        face_t = self._bgr_tensor(entry["frames"][fi])

        ri = random.choice([i for i in range(n) if i != fi] or [fi])
        ref_t = self._bgr_tensor(entry["frames"][ri])

        # Fixed lip region: bottom 40 % of frame, centre 60 %
        f = entry["frames"][fi]
        h, w = f.shape[:2]
        y1, x1 = int(h * 0.60), int(w * 0.20)
        y2, x2 = h, int(w * 0.80)
        lip_crop = f[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else f
        lip_t = self._bgr_tensor(
            cv2.resize(lip_crop, (self.lip_size, self.lip_size), interpolation=cv2.INTER_AREA)
        )

        # Mel window centred on fi
        T_mel = mel.shape[0]
        mel_fps = T_mel / (n / fps) if fps > 0 else T_mel / n
        centre = int(fi / fps * mel_fps) if fps > 0 else fi
        half = self.audio_window // 2
        start = max(0, centre - half)
        win = mel[start: start + self.audio_window]
        if win.shape[0] < self.audio_window:
            win = torch.cat([win, torch.zeros(self.audio_window - win.shape[0], _N_MELS)])

        return {
            "mel": win,          # (16, 80)
            "face": face_t,      # (3, 128, 128)
            "lip_crop": lip_t,   # (3, 64, 64)
            "gt_frame": face_t,
            "ref_face": ref_t,
        }


# ---------------------------------------------------------------------------
# Demo GAN trainer
# ---------------------------------------------------------------------------

DEMO_CFG = dict(
    embed_dim=256,
    audio_d_model=128,
    audio_heads=2,
    audio_layers=3,
    gen_base_ch=32,
    gen_depth=3,
    disc_base_ch=32,
    lr=2e-4,
)


class DemoTrainer:
    """Self-contained GAN trainer — only torch, no torchvision."""

    def __init__(self, cfg: dict, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        torch.set_num_threads(os.cpu_count() or 4)

        E = cfg["embed_dim"]
        self.audio_enc = AudioEncoder(
            n_mels=80, d_model=cfg["audio_d_model"],
            num_heads=cfg["audio_heads"], num_layers=cfg["audio_layers"],
            embed_dim=E,
        ).to(self.device)

        self.id_enc = SimpleIdentityEncoder(E).to(self.device)

        self.gen = LipSyncGenerator(
            in_channels=4, base_ch=cfg["gen_base_ch"],
            num_encoder_blocks=cfg["gen_depth"],
            audio_dim=E, identity_dim=E,
        ).to(self.device)

        self.disc = PatchDiscriminator(
            in_channels=3, base_ch=cfg["disc_base_ch"], n_layers=2,
        ).to(self.device)

        self.recon = HuberLoss()
        self.adv = AdversarialLoss(mode="hinge")

        g_params = (
            list(self.audio_enc.parameters())
            + list(self.id_enc.parameters())
            + list(self.gen.parameters())
        )
        self.opt_g = AdamW(g_params, lr=cfg["lr"], weight_decay=1e-4)
        self.opt_d = AdamW(self.disc.parameters(), lr=cfg["lr"] * 0.5)

    def _fwd(self, batch: dict):
        mel = batch["mel"].to(self.device)
        face = batch["face"].to(self.device)
        ref = batch["ref_face"].to(self.device)
        gt = batch["gt_frame"].to(self.device)

        masked = face.clone()
        masked[:, :, face.shape[-2] // 2:, :] = 0.0
        m4 = torch.cat([masked, torch.zeros_like(masked[:, :1])], dim=1)

        audio_emb = self.audio_enc(mel)[:, -1]
        id_emb = self.id_enc(ref)
        rgb, alpha = self.gen(m4, audio_emb, id_emb)
        generated = alpha * rgb + (1 - alpha) * face
        return generated, gt

    def train_epoch(self, loader: DataLoader) -> dict:
        self.audio_enc.train(); self.id_enc.train()
        self.gen.train(); self.disc.train()
        tots: dict[str, float] = {}

        for batch in loader:
            generated, gt = self._fwd(batch)

            # Discriminator
            rl = self.disc(gt)
            fl = self.disc(generated.detach())
            ld = self.adv.discriminator_loss(rl, fl)
            self.opt_d.zero_grad(); ld.backward(); self.opt_d.step()

            # Generator
            generated, gt = self._fwd(batch)
            lr = self.recon(generated, gt)
            la = self.adv.generator_loss(self.disc(generated))
            lg = 10.0 * lr + la
            self.opt_g.zero_grad(); lg.backward()
            nn.utils.clip_grad_norm_(
                list(self.audio_enc.parameters()) + list(self.gen.parameters()), 1.0
            )
            self.opt_g.step()

            for k, v in [("d", ld), ("g_recon", lr), ("g_adv", la)]:
                tots[k] = tots.get(k, 0.0) + v.item()

        n = max(len(loader), 1)
        return {k: round(v / n, 5) for k, v in tots.items()}

    def fit(self, ds: Dataset, epochs: int, bs: int = 8) -> list[dict]:
        loader = DataLoader(ds, batch_size=bs, shuffle=True,
                            num_workers=0, drop_last=True)
        history = []
        for ep in range(1, epochs + 1):
            t0 = time.perf_counter()
            logs = self.train_epoch(loader)
            dt = time.perf_counter() - t0
            logs.update(epoch=ep, epoch_time_s=round(dt, 2))
            history.append(logs)
            print(
                f"Epoch {ep:3d}/{epochs} | "
                f"D={logs['d']:.4f}  G_rec={logs['g_recon']:.4f}  "
                f"G_adv={logs['g_adv']:.4f}  {dt:.0f}s",
                flush=True,
            )
        return history

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "audio_enc": self.audio_enc.state_dict(),
            "id_enc": self.id_enc.state_dict(),
            "gen": self.gen.state_dict(),
            "cfg": self.cfg,
        }, path)
        print(f"Checkpoint → {path}")

    def load(self, path: Path) -> None:
        ckpt = torch.load(str(path), map_location=self.device, weights_only=True)
        self.audio_enc.load_state_dict(ckpt["audio_enc"])
        self.id_enc.load_state_dict(ckpt["id_enc"])
        self.gen.load_state_dict(ckpt["gen"])
        print(f"Checkpoint ← {path}")


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

class InferencePipeline:
    def __init__(self, trainer: DemoTrainer, face_size: int = 128) -> None:
        self.tr = trainer
        self.face_size = face_size

    def _load_mel(self, wav_path: Path) -> torch.Tensor:
        wav_data, sr = load_wav(wav_path)
        return wav_to_mel(wav_data, sr)   # (T, 80)

    def _mel_win(self, mel: torch.Tensor, fi: int, fps: float,
                 n_frames: int, win: int = 16) -> torch.Tensor:
        T_mel = mel.shape[0]
        mfps = T_mel / (n_frames / fps) if fps > 0 and n_frames > 0 else 1.0
        c = int(fi / fps * mfps) if fps > 0 else fi
        h = win // 2
        s = max(0, c - h)
        w = mel[s: s + win]
        if w.shape[0] < win:
            w = torch.cat([w, torch.zeros(win - w.shape[0], _N_MELS)])
        return w

    @staticmethod
    def _bgr_tensor(frame: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        return torch.from_numpy(rgb).permute(2, 0, 1) / 255.0 * 2.0 - 1.0

    @torch.no_grad()
    def process(
        self,
        video_path: Path,
        audio_path: Path,
        out_path: Path,
        out_fps: float = 25.0,
    ) -> dict:
        t_total_start = time.perf_counter()

        mel_full = self._load_mel(audio_path)   # (T, 80)

        cap = cv2.VideoCapture(str(video_path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or out_fps
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_orig: list[np.ndarray] = []
        while cap.isOpened():
            ok, f = cap.read()
            if not ok:
                break
            frames_orig.append(f)
        cap.release()

        if not frames_orig:
            raise RuntimeError(f"No frames in {video_path}")

        n = len(frames_orig)
        dev = self.tr.device

        # Identity embedding from first frame
        ref_sm = cv2.resize(frames_orig[0], (self.face_size, self.face_size))
        ref_t = self._bgr_tensor(ref_sm).unsqueeze(0).to(dev)
        id_emb = self.tr.id_enc(ref_t)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            out_fps,
            (orig_w, orig_h),
        )

        t_gen_start = time.perf_counter()

        for fi, frame in enumerate(frames_orig):
            face_sm = cv2.resize(frame, (self.face_size, self.face_size))
            face_t = self._bgr_tensor(face_sm).unsqueeze(0).to(dev)

            mel_w = self._mel_win(mel_full, fi, src_fps, n).unsqueeze(0).to(dev)
            audio_emb = self.tr.audio_enc(mel_w)[:, -1]

            masked = face_t.clone()
            masked[:, :, self.face_size // 2:, :] = 0.0
            m4 = torch.cat([masked, torch.zeros_like(masked[:, :1])], dim=1)

            rgb, alpha = self.tr.gen(m4, audio_emb, id_emb)
            gen = (alpha * rgb + (1 - alpha) * face_t).squeeze(0)

            gen_np = ((gen.permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
                      ).clip(0, 255).astype(np.uint8)
            gen_bgr = cv2.cvtColor(gen_np, cv2.COLOR_RGB2BGR)
            gen_full = cv2.resize(gen_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            writer.write(gen_full)

        writer.release()

        t_end = time.perf_counter()
        gen_time = t_end - t_gen_start
        total_time = t_end - t_total_start
        audio_dur = mel_full.shape[0] * 200 / _SR    # approx audio duration
        rt = audio_dur / gen_time if gen_time > 0 else 0.0

        return {
            "video_source": str(video_path),
            "audio_source": str(audio_path),
            "output_video": str(out_path),
            "n_source_frames": n,
            "source_fps": round(src_fps, 3),
            "output_fps": out_fps,
            "generation_time_s": round(gen_time, 3),
            "total_time_s": round(total_time, 3),
            "realtime_factor": round(rt, 3),
            "device": str(dev),
            "timestamp": datetime.now().isoformat(),
            "model_config": self.tr.cfg,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="LipSync Demo Pipeline")
    ap.add_argument("--fast", action="store_true",
                    help="50 clips, 5 epochs (~13 min on i7-12700K)")
    ap.add_argument("--clips", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--speakers", default="s7_processed,s31_processed")
    ap.add_argument("--input-root", type=Path, default=ROOT / "data")
    ap.add_argument("--result-dir", type=Path, default=ROOT / "result")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "result" / "1")
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "checkpoints" / "demo_model.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-train", action="store_true",
                    help="Skip training and use existing checkpoint")
    args = ap.parse_args()

    if args.fast:
        args.clips, args.epochs = 50, 5

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"LipSync Demo  |  clips={args.clips}  epochs={args.epochs}  device={args.device}")
    print("=" * 60)

    trainer = DemoTrainer(DEMO_CFG, device=args.device)

    # ------------------------------------------------------------------
    # Step 1 — Train
    # ------------------------------------------------------------------
    if not args.skip_train:
        sp_dirs = [args.input_root / s.strip() for s in args.speakers.split(",")]
        mpg_all: list[Path] = []
        for sp in sp_dirs:
            mpg_all += sorted(sp.glob("*.mpg"))

        if not mpg_all:
            print(f"ERROR: no .mpg files found under {args.input_root}")
            sys.exit(1)

        random.seed(42)
        random.shuffle(mpg_all)
        selected = mpg_all[: args.clips]
        print(f"\nUsing {len(selected)}/{len(mpg_all)} clips")

        ds = GRIDFastDataset(selected, face_size=128, lip_size=64, audio_window=16)

        print(f"\n--- Training ({args.epochs} epochs) ---")
        t0 = time.perf_counter()
        history = trainer.fit(ds, epochs=args.epochs, bs=args.batch_size)
        train_time = time.perf_counter() - t0

        trainer.save(args.checkpoint)

        train_log = {
            "total_train_time_s": round(train_time, 2),
            "total_train_time_min": round(train_time / 60, 2),
            "epochs": args.epochs,
            "clips": len(selected),
            "history": history,
        }
        (args.output_dir / "training_log.json").write_text(
            json.dumps(train_log, indent=2), encoding="utf-8"
        )
        print(f"\nTraining done in {train_time/60:.1f} min")
    else:
        if not args.checkpoint.exists():
            print(f"ERROR: checkpoint not found at {args.checkpoint}")
            sys.exit(1)
        trainer.load(args.checkpoint)

    # ------------------------------------------------------------------
    # Step 2 — Inference: 4 videos × 3 WAVs = 12 pairs
    # ------------------------------------------------------------------
    videos = sorted(args.result_dir.glob("*.mp4"))
    audios = sorted(args.result_dir.glob("*.wav"))

    if not videos:
        print(f"WARNING: no .mp4 files in {args.result_dir}")
    if not audios:
        print(f"WARNING: no .wav files in {args.result_dir}")

    n_pairs = len(videos) * len(audios)
    print(f"\n--- Inference: {len(videos)} videos × {len(audios)} WAVs = {n_pairs} pairs ---")

    trainer.audio_enc.eval()
    trainer.id_enc.eval()
    trainer.gen.eval()

    pipe = InferencePipeline(trainer, face_size=128)
    all_meta: list[dict] = []

    for vi, video in enumerate(videos):
        for ai, wav in enumerate(audios):
            idx = vi * len(audios) + ai + 1
            out_name = f"{video.stem}__{wav.stem}.mp4"
            out_path = args.output_dir / out_name
            meta_path = args.output_dir / f"{video.stem}__{wav.stem}_meta.json"

            print(f"[{idx}/{n_pairs}] {video.name} + {wav.name}", flush=True)
            try:
                meta = pipe.process(video, wav, out_path, out_fps=25.0)
                meta_path.write_text(json.dumps(meta, indent=2, default=str),
                                     encoding="utf-8")
                all_meta.append(meta)
                print(
                    f"  ✓ frames={meta['n_source_frames']}  "
                    f"gen={meta['generation_time_s']:.1f}s  "
                    f"RT={meta['realtime_factor']:.2f}×",
                    flush=True,
                )
            except Exception as exc:
                print(f"  ERROR: {exc}", flush=True)

    # ------------------------------------------------------------------
    # Step 3 — Summary
    # ------------------------------------------------------------------
    if all_meta:
        avg_gen = sum(m["generation_time_s"] for m in all_meta) / len(all_meta)
        avg_rt = sum(m["realtime_factor"] for m in all_meta) / len(all_meta)
        summary = {
            "total_clips": len(all_meta),
            "avg_generation_time_s": round(avg_gen, 3),
            "avg_realtime_factor": round(avg_rt, 3),
            "note_realtime": "1.0 = real-time, >1.0 = faster than real-time",
            "device": str(trainer.device),
            "model_config": DEMO_CFG,
            "checkpoint": str(args.checkpoint),
            "timestamp": datetime.now().isoformat(),
            "clips": all_meta,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

        print("\n" + "=" * 60)
        print("DONE")
        print(f"  Output dir        : {args.output_dir}")
        print(f"  Generated clips   : {len(all_meta)}")
        print(f"  Avg gen time      : {avg_gen:.1f}s per clip")
        print(f"  Avg realtime × RT : {avg_rt:.2f}×  (1.0 = real-time)")
        print(f"  Summary JSON      : {args.output_dir}/summary.json")
        print("=" * 60)
    else:
        print("No clips generated.")


if __name__ == "__main__":
    main()

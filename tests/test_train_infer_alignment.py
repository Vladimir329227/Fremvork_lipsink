"""Training uses the same mouth composite as inference; reconstruction sanity on synthetic data.

The train/infer mismatch (lower-half blend vs lip-box paste) is a common failure mode versus
Wav2Lip-style pipelines: blend lets the network ignore RGB wherever alpha is low, but paste
inference exposes raw RGB in the mouth ROI — a rectangle of noise.

Reconstruction on a *training* clip here is a pipeline sanity check (not a validation metric):
we assert the stack can fit correlated audio–video when supervised consistently. True
generalisation is measured on held-out val/test clips (see ``run_dataset_reconstruction_verify``).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from lipsync.composite import (
    composite_mouth_region,
    lip_roi_slices,
    mouth_composite_kwargs_from_inference,
    pool_audio_sequence,
)
from lipsync.data import LipSyncDataset, write_correlated_synthetic_dataset
from lipsync.evaluation.metrics import psnr
from lipsync.training.trainer import LipSyncTrainerCore


def test_lip_box_paste_supervises_generator_rgb() -> None:
    """Huber/L1 on composite image must backprop into rgb inside the lip ROI (paste)."""
    face = torch.randn(1, 3, 48, 48)
    rgb = torch.randn(1, 3, 48, 48, requires_grad=True)
    alpha = torch.zeros(1, 1, 48, 48)
    gt = face.clone()
    kw = mouth_composite_kwargs_from_inference(
        {
            "mouth_composite_mode": "paste",
            "mouth_composite_scope": "lip_box",
            "lip_roi_y0": 0.5,
            "lip_roi_y1": 0.92,
            "lip_roi_x0": 0.2,
            "lip_roi_x1": 0.8,
            "lip_roi_feather_px": 0,
        }
    )
    out = composite_mouth_region(face, rgb, alpha, 0.42, **kw)
    loss = (out - gt).pow(2).sum()
    loss.backward()
    assert rgb.grad is not None
    assert rgb.grad.abs().sum() > 1e-6


def test_synthetic_train_val_metadata_disjoint(tmp_path: Path) -> None:
    write_correlated_synthetic_dataset(
        tmp_path,
        n_train_clips=3,
        n_val_clips=2,
        n_frames=8,
        face_size=64,
        seed=7,
    )
    train_ids = {e["id"] for e in json.loads((tmp_path / "train_metadata.json").read_text())}
    val_ids = {e["id"] for e in json.loads((tmp_path / "val_metadata.json").read_text())}
    assert train_ids.isdisjoint(val_ids)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_grid_train_val_disjoint_if_present() -> None:
    """No clip id may appear in both train and val GRID metadata (anti-leak guard)."""
    root = REPO_ROOT / "data/processed_grid"
    tr = root / "train_metadata.json"
    va = root / "val_metadata.json"
    if not tr.is_file() or not va.is_file():
        pytest.skip("processed_grid metadata not in workspace")
    train_ids = {e["id"] for e in json.loads(tr.read_text(encoding="utf-8"))}
    val_ids = {e["id"] for e in json.loads(va.read_text(encoding="utf-8"))}
    assert train_ids.isdisjoint(val_ids)


def test_trainer_uses_inference_composite_kw_parity() -> None:
    """Trainer composite kwargs must match ``mouth_composite_kwargs_from_inference``."""
    cfg_path = REPO_ROOT / "configs" / "base.yaml"
    if not cfg_path.is_file():
        pytest.skip("base.yaml missing")
    with open(cfg_path, encoding="utf-8") as f:
        full = yaml.safe_load(f)
    core = LipSyncTrainerCore(full, device="cpu")
    kw = core._mouth_composite_kw()
    expected = mouth_composite_kwargs_from_inference(full.get("inference", {}))
    assert kw == expected


@pytest.mark.slow
def test_synthetic_train_clip_reconstruction_high_psnr(tmp_path: Path) -> None:
    """Overfit one correlated train clip; ref=frame0 + per-frame mel → near-zero error vs GT.

    Uses only the *training* clip batch (intentional). Val clips are disjoint by construction.
    """
    root = tmp_path / "syn"
    face_size = 96
    n_frames = 16
    write_correlated_synthetic_dataset(
        root,
        n_train_clips=1,
        n_val_clips=1,
        n_frames=n_frames,
        face_size=face_size,
        seed=202,
    )

    cfg_path = REPO_ROOT / "configs" / "synthetic_dataset_only.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["video"]["face_size"] = face_size
    cfg["data"]["face_size"] = face_size
    cfg["data"]["lip_size"] = min(96, face_size)
    cfg["model"]["pretrained_identity"] = False
    cfg["losses"]["w_perceptual"] = 0.0
    cfg["losses"]["w_adv"] = 0.0
    cfg["losses"]["w_sync"] = 0.0
    cfg["losses"]["w_temporal"] = 0.0
    cfg["losses"]["w_mouth_alpha"] = 0.1
    cfg["fp16"] = False
    cfg["data"]["num_workers"] = 0
    cfg["optimizer"]["lr"] = 1.5e-3
    cfg.setdefault("gradient_clipping", {})["enabled"] = False
    cfg["checkpoint_dir"] = str(tmp_path / "ckpt")
    cfg["inference"]["mouth_composite_mode"] = "paste"
    cfg["inference"]["mouth_composite_scope"] = "lip_box"
    cfg["inference"]["lip_roi_feather_px"] = 0

    train = LipSyncDataset(
        root,
        split="train",
        augment=False,
        audio_window=16,
        face_size=face_size,
        lip_size=cfg["data"]["lip_size"],
    )
    assert len(train) == n_frames

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = LipSyncTrainerCore(cfg, device=str(dev))
    mc_kw = core._mouth_composite_kw()

    loader = DataLoader(
        train,
        batch_size=len(train),
        shuffle=False,
        num_workers=0,
        collate_fn=train.collate_fn,
    )
    batch = next(iter(loader))
    # Move batch to device inside _g_step — trainer uses self.device
    n_steps = 450 if dev.type == "cuda" else 550
    core.audio_encoder.train()
    core.identity_encoder.train()
    core.generator.train()
    core.discriminator.train()
    core.syncnet.train()
    for _ in range(n_steps):
        core._g_step(batch)

    mel = batch["mel"].to(core.device)
    face = batch["face"].to(core.device)
    ref_face = batch["ref_face"].to(core.device)
    gt = batch["gt_frame"].to(core.device)

    core.audio_encoder.eval()
    core.identity_encoder.eval()
    core.generator.eval()

    with torch.no_grad():
        masked_face = face.clone()
        masked_face[:, :, face.shape[-2] // 2 :, :] = 0.0
        masked_4ch = torch.cat([masked_face, torch.zeros_like(masked_face[:, :1])], dim=1)
        audio_seq = core.audio_encoder(mel)
        audio_emb = pool_audio_sequence(audio_seq, core._audio_pool_mode())
        identity_emb = core.identity_encoder(ref_face)
        rgb, alpha = core.generator(masked_4ch, audio_emb, identity_emb)
        pred = composite_mouth_region(
            face, rgb, alpha, core._mouth_blend_frac(), **mc_kw
        )

    pred_c = pred.cpu()
    gt_c = gt.cpu()
    _, _, H, W = pred_c.shape
    y0, y1, x0, x1 = mc_kw["lip_box_fracs"]
    sy, sx = lip_roi_slices(H, W, (y0, y1, x0, x1))
    l1_roi = (pred_c[:, :, sy, sx] - gt_c[:, :, sy, sx]).abs().mean().item()
    assert l1_roi < 0.18, f"expected mean |pred-gt| in lip ROI < 0.18 ([-1,1] scale), got {l1_roi:.4f}"

    pred_01 = (pred_c + 1.0) / 2.0
    gt_01 = (gt_c + 1.0) / 2.0
    mean_psnr = sum(
        psnr(pred_01[i : i + 1], gt_01[i : i + 1], max_val=1.0) for i in range(pred.shape[0])
    ) / pred.shape[0]
    assert mean_psnr >= 20.0, f"expected train-clip full-frame PSNR >= 20 dB, got {mean_psnr:.2f}"


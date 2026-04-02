"""Holdout reconstruction check: static (augmented) first frame + clip mel vs GT."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ..composite import (
    composite_mouth_region,
    lip_roi_slices,
    mouth_composite_kwargs_from_inference,
    pool_audio_sequence,
)
from ..data.datasets.lipsync_dataset import _load_image_rgb_norm
from ..data.preprocessing.audio import AudioPreprocessor
from ..inference.mux import extract_wav_from_video, mux_video_audio
from ..models import AudioEncoder, IdentityEncoder, LipSyncGenerator
from .metrics import psnr, ssim


def augment_ref_face(face_chw: torch.Tensor, seed: int) -> torch.Tensor:
    """Color + blur jitter on a (3,H,W) tensor in [-1, 1]."""
    torch.manual_seed(seed)
    random.seed(seed)
    x = ((face_chw + 1) / 2).clamp(0, 1).unsqueeze(0)
    try:
        from torchvision.transforms import v2 as T

        tfm = T.Compose(
            [
                T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.04),
                T.GaussianBlur(kernel_size=5, sigma=(0.15, 0.9)),
            ]
        )
        y = tfm(x).squeeze(0)
    except Exception:
        noise = torch.randn_like(face_chw) * 0.04
        y = ((face_chw + 1) / 2 + noise).clamp(0, 1)
    return y * 2 - 1


def _load_models(ckpt: dict[str, Any], device: torch.device) -> tuple[dict, float, str]:
    cfg = ckpt.get("config", {}).get("model", {})
    inf = ckpt.get("config", {}).get("inference", {})
    mouth_frac = float(inf.get("mouth_blend_from", 0.42))
    audio_pool = inf.get("audio_embed_pool", "last")

    audio_enc = AudioEncoder(
        n_mels=cfg.get("n_mels", 80),
        d_model=cfg.get("audio_d_model", 256),
        num_heads=cfg.get("audio_heads", 4),
        num_layers=cfg.get("audio_layers", 6),
        embed_dim=cfg.get("audio_embed_dim", 512),
    ).to(device)
    audio_enc.load_state_dict(ckpt["audio_encoder"])
    audio_enc.eval()

    id_enc = IdentityEncoder(
        embed_dim=cfg.get("identity_embed_dim", 512),
        pretrained=cfg.get("pretrained_identity", True),
    ).to(device)
    id_enc.load_state_dict(ckpt["identity_encoder"])
    id_enc.eval()

    gen = LipSyncGenerator(
        in_channels=cfg.get("gen_in_channels", 4),
        base_ch=cfg.get("gen_base_ch", 64),
        num_encoder_blocks=cfg.get("gen_depth", 4),
        audio_dim=cfg.get("audio_embed_dim", 512),
        identity_dim=cfg.get("identity_embed_dim", 512),
    ).to(device)
    gen.load_state_dict(ckpt["generator"])
    gen.eval()

    models = {"audio_encoder": audio_enc, "identity_encoder": id_enc, "generator": gen}
    return models, mouth_frac, audio_pool


def run_dataset_reconstruction_verify(
    checkpoint: str | Path,
    data_root: str | Path,
    *,
    split: str = "val",
    num_clips: int = 5,
    seed: int = 42,
    augment: bool = True,
    out_dir: str | Path = "verify_recon_out",
    device: str = "auto",
    batch_size: int = 16,
    mux_audio: bool = True,
) -> dict[str, Any]:
    """Run reconstruction on *split* clips only (not train). Writes MP4s + metrics under *out_dir*."""
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test' (train would inflate metrics).")
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = data_root / f"{split}_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}; use prepared data with a {split} split.")

    with open(meta_path, encoding="utf-8") as f:
        entries: list[dict[str, Any]] = json.load(f)
    if not entries:
        raise ValueError(f"{meta_path} is empty.")

    rng = random.Random(seed)
    pick = list(entries)
    rng.shuffle(pick)
    pick = pick[: min(num_clips, len(pick))]

    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
    if device == "auto" and not torch.cuda.is_available():
        dev = torch.device("cpu")

    try:
        ckpt = torch.load(str(checkpoint), map_location=dev, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(checkpoint), map_location=dev)
    models, mouth_frac, audio_pool = _load_models(ckpt, dev)
    inf_cfg = ckpt.get("config", {}).get("inference", {})
    mc_kw = mouth_composite_kwargs_from_inference(inf_cfg)
    audio_proc = AudioPreprocessor()
    cfg_window = int(ckpt.get("config", {}).get("audio", {}).get("window", 16))
    face_size = int(ckpt.get("config", {}).get("video", {}).get("face_size", 256))

    summary: dict[str, Any] = {
        "clips": [],
        "mean_l1": 0.0,
        "mean_l1_lip_roi": 0.0,
        "mean_psnr": 0.0,
        "mean_psnr_lip_roi": 0.0,
        "mean_ssim": 0.0,
    }
    n_ok = 0

    for entry in pick:
        sid = str(entry["id"])
        sample_dir = data_root / "samples" / sid
        frame_dir = sample_dir / "frames"
        paths = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
        if not paths:
            continue
        try:
            mel_full = torch.load(sample_dir / "audio.pt", map_location="cpu", weights_only=True)
        except TypeError:
            mel_full = torch.load(sample_dir / "audio.pt", map_location="cpu")
        fps = float(entry.get("fps", 25.0))

        n = len(paths)
        gt_faces = torch.stack([_load_image_rgb_norm(p, face_size) for p in paths])
        ref0 = _load_image_rgb_norm(paths[0], face_size)
        h = int(hashlib.md5(sid.encode(), usedforsecurity=False).hexdigest()[:8], 16)
        clip_seed = seed + (h % 100_000)
        ref_in = augment_ref_face(ref0, clip_seed) if augment else ref0.clone()

        mels = torch.stack(
            [audio_proc.mel_for_video_frame(mel_full, i, fps, cfg_window) for i in range(n)]
        )

        preds_list: list[torch.Tensor] = []
        identity_emb = models["identity_encoder"](ref_in.unsqueeze(0).to(dev))
        ref_stack = ref_in.unsqueeze(0).expand(n, -1, -1, -1).to(dev)

        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                face = ref_stack[start:end]
                mel = mels[start:end].to(dev)
                B = face.shape[0]
                audio_seq = models["audio_encoder"](mel)
                audio_emb = pool_audio_sequence(audio_seq, audio_pool)
                id_exp = identity_emb.expand(B, -1)
                masked = face.clone()
                masked[:, :, face.shape[-2] // 2 :, :] = 0.0
                masked_4ch = torch.cat([masked, torch.zeros_like(masked[:, :1])], dim=1)
                rgb, alpha = models["generator"](masked_4ch, audio_emb, id_exp)
                pred = composite_mouth_region(face, rgb, alpha, mouth_frac, **mc_kw)
                preds_list.append(pred.cpu())

        pred_all = torch.cat(preds_list, dim=0)
        l1 = (pred_all - gt_faces).abs().mean().item()
        _, _, H, W = pred_all.shape
        roi_l1 = l1
        if mc_kw.get("composite_scope") == "lip_box":
            y0, y1, x0, x1 = mc_kw["lip_box_fracs"]
            sy, sx = lip_roi_slices(H, W, (y0, y1, x0, x1))
            roi_l1 = (pred_all[:, :, sy, sx] - gt_faces[:, :, sy, sx]).abs().mean().item()
        pred_01 = (pred_all + 1) / 2
        gt_01 = (gt_faces + 1) / 2
        p = psnr(pred_01, gt_01, max_val=1.0)
        p_lip = p
        if mc_kw.get("composite_scope") == "lip_box":
            y0b, y1b, x0b, x1b = mc_kw["lip_box_fracs"]
            syl, sxl = lip_roi_slices(H, W, (y0b, y1b, x0b, x1b))
            pr = pred_01[:, :, syl, sxl]
            gr = gt_01[:, :, syl, sxl]
            p_lip = sum(
                psnr(pr[i : i + 1], gr[i : i + 1], max_val=1.0) for i in range(n)
            ) / max(n, 1)
        s = ssim(pred_01, gt_01, data_range=1.0)
        summary["clips"].append(
            {
                "id": sid,
                "l1": l1,
                "l1_lip_roi": roi_l1,
                "psnr": p,
                "psnr_lip_roi": p_lip,
                "ssim": s,
                "n_frames": n,
            }
        )
        summary["mean_l1"] += l1
        summary["mean_l1_lip_roi"] += roi_l1
        summary["mean_psnr"] += p
        summary["mean_psnr_lip_roi"] += p_lip
        summary["mean_ssim"] += s
        n_ok += 1

        video_path = out_dir / f"{sid}_pred.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (face_size, face_size))
        for i in range(n):
            fr = pred_all[i].permute(1, 2, 0).numpy()
            fr = ((fr + 1) * 127.5).clip(0, 255).astype(np.uint8)
            writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        writer.release()

        if mux_audio:
            src = Path(entry.get("source", ""))
            tmp_wav: Path | None = None
            try:
                if src.is_file():
                    tmp_wav = extract_wav_from_video(src)
                    mux_video_audio(video_path, tmp_wav, video_path)
                else:
                    summary["clips"][-1]["mux_note"] = "source missing; video has no muxed audio"
            except RuntimeError as e:
                summary["clips"][-1]["mux_note"] = str(e)
            finally:
                if tmp_wav is not None and tmp_wav.exists():
                    tmp_wav.unlink(missing_ok=True)

    if n_ok:
        summary["mean_l1"] /= n_ok
        summary["mean_l1_lip_roi"] /= n_ok
        summary["mean_psnr"] /= n_ok
        summary["mean_psnr_lip_roi"] /= n_ok
        summary["mean_ssim"] /= n_ok
    summary["n_evaluated"] = n_ok
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary

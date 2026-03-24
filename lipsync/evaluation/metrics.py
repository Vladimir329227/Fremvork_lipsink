"""Evaluation metrics: LMD, SSIM, PSNR, SyncScore, FID."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio.

    Args:
        pred: (B, C, H, W) tensor in [0, max_val].
        target: Same shape.
        max_val: Dynamic range of the signal.

    Returns:
        Average PSNR in dB across the batch.
    """
    mse = F.mse_loss(pred, target, reduction="mean").item()
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g.outer(g)
    return kernel / kernel.sum()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> float:
    """Structural Similarity Index (SSIM) between two batches.

    Args:
        pred: (B, C, H, W) in [0, data_range].
        target: Same shape.
        window_size: Gaussian window size.
        data_range: Maximum value of the signal.

    Returns:
        Mean SSIM across the batch.
    """
    C = pred.shape[1]
    kernel = _gaussian_kernel(window_size).to(pred.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
    pad = window_size // 2

    def _conv(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, padding=pad, groups=C)

    mu1 = _conv(pred)
    mu2 = _conv(target)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = _conv(pred * pred) - mu1_sq
    sigma2_sq = _conv(target * target) - mu2_sq
    sigma12 = _conv(pred * target) - mu1_mu2

    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return (num / den).mean().item()


# ---------------------------------------------------------------------------
# LMD — Landmark Distance
# ---------------------------------------------------------------------------

def landmark_distance(
    pred_landmarks: torch.Tensor,
    target_landmarks: torch.Tensor,
    normalise: bool = True,
) -> float:
    """Landmark Mean Distance (LMD).

    Measures mean Euclidean distance between predicted and ground-truth
    facial landmarks (68-point format). Optionally normalised by inter-ocular
    distance for scale invariance.

    Args:
        pred_landmarks: (B, 68, 2) predicted landmark coordinates.
        target_landmarks: (B, 68, 2) ground-truth coordinates.
        normalise: Divide by inter-ocular distance.

    Returns:
        Mean LMD across the batch.
    """
    dist = (pred_landmarks - target_landmarks).norm(dim=-1)  # (B, 68)

    if normalise:
        # Inter-ocular distance: distance between left (36) and right (45) eye corners
        left_eye = target_landmarks[:, 36]
        right_eye = target_landmarks[:, 45]
        iod = (left_eye - right_eye).norm(dim=-1, keepdim=True) + 1e-6  # (B, 1)
        dist = dist / iod

    return dist.mean().item()


def lip_landmark_distance(
    pred_landmarks: torch.Tensor,
    target_landmarks: torch.Tensor,
    normalise: bool = False,
) -> float:
    """LMD restricted to the lip region (landmarks 48–68).

    This is the most relevant metric for lip-sync quality.
    Normalisation is disabled by default since eye landmarks are not available
    in the lip-only subset; pass the full 68-point tensor with normalise=True
    if you want scale-invariant scores.
    """
    if pred_landmarks.shape[1] == 68:
        pred_lip = pred_landmarks[:, 48:68]
        target_lip = target_landmarks[:, 48:68]
    else:
        pred_lip = pred_landmarks
        target_lip = target_landmarks
    dist = (pred_lip - target_lip).norm(dim=-1)  # (B, n_lip)
    if normalise and pred_landmarks.shape[1] == 68:
        left_eye = target_landmarks[:, 36]
        right_eye = target_landmarks[:, 45]
        iod = (left_eye - right_eye).norm(dim=-1, keepdim=True) + 1e-6
        dist = dist / iod
    return dist.mean().item()


# ---------------------------------------------------------------------------
# SyncScore
# ---------------------------------------------------------------------------

def sync_score(
    audio_embeddings: torch.Tensor,
    video_embeddings: torch.Tensor,
) -> float:
    """Audio-visual synchronisation score via cosine similarity.

    Args:
        audio_embeddings: (B, D) normalised audio embeddings.
        video_embeddings: (B, D) normalised video embeddings.

    Returns:
        Mean cosine similarity in [-1, 1]; higher = more in sync.
    """
    a = F.normalize(audio_embeddings, dim=-1)
    v = F.normalize(video_embeddings, dim=-1)
    return (a * v).sum(dim=-1).mean().item()


# ---------------------------------------------------------------------------
# Aggregated evaluation
# ---------------------------------------------------------------------------

class LipSyncEvaluator:
    """Compute all metrics in one pass over a dataset.

    Args:
        syncnet: Pre-loaded SyncNet model for computing SyncScore.
        device: Computation device.
    """

    def __init__(self, syncnet=None, device: str = "cpu") -> None:
        self.syncnet = syncnet
        self.device = torch.device(device)

    @torch.no_grad()
    def evaluate(
        self,
        pred_frames: torch.Tensor,
        gt_frames: torch.Tensor,
        pred_landmarks: torch.Tensor | None = None,
        gt_landmarks: torch.Tensor | None = None,
        mel_windows: torch.Tensor | None = None,
        lip_crops: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Compute all available metrics.

        Args:
            pred_frames: (B, 3, H, W) generated frames in [-1, 1].
            gt_frames: (B, 3, H, W) ground-truth frames in [-1, 1].
            pred_landmarks: (B, 68, 2) optional predicted landmarks.
            gt_landmarks: (B, 68, 2) optional ground-truth landmarks.
            mel_windows: (B, 1, T, n_mels) optional audio for SyncScore.
            lip_crops: (B, T*3, lip_H, lip_W) optional lip crops for SyncScore.

        Returns:
            Dict of metric name → float value.
        """
        # Normalise to [0, 1] for PSNR/SSIM
        p = (pred_frames + 1.0) * 0.5
        g = (gt_frames + 1.0) * 0.5
        p = p.to(self.device)
        g = g.to(self.device)

        results: dict[str, float] = {
            "psnr": psnr(p, g),
            "ssim": ssim(p, g),
        }

        if pred_landmarks is not None and gt_landmarks is not None:
            results["lmd"] = landmark_distance(
                pred_landmarks.to(self.device),
                gt_landmarks.to(self.device),
            )
            results["lip_lmd"] = lip_landmark_distance(
                pred_landmarks.to(self.device),
                gt_landmarks.to(self.device),
            )

        if self.syncnet is not None and mel_windows is not None and lip_crops is not None:
            a_emb = self.syncnet.encode_audio(mel_windows.to(self.device))
            v_emb = self.syncnet.encode_video(lip_crops.to(self.device))
            results["sync_score"] = sync_score(a_emb, v_emb)

        return results

    def summarise(self, metrics_list: list[dict[str, float]]) -> dict[str, float]:
        """Average metrics over multiple batches."""
        if not metrics_list:
            return {}
        keys = metrics_list[0].keys()
        return {k: float(np.mean([m[k] for m in metrics_list if k in m])) for k in keys}

"""Lip-sync specific losses: Perceptual, Sync, Temporal, Identity, Adversarial."""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    """VGG19 feature-matching perceptual loss.

    Extracts intermediate feature maps from a frozen VGG19 and computes
    L1 distance between generated and real frame features.

    Args:
        layers: VGG layer names to use for feature extraction.
        weights: Per-layer weighting (uniform if None).
    """

    _LAYER_MAP = {
        "relu1_2": 4,
        "relu2_2": 9,
        "relu3_3": 18,
        "relu4_3": 27,
        "relu5_3": 36,
    }

    def __init__(
        self,
        layers: list[str] | None = None,
        weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        if layers is None:
            layers = ["relu3_3", "relu4_3"]
        self.layer_names = layers
        self.indices = [self._LAYER_MAP[l] for l in layers]
        self.weights = weights or [1.0] * len(layers)

        import torchvision.models as tv_models  # lazy import — requires torchvision
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
        features = vgg.features
        self.slices: nn.ModuleList = nn.ModuleList()
        prev = 0
        for idx in sorted(self.indices):
            self.slices.append(nn.Sequential(*list(features.children())[prev : idx + 1]))
            prev = idx + 1

        for p in self.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n = self._normalize(pred)
        target_n = self._normalize(target)
        loss = torch.tensor(0.0, device=pred.device)
        x_pred, x_target = pred_n, target_n
        for i, (slice_mod, w) in enumerate(zip(self.slices, self.weights)):
            x_pred = slice_mod(x_pred)
            x_target = slice_mod(x_target)
            loss = loss + w * F.l1_loss(x_pred, x_target.detach())
        return loss


class SyncLoss(nn.Module):
    """Audio-visual synchronisation loss via cosine similarity contrastive learning.

    A positive pair (audio_emb, video_emb) should have high cosine similarity;
    temporally misaligned pairs should have low similarity.

    Args:
        margin: Contrastive margin for negative pairs.
        temperature: Softmax temperature for NT-Xent variant.
    """

    def __init__(self, margin: float = 0.0, temperature: float = 0.07) -> None:
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(
        self,
        audio_emb: torch.Tensor,
        video_emb: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        audio_norm = F.normalize(audio_emb, dim=-1)
        video_norm = F.normalize(video_emb, dim=-1)
        sim = (audio_norm * video_norm).sum(dim=-1)  # (B,)

        if labels is None:
            # All pairs are positive (ground-truth alignment)
            return (1.0 - sim).mean()

        # BCE variant: labels = 1 for sync, 0 for out-of-sync
        logits = sim / self.temperature
        return F.binary_cross_entropy_with_logits(logits, labels.float())


class TemporalConsistencyLoss(nn.Module):
    """Penalises frame-to-frame flickering in generated sequences.

    Computes L1 difference between consecutive generated frames and
    optionally matches it to the difference in real frames (warping loss).

    Args:
        weight_real: If > 0, also match the real-frame temporal delta.
    """

    def __init__(self, weight_real: float = 0.0) -> None:
        super().__init__()
        self.weight_real = weight_real

    def forward(
        self,
        generated: torch.Tensor,
        real: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            generated: (B, T, C, H, W) sequence of generated frames.
            real: (B, T, C, H, W) real frames (optional).
        """
        diff_gen = (generated[:, 1:] - generated[:, :-1]).abs().mean()
        if real is not None and self.weight_real > 0:
            diff_real = (real[:, 1:] - real[:, :-1]).abs()
            diff_gen_seq = (generated[:, 1:] - generated[:, :-1]).abs()
            consistency = F.l1_loss(diff_gen_seq, diff_real.detach())
            return diff_gen + self.weight_real * consistency
        return diff_gen


class IdentityLoss(nn.Module):
    """Preserve facial identity using cosine distance between ArcFace embeddings.

    Expects pre-computed identity embeddings (e.g. from InsightFace / ArcFace).
    Call an external extractor before passing to this loss.
    """

    def forward(
        self, pred_emb: torch.Tensor, ref_emb: torch.Tensor
    ) -> torch.Tensor:
        pred_n = F.normalize(pred_emb, dim=-1)
        ref_n = F.normalize(ref_emb, dim=-1)
        cosine = (pred_n * ref_n).sum(dim=-1)
        return (1.0 - cosine).mean()


AdversarialMode = Literal["hinge", "lsgan", "bce", "wgan"]


class AdversarialLoss(nn.Module):
    """Unified adversarial loss supporting hinge, LSGAN, BCE and WGAN.

    Args:
        mode: Loss variant — 'hinge' | 'lsgan' | 'bce' | 'wgan'.
    """

    def __init__(self, mode: AdversarialMode = "hinge") -> None:
        super().__init__()
        self.mode = mode

    def _d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        if self.mode == "hinge":
            return (F.relu(1.0 - real_logits) + F.relu(1.0 + fake_logits)).mean()
        if self.mode == "lsgan":
            return 0.5 * ((real_logits - 1).pow(2).mean() + fake_logits.pow(2).mean())
        if self.mode == "bce":
            real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            return (real_loss + fake_loss) * 0.5
        # wgan
        return fake_logits.mean() - real_logits.mean()

    def _g_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        if self.mode == "hinge":
            return -fake_logits.mean()
        if self.mode == "lsgan":
            return (fake_logits - 1).pow(2).mean()
        if self.mode == "bce":
            return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
        # wgan
        return -fake_logits.mean()

    def discriminator_loss(
        self, real_logits: torch.Tensor, fake_logits: torch.Tensor
    ) -> torch.Tensor:
        return self._d_loss(real_logits, fake_logits)

    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        return self._g_loss(fake_logits)

    def forward(
        self,
        real_logits: torch.Tensor | None = None,
        fake_logits: torch.Tensor | None = None,
        mode: Literal["D", "G"] = "G",
    ) -> torch.Tensor:
        if mode == "D":
            assert real_logits is not None and fake_logits is not None
            return self._d_loss(real_logits, fake_logits)
        assert fake_logits is not None
        return self._g_loss(fake_logits)

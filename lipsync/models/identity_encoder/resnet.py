"""Identity encoder: reference face frame → appearance embedding."""
from __future__ import annotations

import torch
import torch.nn as nn


class IdentityEncoder(nn.Module):
    """ResNet-18 based identity encoder for appearance feature extraction.

    Extracts a compact identity vector from a reference face crop.
    Optionally pre-trains on face verification tasks (e.g. VGGFace2).

    Args:
        embed_dim: Output embedding dimension.
        pretrained: Load ImageNet pre-trained weights as initialisation.
        freeze_backbone: Freeze all layers except the projection head.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        import torchvision.models as tv_models  # lazy import
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        # Remove the final classification head
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # (B, 512, 1, 1)
        backbone_dim = 512

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, face: torch.Tensor) -> torch.Tensor:
        """
        Args:
            face: (B, 3, H, W) normalised face crop (typically 112×112 or 224×224).

        Returns:
            (B, embed_dim) identity embedding.
        """
        feat = self.backbone(face)
        return self.proj(feat)

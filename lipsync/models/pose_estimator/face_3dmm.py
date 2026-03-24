"""3D Morphable Model (3DMM) based head pose estimator.

Decouples facial parameters into:
- shape   (identity, unchanged during generation)
- expression (jaw / lip motion — modified by generator)
- pose    (head rotation/translation — preserved from source video)
- texture  (appearance — preserved from source)

This separation is the key mechanism for correct head rotation handling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FaceParams:
    """Container for decomposed 3DMM parameters.

    All tensors have batch dimension B.
    """
    shape: torch.Tensor        # (B, n_shape)  — identity basis coefficients
    expression: torch.Tensor   # (B, n_expr)   — expression/jaw coefficients
    pose: torch.Tensor         # (B, 6)        — rotation (3) + translation (3)
    texture: torch.Tensor      # (B, n_tex)    — texture basis coefficients

    def clone(self) -> "FaceParams":
        return FaceParams(
            shape=self.shape.clone(),
            expression=self.expression.clone(),
            pose=self.pose.clone(),
            texture=self.texture.clone(),
        )


class PoseEstimator(nn.Module):
    """Lightweight CNN that regresses 3DMM parameters from a face crop.

    Uses a MobileNetV3-Small backbone for real-time performance.

    Args:
        n_shape: Number of shape basis vectors.
        n_expr: Number of expression basis vectors.
        n_tex: Number of texture basis vectors.
        pretrained: Use ImageNet weights for the backbone.
    """

    def __init__(
        self,
        n_shape: int = 80,
        n_expr: int = 64,
        n_tex: int = 80,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.n_shape = n_shape
        self.n_expr = n_expr
        self.n_tex = n_tex

        import torchvision.models as tv_models  # lazy import
        weights = (
            tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        backbone = tv_models.mobilenet_v3_small(weights=weights)
        # Replace classifier with identity
        backbone_out = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        out_dim = n_shape + n_expr + 6 + n_tex
        self.head = nn.Sequential(
            nn.Linear(backbone_out, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, face: torch.Tensor) -> FaceParams:
        """
        Args:
            face: (B, 3, 224, 224) normalised face crops.

        Returns:
            FaceParams with decoded shape/expression/pose/texture tensors.
        """
        feat = self.backbone(face)
        params = self.head(feat)

        idx = 0
        shape = params[:, idx : idx + self.n_shape]; idx += self.n_shape
        expr = params[:, idx : idx + self.n_expr]; idx += self.n_expr
        pose = params[:, idx : idx + 6]; idx += 6
        texture = params[:, idx : idx + self.n_tex]

        return FaceParams(shape=shape, expression=expr, pose=pose, texture=texture)

    @staticmethod
    def swap_expression(source: FaceParams, new_expr: torch.Tensor) -> FaceParams:
        """Create a new FaceParams with *new_expr* but all other params from *source*.

        This is the core operation for lip-sync: replace expression params
        while preserving pose, shape, and texture from the original video.
        """
        result = source.clone()
        result.expression = new_expr
        return result

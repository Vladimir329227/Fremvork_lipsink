"""PatchGAN discriminator for local realism."""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """70×70 PatchGAN discriminator.

    Classifies overlapping image patches as real or fake, encouraging
    high-frequency sharpness and local texture realism.

    Args:
        in_channels: Input channels (3 for RGB, 6 for concatenated real+generated).
        base_ch: Feature map base width.
        n_layers: Number of strided conv layers (controls receptive field size).
        use_spectral_norm: Apply spectral normalisation for training stability.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()

        def _sn(m: nn.Module) -> nn.Module:
            return nn.utils.spectral_norm(m) if use_spectral_norm else m

        layers: list[nn.Module] = [
            _sn(nn.Conv2d(in_channels, base_ch, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        ch = base_ch
        for i in range(1, n_layers):
            next_ch = min(ch * 2, 512)
            layers += [
                _sn(nn.Conv2d(ch, next_ch, 4, stride=2, padding=1, bias=False)),
                nn.InstanceNorm2d(next_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = next_ch

        next_ch = min(ch * 2, 512)
        layers += [
            _sn(nn.Conv2d(ch, next_ch, 4, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(next_ch),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv2d(next_ch, 1, 4, stride=1, padding=1)),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) image tensor.

        Returns:
            (B, 1, H', W') patch logit map.
        """
        return self.model(x)

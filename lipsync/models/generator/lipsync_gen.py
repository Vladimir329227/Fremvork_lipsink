"""LipSync Generator: fuses audio embedding + identity → lip-region patch."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import CrossAttention2D, ResBlock, UpBlock


class LipSyncGenerator(nn.Module):
    """Audio-conditioned generator that synthesises the lip/jaw region.

    Architecture:
    1. Encoder path: strided ResBlocks downsample the masked face crop.
    2. Bottleneck: cross-attention with fused audio + identity context.
    3. Decoder path: UpBlocks with correctly aligned U-Net skip connections.
    4. Output head: tanh RGB patch + sigmoid alpha mask.

    Skip connection alignment (standard U-Net):
    - Encoder at depth D produces feature maps at resolution R/2^D.
    - Decoder going from R/2^D → R/2^(D-1) cats the encoder output at R/2^(D-1),
      i.e. one level shallower than the current decoder input.
    - The outermost decoder (going back to input resolution) has no skip.

    Args:
        in_channels: Input face crop channels (3=RGB, 4=RGB+mask).
        base_ch: Feature-map width at the shallowest encoder level.
        num_encoder_blocks: Encoder / decoder depth.
        audio_dim: Audio embedding dimension.
        identity_dim: Identity embedding dimension.
        output_channels: Channels of generated output (3 = RGB).
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_ch: int = 64,
        num_encoder_blocks: int = 4,
        audio_dim: int = 512,
        identity_dim: int = 512,
        output_channels: int = 3,
    ) -> None:
        super().__init__()
        self.num_levels = num_encoder_blocks
        context_dim = audio_dim + identity_dim

        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )

        # encoder_chs[k] = channels output by encoder block k
        encoder_chs = [in_channels] + [
            min(base_ch * (2 ** i), 512) for i in range(num_encoder_blocks)
        ]
        self.encoders = nn.ModuleList(
            ResBlock(encoder_chs[i], encoder_chs[i + 1], stride=2)
            for i in range(num_encoder_blocks)
        )

        bottle_ch = encoder_chs[-1]
        self.bottleneck = nn.Sequential(
            ResBlock(bottle_ch, bottle_ch),
            CrossAttention2D(bottle_ch, context_dim),
            ResBlock(bottle_ch, bottle_ch),
        )

        # Build decoders with correct skip-channel alignment:
        # Decoder enum index k (0 = deepest):
        #   in_ch  = encoder_chs[depth - k]        (current input channels)
        #   skip   = encoder output at depth-2-k    (one level shallower)
        #   skip_ch= encoder_chs[depth-1-k]         (0 for outermost decoder)
        #   out_ch = skip_ch or base_ch
        D = num_encoder_blocks
        self.decoders = nn.ModuleList()
        for k in range(D):
            in_ch = encoder_chs[D - k]
            if k < D - 1:
                skip_ch = encoder_chs[D - 1 - k]
                out_ch = max(skip_ch, base_ch)
            else:
                skip_ch = 0        # outermost — no encoder output at input resolution
                out_ch = base_ch
            self.decoders.append(UpBlock(in_ch, skip_ch, out_ch, context_dim=context_dim))

        final_ch = base_ch
        self.out_head = nn.Sequential(
            nn.Conv2d(final_ch, max(final_ch // 2, 8), 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(max(final_ch // 2, 8), output_channels + 1, 1),
        )

    def forward(
        self,
        face_crop: torch.Tensor,
        audio_emb: torch.Tensor,
        identity_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            face_crop: (B, in_channels, H, W) — lower half zeroed as mouth mask.
            audio_emb: (B, audio_dim) or (B, 1, audio_dim).
            identity_emb: (B, identity_dim).

        Returns:
            rgb: (B, 3, H, W) in [-1, 1].
            alpha: (B, 1, H, W) in [0, 1].
        """
        if audio_emb.dim() == 3:
            audio_emb = audio_emb.squeeze(1)

        context = self.context_proj(torch.cat([audio_emb, identity_emb], dim=-1))

        # Encoder — save outputs as skip-connection sources
        skips: list[torch.Tensor] = []
        x = face_crop
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck[0](x)
        x = self.bottleneck[1](x, context)
        x = self.bottleneck[2](x)

        # Decoder — skip[k] = encoder output one level shallower than current input
        D = self.num_levels
        for k, dec in enumerate(self.decoders):
            skip_idx = D - 2 - k   # e.g. depth=3: k=0→idx=1, k=1→idx=0, k=2→idx=-1
            skip = skips[skip_idx] if skip_idx >= 0 else None
            x = dec(x, skip, context)

        out = self.out_head(x)
        rgb = torch.tanh(out[:, :3])
        alpha = torch.sigmoid(out[:, 3:4])
        return rgb, alpha

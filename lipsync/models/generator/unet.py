"""U-Net building blocks with optional cross-attention for audio conditioning."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with two 3×3 convolutions and optional downsampling."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        norm: type[nn.Module] = nn.InstanceNorm2d,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            norm(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            norm(out_ch),
        )
        self.skip = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), norm(out_ch))
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class CrossAttention2D(nn.Module):
    """Spatial cross-attention: image features attend to a conditioning vector.

    Args:
        spatial_dim: Number of channels in the spatial feature map.
        context_dim: Dimensionality of the conditioning context vector.
        num_heads: Number of attention heads.
    """

    def __init__(self, spatial_dim: int, context_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, spatial_dim), spatial_dim)
        self.q_proj = nn.Conv2d(spatial_dim, spatial_dim, 1)
        self.k_proj = nn.Linear(context_dim, spatial_dim)
        self.v_proj = nn.Linear(context_dim, spatial_dim)
        self.out_proj = nn.Conv2d(spatial_dim, spatial_dim, 1)
        self.num_heads = num_heads
        self.scale = (spatial_dim // num_heads) ** -0.5

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) spatial feature map.
            context: (B, context_dim) or (B, L, context_dim) conditioning.
        """
        B, C, H, W = x.shape
        if context.dim() == 2:
            context = context.unsqueeze(1)  # (B, 1, context_dim)

        residual = x
        x = self.norm(x)
        q = self.q_proj(x).view(B, self.num_heads, C // self.num_heads, H * W)
        q = q.permute(0, 1, 3, 2)  # (B, heads, HW, d_head)

        k = self.k_proj(context)  # (B, L, C)
        v = self.v_proj(context)  # (B, L, C)
        k = k.view(B, self.num_heads, -1, C // self.num_heads).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, -1, C // self.num_heads)  # (B, heads, L, d)

        attn = torch.einsum("bhid,bhjd->bhij", q, k.permute(0, 1, 3, 2)) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)  # (B, heads, HW, d)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        out = self.out_proj(out)
        return residual + out


class UpBlock(nn.Module):
    """U-Net decoder block: upsample + optional skip connection + ResBlock.

    Args:
        in_ch: Channels of the incoming feature map (before upsample).
        skip_ch: Channels of the encoder skip-connection tensor (0 = no skip).
        out_ch: Output channels after ResBlock.
        context_dim: If given, applies CrossAttention2D with this context dim.
    """

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.skip_ch = skip_ch
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.res = ResBlock(in_ch + skip_ch, out_ch)
        self.cross_attn = CrossAttention2D(out_ch, context_dim) if context_dim else None

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.up(x)
        if skip is not None and self.skip_ch > 0:
            x = torch.cat([x, skip], dim=1)
        x = self.res(x)
        if self.cross_attn is not None and context is not None:
            x = self.cross_attn(x, context)
        return x

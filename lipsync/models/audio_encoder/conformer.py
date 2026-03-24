"""Audio Encoder: mel-spectrogram → Conformer → dense audio embedding."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSubsampling(nn.Module):
    """2× time-axis downsampling via two strided convolutions."""

    def __init__(self, in_channels: int, out_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Linear(32 * ((in_channels + 3) // 4), out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, mel)
        x = x.unsqueeze(1)  # (B, 1, T, mel)
        x = self.conv(x)    # (B, 32, T//4, mel//4)
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * F)
        return self.proj(x)  # (B, T//4, d_model)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        return residual + self.dropout(x)


class ConvolutionModule(nn.Module):
    """Depthwise convolution sub-block of a Conformer block."""

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1) -> None:
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        padding = (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, 1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_out = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)      # (B, d, T)
        x = self.pointwise_in(x)   # (B, 2d, T)
        x = self.glu(x)            # (B, d, T)
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pointwise_out(x)
        x = self.dropout(x)
        return residual + x.transpose(1, 2)


class ConformerBlock(nn.Module):
    """Single Conformer block: FF → MHSA → Conv → FF (sandwich)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        conv_kernel: int = 31,
        ff_expansion: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ff1 = FeedForward(d_model, ff_expansion, dropout)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForward(d_model, ff_expansion, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = self.attn(x)
        x = self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class AudioEncoder(nn.Module):
    """Mel-spectrogram → Conformer stack → per-frame audio embedding.

    Args:
        n_mels: Number of mel frequency bins (default 80).
        d_model: Internal hidden dimension.
        num_heads: Attention heads per Conformer block.
        num_layers: Number of Conformer blocks.
        embed_dim: Output embedding dimension (projected from d_model).
        conv_kernel: Depthwise conv kernel size in Conformer.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 6,
        embed_dim: int = 512,
        conv_kernel: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.subsampling = ConvSubsampling(n_mels, d_model)
        self.blocks = nn.ModuleList(
            [ConformerBlock(d_model, num_heads, conv_kernel, dropout=dropout) for _ in range(num_layers)]
        )
        self.proj = nn.Linear(d_model, embed_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, T, n_mels) mel-spectrogram frames.

        Returns:
            (B, T', embed_dim) audio embeddings, T' ≈ T/4 due to subsampling.
        """
        x = self.subsampling(mel)
        for block in self.blocks:
            x = block(x)
        return self.proj(x)

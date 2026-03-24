"""SyncNet: audio-visual synchronisation discriminator."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SyncNet(nn.Module):
    """Dual-encoder SyncNet for audio-visual lip synchronisation scoring.

    Encodes a short audio segment and the corresponding lip-crop frames into
    separate embedding spaces and measures cosine similarity.
    A pre-trained SyncNet can be used as a frozen perceptual sync loss.

    Args:
        audio_in_channels: Audio input channels (1 for raw mel, typically 1×T×80).
        video_in_channels: Number of lip-crop frames stacked along channel dim.
        embed_dim: Shared embedding dimension.
    """

    def __init__(
        self,
        audio_in_channels: int = 1,
        video_in_channels: int = 15,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()

        self.audio_encoder = nn.Sequential(
            nn.Conv2d(audio_in_channels, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, (3, 3), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, (3, 3), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, (3, 3), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, embed_dim),
        )

        self.video_encoder = nn.Sequential(
            nn.Conv2d(video_in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, embed_dim),
        )

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: (B, 1, T, n_mels) mel-spectrogram window.
        """
        return F.normalize(self.audio_encoder(audio), dim=-1)

    def encode_video(self, lip_crops: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lip_crops: (B, T*3, H, W) stacked RGB lip-crop frames (5 frames = 15 ch).
        """
        return F.normalize(self.video_encoder(lip_crops), dim=-1)

    def forward(
        self, audio: torch.Tensor, lip_crops: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (audio_emb, video_emb) for use in SyncLoss."""
        return self.encode_audio(audio), self.encode_video(lip_crops)

    def similarity(self, audio: torch.Tensor, lip_crops: torch.Tensor) -> torch.Tensor:
        """Cosine similarity score in [−1, 1]; higher = more in sync."""
        a, v = self.forward(audio, lip_crops)
        return (a * v).sum(dim=-1)

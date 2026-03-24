"""Batch inference processor for high-quality offline lip-sync generation."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from ...data.datasets.lipsync_dataset import VideoDataset
from ...data.preprocessing.video import VideoPreprocessor


class BatchProcessor:
    """Process a video file offline with a trained checkpoint.

    Generates all frames at once (or in batches) and writes output video.

    Args:
        checkpoint_path: Path to trained model weights.
        device: 'cuda' | 'cpu' | 'auto'.
        batch_size: Number of frames processed per forward pass.
        use_sr: Apply super-resolution to each output frame.
        sr_backend: SR backend name.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        batch_size: int = 16,
        use_sr: bool = False,
        sr_backend: str = "gfpgan",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.batch_size = batch_size
        self.use_sr = use_sr
        self.device = self._resolve_device(device)
        self._models: dict | None = None
        self._sr = None
        if use_sr:
            from ...models.super_resolution import SuperResolutionWrapper
            self._sr = SuperResolutionWrapper(backend=sr_backend, device=str(self.device))

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_models(self) -> None:
        if self._models is not None:
            return
        from ...models import AudioEncoder, IdentityEncoder, LipSyncGenerator

        ckpt = torch.load(str(self.checkpoint_path), map_location=self.device)
        cfg = ckpt.get("config", {}).get("model", {})

        audio_enc = AudioEncoder(embed_dim=cfg.get("audio_embed_dim", 512)).to(self.device)
        audio_enc.load_state_dict(ckpt["audio_encoder"])
        audio_enc.eval()

        id_enc = IdentityEncoder(embed_dim=cfg.get("identity_embed_dim", 512)).to(self.device)
        id_enc.load_state_dict(ckpt["identity_encoder"])
        id_enc.eval()

        gen = LipSyncGenerator(
            audio_dim=cfg.get("audio_embed_dim", 512),
            identity_dim=cfg.get("identity_embed_dim", 512),
        ).to(self.device)
        gen.load_state_dict(ckpt["generator"])
        gen.eval()

        self._models = {"audio_encoder": audio_enc, "identity_encoder": id_enc, "generator": gen}

    @torch.no_grad()
    def process(
        self,
        video_path: str | Path,
        audio_path: str | Path | None = None,
        output_path: str | Path = "output.mp4",
        fps: float = 25.0,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Generate lip-synced video.

        Args:
            video_path: Source video (provides identity and pose).
            audio_path: Driving audio (WAV). Uses video audio if None.
            output_path: Where to write the result MP4.
            fps: Output frame rate.
            progress_callback: Called with (frame_idx, total_frames).

        Returns:
            Path to the output file.
        """
        import cv2

        self._load_models()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = VideoDataset(
            video_path=video_path,
            audio_path=audio_path,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        # Determine output size from first frame
        sample = dataset[0]
        H = W = sample["face"].shape[-1]

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H),
        )

        # Use first frame as persistent identity reference
        ref_face = sample["ref_face"].unsqueeze(0).to(self.device)
        identity_emb = self._models["identity_encoder"](ref_face)

        total = len(dataset)
        processed = 0
        for batch in loader:
            mel = batch["mel"].to(self.device)
            face = batch["face"].to(self.device)
            B = face.shape[0]

            audio_emb = self._models["audio_encoder"](mel)[:, -1]
            id_expanded = identity_emb.expand(B, -1)

            masked_face = face.clone()
            masked_face[:, :, face.shape[-2] // 2 :, :] = 0.0
            masked_face_4ch = torch.cat([masked_face, torch.zeros_like(masked_face[:, :1])], dim=1)

            rgb, alpha = self._models["generator"](masked_face_4ch, audio_emb, id_expanded)
            generated = alpha * rgb + (1 - alpha) * face  # (B, 3, H, W)

            for i in range(B):
                frame_np = (
                    (generated[i].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
                ).clip(0, 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                if self.use_sr and self._sr is not None:
                    frame_bgr = self._sr.enhance(frame_bgr)
                writer.write(frame_bgr)

            processed += B
            if progress_callback:
                progress_callback(processed, total)

        writer.release()
        print(f"Output saved to {output_path}")
        return output_path

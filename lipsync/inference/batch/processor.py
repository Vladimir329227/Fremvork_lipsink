"""Batch inference processor for high-quality offline lip-sync generation."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from ...composite import _DEFAULT_LIP_BOX, composite_mouth_region, mouth_composite_kwargs_from_inference
from ...data.datasets.lipsync_dataset import VideoDataset


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
        inference_overrides: dict | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.batch_size = batch_size
        self.use_sr = use_sr
        self.device = self._resolve_device(device)
        # Merge over checkpoint ``inference`` (e.g. force blend instead of paste).
        self._inference_overrides: dict = dict(inference_overrides or {})
        self._models: dict | None = None
        self._mouth_frac: float = 0.42
        self._audio_pool: str = "last"
        self._mux_driving_audio: bool = True
        self._mouth_alpha_floor: float | None = 0.45
        self._mouth_composite_mode: str = "blend"
        self._mouth_composite_scope: str = "lip_box"
        self._lip_box_fracs: tuple[float, float, float, float] = _DEFAULT_LIP_BOX
        self._lip_roi_feather_px: int = 0
        self._sr = None
        if use_sr:
            from ...models.super_resolution import SuperResolutionWrapper
            self._sr = SuperResolutionWrapper(backend=sr_backend, device=str(self.device))

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _apply_inference_config(self, inf: dict) -> None:
        """Mouth compositing + mux flags from checkpoint ``inference`` block.

        Composite kwargs must match :func:`mouth_composite_kwargs_from_inference` (same as
        training) so missing keys default to **blend**, not raw **paste** (weak G → noise box).
        """
        inf = {**inf, **self._inference_overrides}
        self._mouth_frac = float(inf.get("mouth_blend_from", 0.42))
        self._audio_pool = inf.get("audio_embed_pool", "last")
        self._mux_driving_audio = bool(inf.get("mux_driving_audio", True))

        kw = mouth_composite_kwargs_from_inference(inf)
        self._mouth_composite_mode = kw["composite_mode"]
        self._mouth_composite_scope = kw["composite_scope"]
        self._lip_box_fracs = kw["lip_box_fracs"]
        self._lip_roi_feather_px = kw["lip_roi_feather_px"]
        self._mouth_alpha_floor = kw["alpha_floor"]

    def _load_inference_from_checkpoint_file(self) -> None:
        from ...composite import pool_audio_sequence

        self._pool_audio_sequence = pool_audio_sequence
        try:
            try:
                ckpt = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(str(self.checkpoint_path), map_location="cpu")
            inf = ckpt.get("config", {}).get("inference", {})
            self._apply_inference_config(inf)
        except Exception:
            self._apply_inference_config({})

    def _ensure_composite_settings(self) -> None:
        """Reload inference flags from checkpoint (e.g. after injecting ``_models``)."""
        self._load_inference_from_checkpoint_file()

    def _load_models(self) -> None:
        if self._models is not None:
            self._load_inference_from_checkpoint_file()
            return
        from ...models import AudioEncoder, IdentityEncoder, LipSyncGenerator

        try:
            ckpt = torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(str(self.checkpoint_path), map_location=self.device)
        cfg = ckpt.get("config", {}).get("model", {})
        inf = ckpt.get("config", {}).get("inference", {})
        from ...composite import pool_audio_sequence

        self._pool_audio_sequence = pool_audio_sequence
        self._apply_inference_config(inf)

        audio_enc = AudioEncoder(
            n_mels=cfg.get("n_mels", 80),
            d_model=cfg.get("audio_d_model", 256),
            num_heads=cfg.get("audio_heads", 4),
            num_layers=cfg.get("audio_layers", 6),
            embed_dim=cfg.get("audio_embed_dim", 512),
        ).to(self.device)
        audio_enc.load_state_dict(ckpt["audio_encoder"])
        audio_enc.eval()

        id_enc = IdentityEncoder(
            embed_dim=cfg.get("identity_embed_dim", 512),
            pretrained=cfg.get("pretrained_identity", True),
        ).to(self.device)
        id_enc.load_state_dict(ckpt["identity_encoder"])
        id_enc.eval()

        gen = LipSyncGenerator(
            in_channels=cfg.get("gen_in_channels", 4),
            base_ch=cfg.get("gen_base_ch", 64),
            num_encoder_blocks=cfg.get("gen_depth", 4),
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

            audio_seq = self._models["audio_encoder"](mel)
            audio_emb = self._pool_audio_sequence(audio_seq, self._audio_pool)
            id_expanded = identity_emb.expand(B, -1)

            masked_face = face.clone()
            masked_face[:, :, face.shape[-2] // 2 :, :] = 0.0
            masked_face_4ch = torch.cat([masked_face, torch.zeros_like(masked_face[:, :1])], dim=1)

            rgb, alpha = self._models["generator"](masked_face_4ch, audio_emb, id_expanded)
            generated = composite_mouth_region(
                face,
                rgb,
                alpha,
                self._mouth_frac,
                alpha_floor=self._mouth_alpha_floor,
                composite_mode=self._mouth_composite_mode,
                composite_scope=self._mouth_composite_scope,
                lip_box_fracs=self._lip_box_fracs,
                lip_roi_feather_px=self._lip_roi_feather_px,
            )

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

        if self._mux_driving_audio:
            from ..mux import extract_wav_from_video, mux_video_audio

            wav_path: Path | None = None
            tmp_wav: Path | None = None
            try:
                if audio_path is not None and Path(audio_path).exists():
                    wav_path = Path(audio_path)
                else:
                    tmp_wav = extract_wav_from_video(video_path)
                    wav_path = tmp_wav
                mux_video_audio(output_path, wav_path, output_path)
            except RuntimeError as e:
                print(f"Audio mux skipped: {e}")
            finally:
                if tmp_wav is not None and tmp_wav.exists():
                    tmp_wav.unlink(missing_ok=True)

        print(f"Output saved to {output_path}")
        return output_path

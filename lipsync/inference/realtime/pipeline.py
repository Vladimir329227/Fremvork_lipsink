"""Real-time lip-sync inference pipeline with sliding audio window.

Target: < 100 ms end-to-end latency at 25 fps on a modern GPU.

Architecture:
- Audio stream is read into a ring buffer.
- Each video frame triggers extraction of the corresponding mel window.
- Generator runs with torch.no_grad() + optional TorchScript/ONNX.
- Super-resolution is applied if enabled.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from ...data.preprocessing.audio import AudioPreprocessor
from ...data.preprocessing.video import VideoPreprocessor


class AudioRingBuffer:
    """Thread-safe ring buffer for streaming audio samples."""

    def __init__(self, max_seconds: float = 2.0, sample_rate: int = 16000) -> None:
        self._max = int(max_seconds * sample_rate)
        self._buf: deque[float] = deque(maxlen=self._max)
        self._lock = threading.Lock()
        self.sample_rate = sample_rate

    def push(self, samples: np.ndarray) -> None:
        with self._lock:
            self._buf.extend(samples.flatten().tolist())

    def get_latest(self, n_samples: int) -> np.ndarray:
        with self._lock:
            buf = list(self._buf)
        if len(buf) < n_samples:
            buf = [0.0] * (n_samples - len(buf)) + buf
        return np.array(buf[-n_samples:], dtype=np.float32)

    def __len__(self) -> int:
        return len(self._buf)


class RealTimePipeline:
    """Real-time lip-sync inference engine.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        device: 'cuda' | 'cpu' | 'auto'.
        audio_window_ms: Audio context window in milliseconds.
        use_sr: Apply super-resolution on the output face patch.
        sr_backend: SR backend ('gfpgan' | 'codeformer' | 'none').
        fps: Target frames per second.
        onnx_path: If provided, use ONNX runtime instead of PyTorch.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        audio_window_ms: float = 200.0,
        use_sr: bool = False,
        sr_backend: str = "gfpgan",
        fps: float = 25.0,
        onnx_path: str | Path | None = None,
    ) -> None:
        self.device = self._resolve_device(device)
        self.fps = fps
        self.use_sr = use_sr
        self.onnx_path = onnx_path

        self.audio_proc = AudioPreprocessor()
        self.video_proc = VideoPreprocessor(target_fps=fps)

        # Calculate mel window size from milliseconds
        self.audio_window_frames = int(
            audio_window_ms / 1000.0
            * self.audio_proc.sample_rate
            / self.audio_proc.hop_length
        )

        self.audio_buffer = AudioRingBuffer(sample_rate=self.audio_proc.sample_rate)
        self._models = None
        self._ort_session = None
        self._checkpoint_path = Path(checkpoint_path)
        self._sr = None
        self._ref_identity: torch.Tensor | None = None

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

        ckpt = torch.load(str(self._checkpoint_path), map_location=self.device)
        cfg = ckpt.get("config", {}).get("model", {})

        audio_enc = AudioEncoder(
            n_mels=cfg.get("n_mels", 80),
            embed_dim=cfg.get("audio_embed_dim", 512),
        ).to(self.device)
        audio_enc.load_state_dict(ckpt["audio_encoder"])
        audio_enc.eval()

        id_enc = IdentityEncoder(
            embed_dim=cfg.get("identity_embed_dim", 512),
        ).to(self.device)
        id_enc.load_state_dict(ckpt["identity_encoder"])
        id_enc.eval()

        gen = LipSyncGenerator(
            audio_dim=cfg.get("audio_embed_dim", 512),
            identity_dim=cfg.get("identity_embed_dim", 512),
        ).to(self.device)
        gen.load_state_dict(ckpt["generator"])
        gen.eval()

        self._models = {
            "audio_encoder": audio_enc,
            "identity_encoder": id_enc,
            "generator": gen,
        }

    def set_reference_frame(self, frame_bgr: np.ndarray) -> None:
        """Pre-compute identity embedding from a reference face frame."""
        self._load_models()
        face_tensor = VideoPreprocessor.frame_to_tensor(
            cv2_resize(frame_bgr, 256)
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self._ref_identity = self._models["identity_encoder"](face_tensor)

    def push_audio(self, pcm_samples: np.ndarray) -> None:
        """Feed raw PCM samples (float32, mono) into the audio ring buffer."""
        self.audio_buffer.push(pcm_samples)

    @torch.no_grad()
    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Process one video frame and return the lip-synced output.

        Args:
            frame_bgr: (H, W, 3) uint8 BGR frame from camera or video.

        Returns:
            (H, W, 3) uint8 BGR frame with lip motion applied.
        """
        self._load_models()

        # Detect face and get landmarks
        landmarks = self.video_proc.detect_landmarks(frame_bgr)
        if landmarks is None:
            return frame_bgr

        face_crop = VideoPreprocessor.crop_face(frame_bgr, landmarks, 256)
        face_tensor = VideoPreprocessor.frame_to_tensor(face_crop).unsqueeze(0).to(self.device)

        # Get audio mel window
        n_samples = self.audio_window_frames * self.audio_proc.hop_length
        raw = self.audio_buffer.get_latest(n_samples)
        waveform = torch.from_numpy(raw).unsqueeze(0)
        mel = self.audio_proc.waveform_to_mel(waveform)  # (T, n_mels)
        if mel.shape[0] < self.audio_window_frames:
            pad = torch.zeros(self.audio_window_frames - mel.shape[0], mel.shape[1])
            mel = torch.cat([pad, mel], dim=0)
        mel = mel[-self.audio_window_frames:].unsqueeze(0).to(self.device)  # (1, T, n_mels)

        # Identity embedding
        if self._ref_identity is None:
            self._ref_identity = self._models["identity_encoder"](face_tensor)

        # Generate
        audio_emb = self._models["audio_encoder"](mel)[:, -1]
        masked_face = face_tensor.clone()
        masked_face[:, :, face_tensor.shape[-2] // 2 :, :] = 0.0
        masked_face_4ch = torch.cat([masked_face, torch.zeros_like(masked_face[:, :1])], dim=1)
        rgb, alpha = self._models["generator"](masked_face_4ch, audio_emb, self._ref_identity)
        generated = (alpha * rgb + (1 - alpha) * face_tensor).squeeze(0)

        # Back to numpy BGR
        out_np = ((generated.permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        import cv2
        out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

        if self.use_sr and self._sr is not None:
            out_bgr = self._sr.enhance(out_bgr)

        # Paste back into original frame
        result = frame_bgr.copy()
        import cv2 as cv
        x1, y1 = int(landmarks[:, 0].min()), int(landmarks[:, 1].min())
        x2, y2 = int(landmarks[:, 0].max()), int(landmarks[:, 1].max())
        margin = int((x2 - x1) * 0.15)
        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2, y2 = min(result.shape[1], x2 + margin), min(result.shape[0], y2 + margin)
        h, w = y2 - y1, x2 - x1
        if h > 0 and w > 0:
            out_resized = cv.resize(out_bgr, (w, h))
            result[y1:y2, x1:x2] = out_resized

        return result

    def export_onnx(self, save_path: str | Path, input_size: int = 256) -> None:
        """Export the generator to ONNX for deployment.

        Args:
            save_path: Output .onnx file path.
            input_size: Face crop resolution used during export.
        """
        self._load_models()
        gen = self._models["generator"]
        dummy_face = torch.randn(1, 4, input_size, input_size).to(self.device)
        dummy_audio = torch.randn(1, 512).to(self.device)
        dummy_id = torch.randn(1, 512).to(self.device)
        torch.onnx.export(
            gen,
            (dummy_face, dummy_audio, dummy_id),
            str(save_path),
            input_names=["face_crop", "audio_emb", "identity_emb"],
            output_names=["rgb", "alpha"],
            dynamic_axes={
                "face_crop": {0: "batch"},
                "audio_emb": {0: "batch"},
                "identity_emb": {0: "batch"},
            },
            opset_version=17,
        )
        print(f"ONNX model exported to {save_path}")


def cv2_resize(img: np.ndarray, size: int) -> np.ndarray:
    import cv2
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

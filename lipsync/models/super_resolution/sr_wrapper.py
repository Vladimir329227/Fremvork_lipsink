"""Super-resolution wrapper for GFPGAN and CodeFormer.

Provides a unified interface so the rest of the framework does not depend
on which SR backend is installed.  Falls back gracefully if packages are
missing so the framework can run without SR.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

SRBackend = Literal["gfpgan", "codeformer", "realesrgan", "none"]


class SuperResolutionWrapper(nn.Module):
    """Lazy-loading wrapper for face super-resolution backends.

    Supports GFPGAN v1.4, CodeFormer, and Real-ESRGAN.
    All backends operate on numpy BGR images (uint8) to stay compatible
    with face-detection pipelines that use OpenCV conventions.

    Args:
        backend: Which SR model to use.
        model_path: Path to pre-downloaded model weights.
        upscale: Integer upscale factor (1 = enhance only, no resize).
        device: 'cuda' | 'cpu' | 'auto'.
    """

    def __init__(
        self,
        backend: SRBackend = "gfpgan",
        model_path: str | Path | None = None,
        upscale: int = 1,
        device: str = "auto",
    ) -> None:
        super().__init__()
        self.backend = backend
        self.model_path = Path(model_path) if model_path else None
        self.upscale = upscale
        self.device = self._resolve_device(device)
        self._model = None  # lazy initialisation

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _init_model(self) -> None:
        if self.backend == "gfpgan":
            self._init_gfpgan()
        elif self.backend == "codeformer":
            self._init_codeformer()
        elif self.backend == "realesrgan":
            self._init_realesrgan()

    def _init_gfpgan(self) -> None:
        try:
            from gfpgan import GFPGANer

            weight_path = str(self.model_path) if self.model_path else None
            self._model = GFPGANer(
                model_path=weight_path
                or "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
                upscale=self.upscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
        except ImportError:
            raise ImportError(
                "gfpgan is not installed. Run: pip install gfpgan"
            )

    def _init_codeformer(self) -> None:
        try:
            from basicsr.archs.codeformer_arch import CodeFormer as CF

            self._cf_net = CF(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=["32", "64", "128", "256"],
            ).to(self.device)
            if self.model_path and self.model_path.exists():
                ckpt = torch.load(self.model_path, map_location=self.device)
                self._cf_net.load_state_dict(ckpt["params_ema"])
            self._cf_net.eval()
            self._model = self._cf_net
        except ImportError:
            raise ImportError(
                "basicsr is not installed. Run: pip install basicsr"
            )

    def _init_realesrgan(self) -> None:
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
            self._model = RealESRGANer(
                scale=self.upscale,
                model_path=str(self.model_path) if self.model_path else None,
                model=net,
                device=self.device,
            )
        except ImportError:
            raise ImportError(
                "realesrgan is not installed. Run: pip install realesrgan"
            )

    @torch.no_grad()
    def enhance(self, img_bgr: np.ndarray, fidelity: float = 0.5) -> np.ndarray:
        """Enhance a single face image.

        Args:
            img_bgr: (H, W, 3) uint8 BGR numpy array.
            fidelity: CodeFormer fidelity weight (0 = quality, 1 = identity).

        Returns:
            Enhanced (H', W', 3) uint8 BGR array.
        """
        if self._model is None:
            self._init_model()

        if self.backend == "gfpgan":
            _, _, output = self._model.enhance(
                img_bgr, has_aligned=False, only_center_face=False, paste_back=True
            )
            return output

        if self.backend == "codeformer":
            import cv2

            face_t = torch.from_numpy(img_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
            face_t = face_t.unsqueeze(0).to(self.device)
            output = self._model(face_t, w=fidelity, adain=True)[0]
            output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            return (output[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8)

        if self.backend == "realesrgan":
            output, _ = self._model.enhance(img_bgr)
            return output

        return img_bgr

    def forward(self, img_bgr: np.ndarray, **kwargs) -> np.ndarray:
        return self.enhance(img_bgr, **kwargs)

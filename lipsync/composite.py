"""Face compositing: paste or blend generator output in a tight lip ROI (Wav2Lip-style)."""
from __future__ import annotations

from typing import Any

import torch

# Default lip box on a frontal face crop (256²): mouth/jaw band, not full lower half.
# Fractions of H (rows) and W (cols). Tuned like common lip-sync pipelines (small central ROI).
_DEFAULT_LIP_BOX = (0.52, 0.86, 0.22, 0.78)


def pool_audio_sequence(seq: torch.Tensor, mode: str = "last") -> torch.Tensor:
    """(B, T, D) -> (B, D).

    *last* — embedding aligned with the end of the mel window (typical viseme timing).
    *mean* — average over time (can wash out per-frame lip motion if the window is wide).
    """
    if seq.dim() == 2:
        return seq
    if mode == "last":
        return seq[:, -1]
    if mode == "mean":
        return seq.mean(dim=1)
    raise ValueError(f"Unknown audio pool mode: {mode}")


def _normalize_composite_mode(mode: str) -> str:
    if mode == "hard_lower":
        return "paste"
    return mode


def _lip_roi_slices(
    h: int, w: int, fracs: tuple[float, float, float, float]
) -> tuple[slice, slice]:
    y0f, y1f, x0f, x1f = fracs
    ys = max(0, min(h - 1, int(h * y0f)))
    ye = max(ys + 1, min(h, int(h * y1f)))
    xs = max(0, min(w - 1, int(w * x0f)))
    xe = max(xs + 1, min(w, int(w * x1f)))
    return slice(ys, ye), slice(xs, xe)


def lip_roi_slices(
    h: int, w: int, fracs: tuple[float, float, float, float]
) -> tuple[slice, slice]:
    """Public alias for pixel slices of the lip ROI (same as training / inference use)."""
    return _lip_roi_slices(h, w, fracs)


def _feather_mask_2d(
    h: int,
    w: int,
    feather_px: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Weight in ``[0, 1]`` for generator content; 1 in the interior, falls off at ROI edges."""
    if feather_px <= 0 or h < 2 or w < 2:
        return torch.ones(1, 1, h, w, device=device, dtype=dtype)
    fe = min(int(feather_px), h // 2, w // 2)
    if fe <= 0:
        return torch.ones(1, 1, h, w, device=device, dtype=dtype)
    yy = torch.arange(h, device=device, dtype=dtype).view(h, 1)
    xx = torch.arange(w, device=device, dtype=dtype).view(1, w)
    h1 = torch.tensor(h - 1, device=device, dtype=dtype)
    w1 = torch.tensor(w - 1, device=device, dtype=dtype)
    dist_y = torch.minimum(yy, h1 - yy)
    dist_x = torch.minimum(xx, w1 - xx)
    fy = (dist_y / float(fe)).clamp(0.0, 1.0)
    fx = (dist_x / float(fe)).clamp(0.0, 1.0)
    m = fy * fx
    # Smoothstep for less visible rectangular boundary (common in lip paste forks).
    return (m * m * (3.0 - 2.0 * m)).view(1, 1, h, w)


def composite_mouth_region(
    face: torch.Tensor,
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    mouth_start_frac: float = 0.42,
    *,
    alpha_floor: float | None = None,
    composite_mode: str = "blend",
    composite_scope: str = "lower_half",
    lip_box_fracs: tuple[float, float, float, float] = _DEFAULT_LIP_BOX,
    lip_roi_feather_px: int = 0,
) -> torch.Tensor:
    """Merge generator RGB/alpha with the driving face.

    *composite_scope*:
    - ``lower_half``: legacy — all rows from *mouth_start_frac* downward (wide band; prone
      to static artefacts on cheeks/jaw if the model only fits the mouth).
    - ``lip_box``: Wav2Lip-style — only a central rectangle around the mouth is updated;
      cheeks, chin sides, and forehead stay from *face*.

    *composite_mode*:
    - ``blend``: alpha * rgb + (1-alpha) * face inside the scope region.
    - ``paste`` (YAML alias ``hard_lower``): copy generator RGB inside the region.

    **Train / infer parity:** use the same ``composite_scope``, ``composite_mode``, and
    ``lip_box_fracs`` for training and inference (read from YAML ``inference`` in both
    :func:`mouth_composite_kwargs_from_inference` and the trainer). If training uses
    ``lower_half`` + ``blend`` while inference uses ``lip_box`` + ``paste``, the generator
    can learn useless RGB wherever alpha is low (cheeks) and inference will paste that
    noise into the mouth box (Wav2Lip-style pipelines keep the composited training target
    aligned with the final paste).

    *lip_roi_feather_px* (``lip_box`` only): if > 0, blend generator toward the original
    face near the ROI border (smoothstep falloff), reducing hard rectangular edges.
    """
    mode = _normalize_composite_mode(composite_mode)
    if mode not in ("blend", "paste"):
        raise ValueError(f"Unknown composite_mode: {composite_mode}")
    if composite_scope not in ("lower_half", "lip_box"):
        raise ValueError(f"Unknown composite_scope: {composite_scope}")

    _, _, h, w = face.shape
    out = face.clone()

    if composite_scope == "lower_half":
        mh = max(1, min(h - 1, int(h * mouth_start_frac)))
        if mode == "paste":
            out[:, :, mh:, :] = rgb[:, :, mh:, :]
            return out
        a = alpha
        if alpha_floor is not None and alpha_floor > 0.0:
            a = a.clamp(min=float(alpha_floor))
        blended = a * rgb + (1.0 - a) * face
        out[:, :, mh:, :] = blended[:, :, mh:, :]
        return out

    # lip_box
    sy, sx = _lip_roi_slices(h, w, lip_box_fracs)
    f_roi = face[:, :, sy, sx]
    r_roi = rgb[:, :, sy, sx]
    a_roi = alpha[:, :, sy, sx]
    rh, rw = r_roi.shape[-2], r_roi.shape[-1]
    w_feather = _feather_mask_2d(
        rh, rw, lip_roi_feather_px, device=face.device, dtype=face.dtype
    )

    if mode == "paste":
        out[:, :, sy, sx] = w_feather * r_roi + (1.0 - w_feather) * f_roi
        return out
    a = a_roi
    if alpha_floor is not None and alpha_floor > 0.0:
        a = a.clamp(min=float(alpha_floor))
    blended = a * r_roi + (1.0 - a) * f_roi
    out[:, :, sy, sx] = w_feather * blended + (1.0 - w_feather) * f_roi
    return out


def mouth_composite_kwargs_from_inference(inf: dict[str, Any]) -> dict[str, Any]:
    """Keyword args for :func:`composite_mouth_region` from YAML ``inference`` dict."""
    mode_raw = inf.get("mouth_composite_mode", "blend")
    if not isinstance(mode_raw, str):
        mode_raw = "blend"
    mode = _normalize_composite_mode(mode_raw)
    if mode not in ("blend", "paste"):
        mode = "paste"

    scope = inf.get("mouth_composite_scope", "lip_box")
    if scope not in ("lower_half", "lip_box"):
        scope = "lip_box"

    y0 = float(inf.get("lip_roi_y0", _DEFAULT_LIP_BOX[0]))
    y1 = float(inf.get("lip_roi_y1", _DEFAULT_LIP_BOX[1]))
    x0 = float(inf.get("lip_roi_x0", _DEFAULT_LIP_BOX[2]))
    x1 = float(inf.get("lip_roi_x1", _DEFAULT_LIP_BOX[3]))
    lip_box_fracs = (y0, y1, x0, x1)

    raw_feather = inf.get("lip_roi_feather_px", 0)
    try:
        lip_roi_feather_px = max(0, int(raw_feather))
    except (TypeError, ValueError):
        lip_roi_feather_px = 0

    raw_floor = inf.get("mouth_alpha_min", None)
    if _normalize_composite_mode(str(mode)) == "paste":
        alpha_floor = None
    elif raw_floor is None:
        alpha_floor = 0.45
    else:
        f = float(raw_floor)
        alpha_floor = None if f <= 0 else f

    return {
        "alpha_floor": alpha_floor,
        "composite_mode": mode,
        "composite_scope": scope,
        "lip_box_fracs": lip_box_fracs,
        "lip_roi_feather_px": lip_roi_feather_px,
    }

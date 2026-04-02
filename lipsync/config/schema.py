"""Strict config validation with fail-fast errors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Keep in sync with ``configs/base.yaml`` ``inference:`` and ``LipSyncConfig.inference``.
DEFAULT_INFERENCE: dict[str, Any] = {
    "smoothing": 0.0,
    "paste_mode": "direct",
    "keep_original_audio": True,
    "mux_driving_audio": True,
    "mouth_blend_from": 0.42,
    # last: current-frame-aligned embedding (closer to mel→lip timing); mean dilutes motion.
    "audio_embed_pool": "last",
    "mouth_composite_mode": "blend",
    "mouth_composite_scope": "lip_box",
    "lip_roi_y0": 0.52,
    "lip_roi_y1": 0.86,
    "lip_roi_x0": 0.22,
    "lip_roi_x1": 0.78,
    "lip_roi_feather_px": 6,
    "mouth_alpha_min": 0.45,
}


def merge_inference_defaults(inference: dict[str, Any] | None) -> dict[str, Any]:
    """Overlay *inference* on defaults so checkpoints list full infer config (train == infer)."""
    out = dict(DEFAULT_INFERENCE)
    out.update(inference or {})
    return out


@dataclass
class ValidationErrorItem:
    key: str
    message: str


class ConfigValidationError(ValueError):
    pass


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float))


def validate_config(config: dict[str, Any]) -> None:
    errors: list[ValidationErrorItem] = []

    def err(key: str, msg: str) -> None:
        errors.append(ValidationErrorItem(key, msg))

    if not isinstance(config, dict):
        raise ConfigValidationError("Config must be a dict")

    epochs = config.get("epochs", 100)
    if not isinstance(epochs, int) or epochs <= 0:
        err("epochs", "must be positive int")

    batch_size = config.get("batch_size", 8)
    if not isinstance(batch_size, int) or batch_size <= 0:
        err("batch_size", "must be positive int")

    optimizer = config.get("optimizer", {})
    if not isinstance(optimizer, dict):
        err("optimizer", "must be dict")
    else:
        if "name" not in optimizer:
            err("optimizer.name", "is required")
        lr = optimizer.get("lr", 1e-3)
        if not _is_number(lr) or lr <= 0:
            err("optimizer.lr", "must be positive number")

    model = config.get("model", {})
    if not isinstance(model, dict):
        err("model", "must be dict")
    else:
        for k in ["audio_embed_dim", "identity_embed_dim", "gen_base_ch", "gen_depth"]:
            if k in model and (not isinstance(model[k], int) or model[k] <= 0):
                err(f"model.{k}", "must be positive int")

    losses = config.get("losses", {})
    if not isinstance(losses, dict):
        err("losses", "must be dict")
    else:
        for k, v in losses.items():
            if k.startswith("w_") and (not _is_number(v) or v < 0):
                err(f"losses.{k}", "must be non-negative number")
        if "lip_recon_boost" in losses:
            v = losses["lip_recon_boost"]
            if not _is_number(v) or float(v) < 1.0:
                err("losses.lip_recon_boost", "must be a number >= 1 (mouth ROI Huber weight multiplier)")

    runtime = config.get("runtime", {})
    if runtime and not isinstance(runtime, dict):
        err("runtime", "must be dict")

    audio = config.get("audio", {})
    if audio and not isinstance(audio, dict):
        err("audio", "must be dict")
    elif isinstance(audio, dict):
        for k in ["sample_rate", "n_mels", "window"]:
            if k in audio and (not isinstance(audio[k], int) or audio[k] <= 0):
                err(f"audio.{k}", "must be positive int")

    video = config.get("video", {})
    if video and not isinstance(video, dict):
        err("video", "must be dict")
    elif isinstance(video, dict):
        for k in ["face_size", "lip_size"]:
            if k in video and (not isinstance(video[k], int) or video[k] <= 0):
                err(f"video.{k}", "must be positive int")
        if "target_fps" in video and (not _is_number(video["target_fps"]) or video["target_fps"] <= 0):
            err("video.target_fps", "must be positive number")

    lipsync = config.get("lipsync", {})
    if lipsync and not isinstance(lipsync, dict):
        err("lipsync", "must be dict")
    elif isinstance(lipsync, dict):
        for k in ["sync_window", "temporal_radius"]:
            if k in lipsync and (not isinstance(lipsync[k], int) or lipsync[k] < 0):
                err(f"lipsync.{k}", "must be non-negative int")
        if "mouth_region_weight" in lipsync and (
            not _is_number(lipsync["mouth_region_weight"]) or lipsync["mouth_region_weight"] < 0
        ):
            err("lipsync.mouth_region_weight", "must be non-negative number")

    data = config.get("data", {})
    if data and not isinstance(data, dict):
        err("data", "must be dict")
    elif isinstance(data, dict):
        if "static_face_prob" in data:
            v = data["static_face_prob"]
            if not _is_number(v) or float(v) < 0.0 or float(v) > 1.0:
                err("data.static_face_prob", "must be a number in [0, 1]")

    inference = config.get("inference", {})
    if inference and not isinstance(inference, dict):
        err("inference", "must be dict")
    elif isinstance(inference, dict):
        if "smoothing" in inference and (not _is_number(inference["smoothing"]) or inference["smoothing"] < 0):
            err("inference.smoothing", "must be non-negative number")
        if "paste_mode" in inference and inference["paste_mode"] not in {"direct", "seamless"}:
            err("inference.paste_mode", "must be one of: direct, seamless")
        if "keep_original_audio" in inference and not isinstance(inference["keep_original_audio"], bool):
            err("inference.keep_original_audio", "must be bool")
        if "mux_driving_audio" in inference and not isinstance(inference["mux_driving_audio"], bool):
            err("inference.mux_driving_audio", "must be bool")
        if "mouth_blend_from" in inference:
            v = inference["mouth_blend_from"]
            if not _is_number(v) or not (0.0 < float(v) < 1.0):
                err("inference.mouth_blend_from", "must be in (0, 1)")
        if "audio_embed_pool" in inference and inference["audio_embed_pool"] not in {"mean", "last"}:
            err("inference.audio_embed_pool", "must be 'mean' or 'last'")
        if "mouth_composite_mode" in inference and inference["mouth_composite_mode"] not in {
            "blend",
            "paste",
            "hard_lower",
        }:
            err(
                "inference.mouth_composite_mode",
                "must be 'blend', 'paste', or 'hard_lower' (alias for paste)",
            )
        if "mouth_composite_scope" in inference and inference["mouth_composite_scope"] not in {
            "lower_half",
            "lip_box",
        }:
            err("inference.mouth_composite_scope", "must be 'lower_half' or 'lip_box'")
        for key in ("lip_roi_y0", "lip_roi_y1", "lip_roi_x0", "lip_roi_x1"):
            if key in inference:
                v = inference[key]
                if not _is_number(v) or float(v) < 0.0 or float(v) > 1.0:
                    err(f"inference.{key}", "must be a number in [0, 1]")
        if all(k in inference for k in ("lip_roi_y0", "lip_roi_y1", "lip_roi_x0", "lip_roi_x1")):
            y0, y1 = float(inference["lip_roi_y0"]), float(inference["lip_roi_y1"])
            x0, x1 = float(inference["lip_roi_x0"]), float(inference["lip_roi_x1"])
            if y0 >= y1 or x0 >= x1:
                err("inference.lip_roi_*", "require y0 < y1 and x0 < x1")
        if "mouth_alpha_min" in inference and inference["mouth_alpha_min"] is not None:
            v = inference["mouth_alpha_min"]
            if not _is_number(v) or float(v) < 0 or float(v) > 1.0:
                err("inference.mouth_alpha_min", "must be a number in [0, 1] or omit")
        if "lip_roi_feather_px" in inference:
            v = inference["lip_roi_feather_px"]
            if not isinstance(v, int) or v < 0:
                err("inference.lip_roi_feather_px", "must be non-negative int")

    if errors:
        text = "\n".join(f"- {e.key}: {e.message}" for e in errors)
        raise ConfigValidationError(f"Configuration validation failed:\n{text}")

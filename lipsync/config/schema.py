"""Strict config validation with fail-fast errors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

    if errors:
        text = "\n".join(f"- {e.key}: {e.message}" for e in errors)
        raise ConfigValidationError(f"Configuration validation failed:\n{text}")

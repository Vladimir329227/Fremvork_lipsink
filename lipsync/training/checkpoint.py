"""Checkpoint schema v2 + migration utilities."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime
from typing import Any


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def make_metadata(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "framework": "lipsync",
        "framework_version": "0.2.0",
        "python": platform.python_version(),
        "git_hash": _git_hash(),
        "config_hash": config_hash(config),
        "training_signature": {
            "model": config.get("model", {}),
            "optimizer": config.get("optimizer", {}).get("name", "unknown"),
            "losses": sorted(list(config.get("losses", {}).keys())),
        },
    }


def migrate_to_v2(ckpt: dict[str, Any]) -> dict[str, Any]:
    """Migrate legacy checkpoint dict (v1/no-schema) to schema v2."""
    if ckpt.get("meta", {}).get("schema_version") == 2:
        return ckpt

    config = ckpt.get("config", {})
    meta = make_metadata(config)
    migrated = dict(ckpt)
    migrated["meta"] = meta
    return migrated


def validate_checkpoint_v2(ckpt: dict[str, Any]) -> None:
    required_top = ["audio_encoder", "identity_encoder", "generator", "config", "meta"]
    missing = [k for k in required_top if k not in ckpt]
    if missing:
        raise ValueError(f"Invalid checkpoint: missing keys {missing}")
    meta = ckpt.get("meta", {})
    if meta.get("schema_version") != 2:
        raise ValueError("Invalid checkpoint schema: expected schema_version=2")

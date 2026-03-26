"""Runtime compatibility checks and execution profiles.

This module provides:
- dependency diagnostics (`collect_runtime_report`)
- compatibility checks (`assert_runtime_compatible`)
- deterministic setup (`set_deterministic`)
- execution profiles (`resolve_profile`)
"""
from __future__ import annotations

import importlib
import os
import platform
import random
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import torch


@dataclass
class RuntimeProfile:
    name: str
    fp16: bool
    batch_size: int
    num_workers: int
    prefer_onnx: bool
    use_torchscript: bool
    gradient_checkpointing: bool


PROFILES: dict[str, RuntimeProfile] = {
    "cpu-safe": RuntimeProfile(
        name="cpu-safe",
        fp16=False,
        batch_size=2,
        num_workers=0,
        prefer_onnx=False,
        use_torchscript=True,
        gradient_checkpointing=False,
    ),
    "gpu-fast": RuntimeProfile(
        name="gpu-fast",
        fp16=True,
        batch_size=16,
        num_workers=4,
        prefer_onnx=True,
        use_torchscript=False,
        gradient_checkpointing=False,
    ),
    "gpu-quality": RuntimeProfile(
        name="gpu-quality",
        fp16=True,
        batch_size=8,
        num_workers=4,
        prefer_onnx=False,
        use_torchscript=False,
        gradient_checkpointing=True,
    ),
}


def resolve_profile(name: str) -> RuntimeProfile:
    key = name.lower()
    if key not in PROFILES:
        raise KeyError(f"Unknown profile '{name}'. Available: {', '.join(PROFILES)}")
    return PROFILES[key]


def _try_import(module: str) -> tuple[bool, str | None]:
    try:
        m = importlib.import_module(module)
        ver = getattr(m, "__version__", "unknown")
        return True, str(ver)
    except Exception:
        return False, None


def collect_runtime_report() -> dict[str, Any]:
    """Collect runtime environment diagnostics."""
    modules = [
        "torch", "torchvision", "torchaudio", "cv2",
        "onnx", "onnxruntime", "fastapi", "face_alignment",
    ]
    deps = {}
    for m in modules:
        ok, ver = _try_import(m)
        deps[m] = {"available": ok, "version": ver}

    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "deps": deps,
    }
    if torch.cuda.is_available():
        try:
            report["cuda_name"] = torch.cuda.get_device_name(0)
        except Exception:
            report["cuda_name"] = "unknown"
    return report


def assert_runtime_compatible(
    require_torchvision: bool = False,
    require_torchaudio: bool = False,
    require_cv2: bool = True,
    require_onnxruntime: bool = False,
) -> dict[str, Any]:
    """Validate runtime dependencies and raise helpful error if broken."""
    report = collect_runtime_report()
    deps = report["deps"]

    required = {
        "torch": True,
        "cv2": require_cv2,
        "torchvision": require_torchvision,
        "torchaudio": require_torchaudio,
        "onnxruntime": require_onnxruntime,
    }

    missing = [k for k, req in required.items() if req and not deps[k]["available"]]
    if missing:
        hint = {
            "torch": "pip install torch torchvision torchaudio",
            "cv2": "pip install opencv-python",
            "torchvision": "pip install torchvision",
            "torchaudio": "pip install torchaudio",
            "onnxruntime": "pip install onnxruntime",
        }
        details = "\n".join(f"- {m}: {hint.get(m, '')}" for m in missing)
        raise RuntimeError(
            "Runtime compatibility check failed. Missing dependencies:\n"
            f"{details}\n\nFull report: {report}"
        )

    # Torch/torchvision mismatch is common; test a small import path
    if deps["torch"]["available"] and deps["torchvision"]["available"]:
        try:
            import torchvision
            _ = torchvision.__version__
        except Exception as exc:
            if require_torchvision:
                raise RuntimeError(
                    "torchvision is installed but incompatible with torch. "
                    "Install matching versions."
                ) from exc

    return report


def set_deterministic(seed: int = 42) -> None:
    """Enable deterministic behaviour across Python/NumPy/PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Safe deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def apply_profile_to_config(config: dict[str, Any], profile_name: str) -> dict[str, Any]:
    """Apply runtime profile defaults into a training config dict."""
    profile = resolve_profile(profile_name)
    out = dict(config)
    out.setdefault("fp16", profile.fp16)
    out.setdefault("batch_size", profile.batch_size)
    out.setdefault("gradient_checkpointing", profile.gradient_checkpointing)
    data_cfg = dict(out.get("data", {}))
    data_cfg.setdefault("num_workers", profile.num_workers)
    out["data"] = data_cfg
    out.setdefault("runtime", {})
    out["runtime"] = {
        **out["runtime"],
        "profile": profile.name,
        "prefer_onnx": profile.prefer_onnx,
        "use_torchscript": profile.use_torchscript,
    }
    return out


def runtime_report_text(report: dict[str, Any]) -> str:
    deps = report.get("deps", {})
    lines = [
        f"Python: {report.get('python')}",
        f"Platform: {report.get('platform')}",
        f"CUDA: {report.get('torch_cuda_available')} ({report.get('torch_cuda_device_count')} devices)",
    ]
    if report.get("cuda_name"):
        lines.append(f"CUDA Device: {report['cuda_name']}")
    lines.append("Dependencies:")
    for name, d in deps.items():
        state = "OK" if d["available"] else "MISSING"
        lines.append(f"  - {name:12s} {state:8s} {d['version'] or ''}")
    return "\n".join(lines)

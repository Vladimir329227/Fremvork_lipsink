"""Operational tools: doctor, benchmark, realtime profiling."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .runtime import assert_runtime_compatible, collect_runtime_report, runtime_report_text


def doctor() -> dict[str, Any]:
    report = collect_runtime_report()
    return report


def benchmark(
    device: str = "auto",
    batch_size: int = 1,
    steps: int = 50,
    image_size: int = 256,
) -> dict[str, Any]:
    """Simple synthetic benchmark for generator-like conv stack."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    model = torch.nn.Sequential(
        torch.nn.Conv2d(4, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 4, 3, padding=1),
    ).to(dev)
    model.eval()

    x = torch.randn(batch_size, 4, image_size, image_size, device=dev)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            _ = model(x)
    dt = time.perf_counter() - t0

    frames = steps * batch_size
    fps = frames / dt if dt > 0 else 0.0
    latency_ms = (dt / steps) * 1000.0

    result = {
        "device": str(dev),
        "steps": steps,
        "batch_size": batch_size,
        "image_size": image_size,
        "avg_latency_ms": round(latency_ms, 3),
        "throughput_fps": round(fps, 3),
    }
    return result


def profile_realtime(
    generation_times_s: list[float],
    clip_durations_s: list[float],
) -> dict[str, float]:
    """Compute p50/p95/p99 latency and realtime factor summary."""
    if not generation_times_s:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "avg_realtime_factor": 0.0}

    arr = np.array(generation_times_s, dtype=np.float64)
    rt_factors = []
    for gen_t, clip_t in zip(generation_times_s, clip_durations_s):
        rt_factors.append((clip_t / gen_t) if gen_t > 0 else 0.0)

    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "avg_realtime_factor": float(np.mean(rt_factors) if rt_factors else 0.0),
    }


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")

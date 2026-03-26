"""Practical lip-sync demo example for this repository.

Runs inference-only demo using existing checkpoint and assets:
    - checkpoint: checkpoints/demo_model.pt
    - videos: result/*.mp4
    - audios: result/*.wav

Output goes to:
    result/example_demo/
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    checkpoint = root / "checkpoints" / "demo_model.pt"
    result_dir = root / "result"
    output_dir = result_dir / "example_demo"

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    videos = sorted(result_dir.glob("*.mp4"))
    audios = sorted(result_dir.glob("*.wav"))
    if not videos or not audios:
        raise RuntimeError("Expected input assets in result/: *.mp4 and *.wav")

    cmd = [
        sys.executable,
        str(root / "scripts" / "run_lipsync_result.py"),
        "--skip-train",
        "--checkpoint",
        str(checkpoint),
        "--result-dir",
        str(result_dir),
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
    ]
    print("Running command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    summary = output_dir / "summary.json"
    print("\nDone.")
    print(f"Output directory: {output_dir}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()


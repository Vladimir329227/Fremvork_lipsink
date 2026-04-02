#!/usr/bin/env python3
"""CLI wrapper: still image + WAV → MP4 (length = audio duration)."""
from __future__ import annotations

import argparse
from pathlib import Path

from lipsync.inference.static_clip import image_to_static_mp4


def main() -> None:
    p = argparse.ArgumentParser(description="PNG/JPG + WAV → static MP4")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True, help="WAV defines output duration")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fps", type=float, default=25.0)
    args = p.parse_args()
    out = image_to_static_mp4(args.image, args.audio, args.output, fps=args.fps)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

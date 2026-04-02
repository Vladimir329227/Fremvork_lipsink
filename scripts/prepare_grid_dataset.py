#!/usr/bin/env python3
"""Prepare GRID corpus videos for this lip-sync framework."""
from __future__ import annotations

import argparse
from pathlib import Path

from lipsync.data import prepare_grid_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GRID dataset for lipsync training")
    parser.add_argument("--input-root", type=Path, default=Path("data"), help="Directory containing s*_processed folders")
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"), help="Output dataset root")
    parser.add_argument("--speakers", type=str, default="", help="Comma-separated speaker folders, e.g. s7_processed,s31_processed")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--face-size", type=int, default=256)
    parser.add_argument("--lip-size", type=int, default=96)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="Process only first N clips (0 = all)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--split-by-speaker",
        action="store_true",
        help="Keep each speaker in a single split (train or val)",
    )
    args = parser.parse_args()

    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()] if args.speakers else None
    summary = prepare_grid_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        speakers=speakers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        face_size=args.face_size,
        lip_size=args.lip_size,
        fps=args.fps,
        device=args.device,
        min_frames=args.min_frames,
        limit=args.limit,
        overwrite=args.overwrite,
        split_by_speaker=args.split_by_speaker,
    )
    print("Done.")
    print(summary)
    print(f"Dataset ready at: {args.output_root}")
    print(f"Train with: python cli.py train --data-root {args.output_root} --config configs/base.yaml")


if __name__ == "__main__":
    main()


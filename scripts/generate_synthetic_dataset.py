#!/usr/bin/env python3
"""Generate correlated synthetic train/val data for overfit / PSNR sanity checks."""
from __future__ import annotations

import argparse
from pathlib import Path

from lipsync.data import write_correlated_synthetic_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, default=Path("_syn_train"))
    p.add_argument("--train-clips", type=int, default=6)
    p.add_argument("--val-clips", type=int, default=2)
    p.add_argument("--n-frames", type=int, default=48)
    p.add_argument("--face-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    s = write_correlated_synthetic_dataset(
        args.output_root,
        n_train_clips=args.train_clips,
        n_val_clips=args.val_clips,
        n_frames=args.n_frames,
        face_size=args.face_size,
        seed=args.seed,
    )
    print(s)
    print(
        "Train (synthetic ONLY): python cli.py train --data-root "
        f"{args.output_root} --config configs/synthetic_dataset_only.yaml"
    )
    print(f"Score: python cli.py eval --checkpoint <ckpt> --data-root {args.output_root} --split train")


if __name__ == "__main__":
    main()

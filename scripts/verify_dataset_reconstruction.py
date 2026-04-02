#!/usr/bin/env python3
"""Val/test reconstruction: augmented first frame + clip audio vs GT (see plan E)."""
from __future__ import annotations

import argparse
from pathlib import Path

from lipsync.evaluation.reconstruction import run_dataset_reconstruction_verify


def main() -> None:
    p = argparse.ArgumentParser(description="Dataset reconstruction verify (holdout split only)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split", type=str, default="val", choices=("val", "test"))
    p.add_argument("--num-clips", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--out-dir", type=Path, default=Path("verify_recon_out"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--no-mux", action="store_true", help="Skip ffmpeg audio mux")
    args = p.parse_args()
    s = run_dataset_reconstruction_verify(
        args.checkpoint,
        args.data_root,
        split=args.split,
        num_clips=args.num_clips,
        seed=args.seed,
        augment=not args.no_augment,
        out_dir=args.out_dir,
        device=args.device,
        batch_size=args.batch_size,
        mux_audio=not args.no_mux,
    )
    print(s)


if __name__ == "__main__":
    main()

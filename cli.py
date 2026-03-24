#!/usr/bin/env python3
"""LipSync Framework — Command-Line Interface.

Usage examples::

    # Train from base config
    python cli.py train --config configs/base.yaml

    # Train with overrides
    python cli.py train --config configs/base.yaml --epochs 200 --batch-size 16

    # Run inference
    python cli.py infer --checkpoint checkpoints/best_model.pt \\
                        --audio voice.wav --video face.mp4 --output result.mp4

    # Start API server
    python cli.py serve --checkpoint checkpoints/best_model.pt --port 8000

    # Export to ONNX
    python cli.py export --checkpoint checkpoints/best_model.pt --output model.onnx

    # Evaluate on a dataset
    python cli.py eval --checkpoint checkpoints/best_model.pt --data-root data/test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    from lipsync import LipSyncConfig, LipSyncTrainer
    from lipsync.data import LipSyncDataset
    from lipsync.training.callbacks import EarlyStopping, WandbLogger

    # Build config
    if args.config:
        trainer = LipSyncTrainer.from_config(
            args.config,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            fp16=not args.no_fp16,
        )
    else:
        cfg = LipSyncConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            fp16=not args.no_fp16,
            checkpoint_dir=args.checkpoint_dir,
            log_wandb=args.wandb,
        )
        if args.optimizer:
            cfg.optimizer["name"] = args.optimizer
        if args.lr:
            cfg.optimizer["lr"] = args.lr
        trainer = LipSyncTrainer(cfg, device=args.device)

    # Load datasets
    train_ds = LipSyncDataset(args.data_root, split="train", augment=True)
    val_ds = LipSyncDataset(args.data_root, split="val") if Path(args.data_root, "val_metadata.json").exists() else None

    callbacks = []
    if args.early_stopping > 0:
        callbacks.append(EarlyStopping(patience=args.early_stopping))

    print(f"Training for {trainer.config.epochs} epochs on device {trainer.device}")
    trainer.fit(train_ds, val_ds, callbacks=callbacks)
    print("Training complete.")


def cmd_infer(args: argparse.Namespace) -> None:
    from lipsync import LipSyncTrainer

    trainer = LipSyncTrainer.from_checkpoint(args.checkpoint, device=args.device)
    result = trainer.predict(
        audio=args.audio,
        video=args.video,
        output_path=args.output,
        use_sr=args.use_sr,
    )
    print(f"Generated {len(result)} frames → {args.output}")


def cmd_serve(args: argparse.Namespace) -> None:
    from lipsync.api.app import run_server

    print(f"Starting API server at http://{args.host}:{args.port}")
    run_server(
        checkpoint_path=args.checkpoint,
        host=args.host,
        port=args.port,
        device=args.device,
        use_sr=args.use_sr,
    )


def cmd_export(args: argparse.Namespace) -> None:
    from lipsync import LipSyncTrainer

    trainer = LipSyncTrainer.from_checkpoint(args.checkpoint, device=args.device)
    trainer.export_onnx(args.output)
    print(f"ONNX model exported to {args.output}")


def cmd_eval(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    from lipsync import LipSyncTrainer
    from lipsync.data import LipSyncDataset
    from lipsync.evaluation import LipSyncEvaluator

    trainer = LipSyncTrainer.from_checkpoint(args.checkpoint, device=args.device)
    ds = LipSyncDataset(args.data_root, split=args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=2)

    evaluator = LipSyncEvaluator(device=str(trainer.device))
    all_metrics = []

    trainer._core.audio_encoder.eval()
    trainer._core.generator.eval()
    trainer._core.identity_encoder.eval()

    with torch.no_grad():
        for batch in loader:
            mel = batch["mel"].to(trainer.device)
            face = batch["face"].to(trainer.device)
            ref_face = batch["ref_face"].to(trainer.device)
            gt = batch["gt_frame"].to(trainer.device)

            audio_emb = trainer._core.audio_encoder(mel)[:, -1]
            id_emb = trainer._core.identity_encoder(ref_face)

            masked = face.clone()
            masked[:, :, face.shape[-2] // 2 :, :] = 0.0
            masked_4ch = torch.cat([masked, torch.zeros_like(masked[:, :1])], dim=1)

            rgb, alpha = trainer._core.generator(masked_4ch, audio_emb, id_emb)
            pred = alpha * rgb + (1 - alpha) * face

            lm_pred = batch.get("landmarks")
            metrics = evaluator.evaluate(
                pred_frames=pred,
                gt_frames=gt,
                pred_landmarks=lm_pred,
                gt_landmarks=lm_pred,
            )
            all_metrics.append(metrics)

    summary = evaluator.summarise(all_metrics)
    print("\n=== Evaluation Results ===")
    for k, v in summary.items():
        print(f"  {k:20s}: {v:.4f}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lipsync",
        description="LipSync Framework CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    p_train = sub.add_parser("train", help="Train the lip-sync model")
    p_train.add_argument("--config", type=str, default=None, help="Path to YAML config")
    p_train.add_argument("--data-root", type=str, required=True, help="Dataset root directory")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=None)
    p_train.add_argument("--optimizer", type=str, default=None,
                         choices=["sgd", "momentum_sgd", "clipping_sgd", "adamw", "lion"])
    p_train.add_argument("--device", type=str, default="auto")
    p_train.add_argument("--no-fp16", action="store_true")
    p_train.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p_train.add_argument("--early-stopping", type=int, default=0, metavar="PATIENCE")
    p_train.add_argument("--wandb", action="store_true", help="Enable W&B logging")

    # --- infer ---
    p_infer = sub.add_parser("infer", help="Run lip-sync inference on a video")
    p_infer.add_argument("--checkpoint", type=str, required=True)
    p_infer.add_argument("--audio", type=str, required=True, help="Driving audio (WAV)")
    p_infer.add_argument("--video", type=str, required=True, help="Source video")
    p_infer.add_argument("--output", type=str, default="output.mp4")
    p_infer.add_argument("--device", type=str, default="auto")
    p_infer.add_argument("--use-sr", action="store_true", help="Apply super-resolution")

    # --- serve ---
    p_serve = sub.add_parser("serve", help="Start the REST/WebSocket API server")
    p_serve.add_argument("--checkpoint", type=str, required=True)
    p_serve.add_argument("--host", type=str, default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--device", type=str, default="auto")
    p_serve.add_argument("--use-sr", action="store_true")

    # --- export ---
    p_export = sub.add_parser("export", help="Export generator to ONNX")
    p_export.add_argument("--checkpoint", type=str, required=True)
    p_export.add_argument("--output", type=str, default="model.onnx")
    p_export.add_argument("--device", type=str, default="cpu")

    # --- eval ---
    p_eval = sub.add_parser("eval", help="Evaluate a trained model on a test set")
    p_eval.add_argument("--checkpoint", type=str, required=True)
    p_eval.add_argument("--data-root", type=str, required=True)
    p_eval.add_argument("--split", type=str, default="test")
    p_eval.add_argument("--batch-size", type=int, default=16)
    p_eval.add_argument("--device", type=str, default="auto")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "train": cmd_train,
        "infer": cmd_infer,
        "serve": cmd_serve,
        "export": cmd_export,
        "eval": cmd_eval,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

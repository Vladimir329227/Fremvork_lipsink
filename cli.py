#!/usr/bin/env python3
"""LipSync Framework — Command-Line Interface."""
from __future__ import annotations

import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Existing sub-command handlers
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    from lipsync import LipSyncConfig, LipSyncTrainer
    from lipsync.data import LipSyncDataset
    from lipsync.runtime import apply_profile_to_config

    if args.config:
        cfg = LipSyncConfig.from_yaml(args.config)
        cfg_dict = cfg.to_dict()
        if args.profile:
            cfg_dict = apply_profile_to_config(cfg_dict, args.profile)
            cfg = LipSyncConfig.from_dict(cfg_dict)
        trainer = LipSyncTrainer(cfg, device=args.device)
    else:
        cfg = LipSyncConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            fp16=not args.no_fp16,
            checkpoint_dir=args.checkpoint_dir or "checkpoints",
            log_wandb=args.wandb,
        )
        if args.optimizer:
            cfg.optimizer["name"] = args.optimizer
        if args.lr:
            cfg.optimizer["lr"] = args.lr
        if args.profile:
            cfg = LipSyncConfig.from_dict(apply_profile_to_config(cfg.to_dict(), args.profile))
        trainer = LipSyncTrainer(cfg, device=args.device)

    cfg_dict = trainer.config.to_dict()
    cfg_dict.setdefault("audio", {})
    cfg_dict.setdefault("video", {})
    cfg_dict.setdefault("lipsync", {})
    cfg_dict.setdefault("inference", {})
    if args.audio_sample_rate:
        cfg_dict["audio"]["sample_rate"] = args.audio_sample_rate
    if args.audio_n_mels:
        cfg_dict["audio"]["n_mels"] = args.audio_n_mels
    if args.audio_window:
        cfg_dict["audio"]["window"] = args.audio_window
    if args.video_face_size:
        cfg_dict["video"]["face_size"] = args.video_face_size
    if args.video_lip_size:
        cfg_dict["video"]["lip_size"] = args.video_lip_size
    if args.video_target_fps:
        cfg_dict["video"]["target_fps"] = args.video_target_fps
    if args.sync_window is not None:
        cfg_dict["lipsync"]["sync_window"] = args.sync_window
    if args.temporal_radius is not None:
        cfg_dict["lipsync"]["temporal_radius"] = args.temporal_radius
    if args.mouth_region_weight is not None:
        cfg_dict["lipsync"]["mouth_region_weight"] = args.mouth_region_weight
    if args.smoothing is not None:
        cfg_dict["inference"]["smoothing"] = args.smoothing
    if args.paste_mode is not None:
        cfg_dict["inference"]["paste_mode"] = args.paste_mode
    if args.keep_original_audio is not None:
        cfg_dict["inference"]["keep_original_audio"] = args.keep_original_audio == "true"
    cfg_dict["epochs"] = args.epochs
    cfg_dict["batch_size"] = args.batch_size
    if args.checkpoint_dir is not None:
        cfg_dict["checkpoint_dir"] = args.checkpoint_dir
    trainer = LipSyncTrainer(LipSyncConfig.from_dict(cfg_dict), device=args.device)

    dcfg = trainer.config.to_dict().get("data", {})
    acfg = trainer.config.to_dict().get("audio", {})
    vcfg = trainer.config.to_dict().get("video", {})
    audio_window = args.audio_window or dcfg.get("audio_window") or acfg.get("window", 16)
    face_size = args.video_face_size or dcfg.get("face_size") or vcfg.get("face_size", 256)
    lip_size = args.video_lip_size or dcfg.get("lip_size") or vcfg.get("lip_size", 96)
    target_fps = args.video_target_fps or dcfg.get("target_fps") or vcfg.get("target_fps", 25.0)
    static_face_prob = float(dcfg.get("static_face_prob", 0.0))

    train_ds = LipSyncDataset(
        args.data_root,
        split="train",
        augment=True,
        audio_window=audio_window,
        face_size=face_size,
        lip_size=lip_size,
        video_fps=float(target_fps),
        static_face_prob=static_face_prob,
    )
    val_path = Path(args.data_root, "val_metadata.json")
    val_ds = None
    if val_path.exists():
        import json

        with open(val_path, encoding="utf-8") as vf:
            val_entries = json.load(vf)
        if val_entries:
            val_ds = LipSyncDataset(
                args.data_root,
                split="val",
                augment=False,
                audio_window=audio_window,
                face_size=face_size,
                lip_size=lip_size,
                video_fps=float(target_fps),
                static_face_prob=1.0,
            )
    if val_ds is None:
        print(
            "WARNING: No val_metadata.json or empty validation set — "
            "training without validation increases overfitting risk."
        )

    callbacks = []
    if args.early_stopping > 0:
        from lipsync.training.callbacks import EarlyStopping
        callbacks.append(EarlyStopping(patience=args.early_stopping))

    print(f"Training for {trainer.config.epochs} epochs on device {trainer.device}")
    trainer.fit(train_ds, val_ds, callbacks=callbacks)
    print("Training complete.")


def cmd_static_video(args: argparse.Namespace) -> None:
    from lipsync.inference.static_clip import image_to_static_mp4

    out = image_to_static_mp4(
        image_path=args.image,
        audio_path=args.audio,
        output_path=args.output,
        fps=args.fps,
    )
    print(f"Static video written -> {out}")


def cmd_infer(args: argparse.Namespace) -> None:
    from lipsync import LipSyncTrainer

    trainer = LipSyncTrainer.from_checkpoint(args.checkpoint, device=args.device)
    infer_overrides: dict = {}
    if getattr(args, "mouth_composite_mode", None):
        infer_overrides["mouth_composite_mode"] = args.mouth_composite_mode
    elif getattr(args, "infer_blend", False):
        infer_overrides["mouth_composite_mode"] = "blend"
    if getattr(args, "mouth_alpha_min", None) is not None:
        infer_overrides["mouth_alpha_min"] = float(args.mouth_alpha_min)

    result = trainer.predict(
        audio=args.audio,
        video=args.video,
        output_path=args.output,
        use_sr=args.use_sr,
        inference_overrides=infer_overrides or None,
    )
    print(f"Generated {len(result)} frames -> {args.output}")


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


def cmd_wav2lip_user_pipeline(args: argparse.Namespace) -> None:
    """Preprocess video → short Wav2Lip finetune (cloned repo) → inference."""
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parent / "scripts" / "wav2lip_user_train_and_infer.py"
    cmd = [
        sys.executable,
        str(script),
        "--train-video",
        args.train_video,
        "--audio",
        args.audio,
        "--max-steps",
        str(args.max_steps),
        "--workspace",
        args.workspace,
        "--clip-name",
        args.clip_name,
    ]
    if args.train_audio:
        cmd.extend(["--train-audio", args.train_audio])
    if args.infer_video:
        cmd.extend(["--infer-video", args.infer_video])
    subprocess.run(cmd, check=True)


def cmd_wav2lip_infer(args: argparse.Namespace) -> None:
    """Official Rudrabha/Wav2Lip inference (cloned under external/Wav2Lip)."""
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parent / "scripts" / "wav2lip_infer.py"
    cmd: list[str] = [
        sys.executable,
        str(script),
        "--face",
        args.face,
        "--audio",
        args.audio,
        "--outfile",
        args.output,
        "--resize-factor",
        str(args.resize_factor),
    ]
    if args.pads is not None:
        cmd.extend(["--pads", *[str(x) for x in args.pads]])
    if args.nosmooth:
        cmd.append("--nosmooth")
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    subprocess.run(cmd, check=True)


def cmd_eval(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    from lipsync import LipSyncTrainer
    from lipsync.composite import (
        composite_mouth_region,
        mouth_composite_kwargs_from_inference,
        pool_audio_sequence,
    )
    from lipsync.data import LipSyncDataset
    from lipsync.evaluation import LipSyncEvaluator

    trainer = LipSyncTrainer.from_checkpoint(args.checkpoint, device=args.device)
    ds = LipSyncDataset(args.data_root, split=args.split)
    collate = getattr(ds, "collate_fn", None)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=2,
        collate_fn=collate,
    )

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

            audio_seq = trainer._core.audio_encoder(mel)
            audio_emb = pool_audio_sequence(audio_seq, trainer._core._audio_pool_mode())
            id_emb = trainer._core.identity_encoder(ref_face)

            masked = face.clone()
            masked[:, :, face.shape[-2] // 2 :, :] = 0.0
            masked_4ch = torch.cat([masked, torch.zeros_like(masked[:, :1])], dim=1)

            rgb, alpha = trainer._core.generator(masked_4ch, audio_emb, id_emb)
            mc_kw = mouth_composite_kwargs_from_inference(trainer.config.to_dict().get("inference", {}))
            pred = composite_mouth_region(
                face, rgb, alpha, trainer._core._mouth_blend_frac(), **mc_kw
            )

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
    psnr_v = summary.get("psnr")
    if psnr_v is not None:
        if psnr_v >= 28.0:
            print(
                f"\nPSNR {psnr_v:.2f} dB: strong reconstruction on this split "
                "(good target: >= 28 dB on small/overfit setups)."
            )
        elif psnr_v >= 22.0:
            print(
                f"\nPSNR {psnr_v:.2f} dB: moderate; try more epochs or tune losses in configs/base.yaml."
            )
        else:
            print(
                f"\nPSNR {psnr_v:.2f} dB: low; check alignment, lr, or set losses.w_adv: 0 in YAML."
            )


# ---------------------------------------------------------------------------
# New framework-level operations
# ---------------------------------------------------------------------------

def cmd_data_validate(args: argparse.Namespace) -> None:
    from lipsync.data import validate_dataset

    report = validate_dataset(args.data_root, split=args.split)
    data = report.to_dict()

    print("=== Data Validation Report ===")
    print(f"Data root      : {data['data_root']}")
    print(f"Total samples  : {data['total_samples']}")
    print(f"Valid samples  : {data['valid_samples']}")
    print(f"Errors         : {data['error_count']}")
    print(f"Warnings       : {data['warning_count']}")

    max_show = args.max_issues
    if data["issues"] and max_show > 0:
        print("\nTop issues:")
        for issue in data["issues"][:max_show]:
            print(f"  [{issue['severity']}] {issue['sample_id']}: {issue['message']}")

    if args.output:
        from lipsync.ops import save_json
        save_json(args.output, data)
        print(f"\nSaved report -> {args.output}")


def cmd_doctor(args: argparse.Namespace) -> None:
    from lipsync.ops import doctor
    from lipsync.runtime import runtime_report_text

    report = doctor()
    print(runtime_report_text(report))

    if args.output:
        from lipsync.ops import save_json
        save_json(args.output, report)
        print(f"\nSaved report -> {args.output}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    from lipsync.ops import benchmark, save_json

    result = benchmark(
        device=args.device,
        batch_size=args.batch_size,
        steps=args.steps,
        image_size=args.image_size,
    )

    print("=== Benchmark ===")
    for k, v in result.items():
        print(f"  {k:16s}: {v}")

    if args.output:
        save_json(args.output, result)
        print(f"\nSaved benchmark -> {args.output}")


def cmd_profile_realtime(args: argparse.Namespace) -> None:
    import json
    from pathlib import Path
    from lipsync.ops import profile_realtime, save_json

    files = sorted(Path(args.meta_dir).glob("*_meta.json"))
    if not files:
        raise FileNotFoundError(f"No *_meta.json in {args.meta_dir}")

    gen_times = []
    clip_durs = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        gen = float(d.get("generation_time_s", 0.0))
        fps = float(d.get("source_fps", d.get("output_fps", 25.0)))
        n = int(d.get("n_source_frames", 0))
        clip_dur = (n / fps) if fps > 0 else 0.0
        if gen > 0 and clip_dur > 0:
            gen_times.append(gen)
            clip_durs.append(clip_dur)

    stats = profile_realtime(gen_times, clip_durs)
    stats.update({"clips": len(gen_times), "meta_dir": args.meta_dir})

    print("=== Realtime Profile ===")
    for k, v in stats.items():
        print(f"  {k:20s}: {v}")

    if args.output:
        save_json(args.output, stats)
        print(f"\nSaved profile -> {args.output}")


def cmd_data_download(args: argparse.Namespace) -> None:
    from lipsync.data import GRID_DATASET_REF, download_grid_from_kaggle

    report = download_grid_from_kaggle(
        output_dir=args.output_dir,
        dataset_ref=args.dataset or GRID_DATASET_REF,
        unzip=not args.no_unzip,
        force=args.force,
    )
    print("=== Data Download ===")
    for k, v in report.items():
        print(f"  {k:16s}: {v}")


def cmd_data_prepare(args: argparse.Namespace) -> None:
    from lipsync.data import prepare_grid_dataset

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
    print("=== Data Prepare ===")
    for k, v in summary.items():
        print(f"  {k:16s}: {v}")


def cmd_dataset_verify(args: argparse.Namespace) -> None:
    from lipsync.evaluation.reconstruction import run_dataset_reconstruction_verify

    summary = run_dataset_reconstruction_verify(
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
    print("=== Dataset reconstruction (holdout) ===")
    for k, v in summary.items():
        if k != "clips":
            print(f"  {k}: {v}")
    for c in summary.get("clips", []):
        print(f"  clip {c}")


def cmd_apply_batch(args: argparse.Namespace) -> None:
    import json
    from lipsync import apply_batch, apply_batch_pairs

    if args.pairs_json:
        payload = json.loads(Path(args.pairs_json).read_text(encoding="utf-8"))
        pairs = [(item["video"], item["audio"], item["output"]) for item in payload]
        outputs = apply_batch_pairs(
            checkpoint=args.checkpoint,
            pairs=pairs,
            device=args.device,
            use_sr=args.use_sr,
            sr_backend=args.sr_backend,
            fps=args.fps,
        )
        print(f"Generated {len(outputs)} outputs from pairs JSON.")
        return

    if not args.video or not args.audio:
        raise ValueError("For single apply-batch run, --video and --audio are required.")

    out = apply_batch(
        checkpoint=args.checkpoint,
        video=args.video,
        audio=args.audio,
        output=args.output,
        device=args.device,
        use_sr=args.use_sr,
        sr_backend=args.sr_backend,
        fps=args.fps,
    )
    print(f"Generated output -> {out}")


def cmd_apply_realtime(args: argparse.Namespace) -> None:
    from lipsync import apply_realtime

    print(f"Starting realtime apply at ws/http://{args.host}:{args.port}")
    apply_realtime(
        checkpoint=args.checkpoint,
        host=args.host,
        port=args.port,
        device=args.device,
        use_sr=args.use_sr,
        sr_backend=args.sr_backend,
        fps=args.fps,
        audio_window_ms=args.audio_window_ms,
    )


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
    p_train.add_argument("--profile", type=str, default=None,
                         choices=["cpu-safe", "gpu-fast", "gpu-quality"],
                         help="Runtime profile presets")
    p_train.add_argument("--device", type=str, default="auto")
    p_train.add_argument("--no-fp16", action="store_true")
    p_train.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override checkpoint directory (default: from YAML or checkpoints)",
    )
    p_train.add_argument("--early-stopping", type=int, default=0, metavar="PATIENCE")
    p_train.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p_train.add_argument("--audio-sample-rate", type=int, default=None)
    p_train.add_argument("--audio-n-mels", type=int, default=None)
    p_train.add_argument("--audio-window", type=int, default=None)
    p_train.add_argument("--video-face-size", type=int, default=None)
    p_train.add_argument("--video-lip-size", type=int, default=None)
    p_train.add_argument("--video-target-fps", type=float, default=None)
    p_train.add_argument("--sync-window", type=int, default=None)
    p_train.add_argument("--temporal-radius", type=int, default=None)
    p_train.add_argument("--mouth-region-weight", type=float, default=None)
    p_train.add_argument("--smoothing", type=float, default=None)
    p_train.add_argument("--paste-mode", type=str, default=None, choices=["direct", "seamless"])
    p_train.add_argument(
        "--keep-original-audio",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Override inference.keep_original_audio",
    )

    # --- static-video ---
    p_static = sub.add_parser(
        "static-video",
        help="Create MP4 from a still image repeated for the duration of a WAV",
    )
    p_static.add_argument("--image", type=str, required=True)
    p_static.add_argument("--audio", type=str, required=True)
    p_static.add_argument("--output", type=str, required=True)
    p_static.add_argument("--fps", type=float, default=25.0)

    # --- wav2lip-infer (official GitHub pipeline, pretrained) ---
    p_w2l = sub.add_parser(
        "wav2lip-infer",
        help="Lip-sync via Rudrabha/Wav2Lip (external/Wav2Lip + HF weights); not our trainer",
    )
    p_w2l.add_argument("--face", type=str, required=True, help="Video or image with face")
    p_w2l.add_argument("--audio", type=str, required=True, help="Driving audio")
    p_w2l.add_argument("--output", type=str, default="wav2lip_out.mp4", help="Output MP4")
    p_w2l.add_argument(
        "--pads",
        nargs=4,
        type=int,
        default=None,
        metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),
        help="Face bbox padding (default in script: 0 10 0 0)",
    )
    p_w2l.add_argument("--resize-factor", type=int, default=1)
    p_w2l.add_argument("--nosmooth", action="store_true")
    p_w2l.add_argument("--checkpoint", type=str, default=None, help="Path to wav2lip *.pth")

    p_w2l_user = sub.add_parser(
        "wav2lip-user-train",
        help="Preprocess your video, finetune Wav2Lip (short run), write lip_sync_output.mp4",
    )
    p_w2l_user.add_argument("--train-video", type=str, required=True)
    p_w2l_user.add_argument("--audio", type=str, required=True, help="Driving audio for final MP4")
    p_w2l_user.add_argument(
        "--train-audio",
        type=str,
        default=None,
        help="Audio aligned with train-video if MP4 has no sound (default: same as --audio)",
    )
    p_w2l_user.add_argument(
        "--infer-video",
        type=str,
        default=None,
        help="Video for inference (default: same as --train-video)",
    )
    p_w2l_user.add_argument("--workspace", type=str, default="external/wav2lip_user_workspace")
    p_w2l_user.add_argument("--clip-name", type=str, default="user_clip")
    p_w2l_user.add_argument("--max-steps", type=int, default=500)

    # --- infer ---
    p_infer = sub.add_parser("infer", help="Run lip-sync inference on a video")
    p_infer.add_argument("--checkpoint", type=str, required=True)
    p_infer.add_argument("--audio", type=str, required=True, help="Driving audio (WAV)")
    p_infer.add_argument("--video", type=str, required=True, help="Source video")
    p_infer.add_argument("--output", type=str, default="output.mp4")
    p_infer.add_argument("--device", type=str, default="auto")
    p_infer.add_argument("--use-sr", action="store_true", help="Apply super-resolution")
    p_infer.add_argument(
        "--infer-blend",
        action="store_true",
        help="Override checkpoint: use alpha blend in lip ROI (reduces rectangular noise vs raw paste)",
    )
    p_infer.add_argument(
        "--mouth-composite-mode",
        type=str,
        default=None,
        choices=("blend", "paste", "hard_lower"),
        help="Override checkpoint inference mouth compositing (hard_lower = paste)",
    )
    p_infer.add_argument(
        "--mouth-alpha-min",
        type=float,
        default=None,
        help="When using blend, minimum alpha in ROI (default from checkpoint or 0.45)",
    )

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
    p_eval = sub.add_parser("eval", help="Evaluate a trained model (PSNR/SSIM on frames)")
    p_eval.add_argument("--checkpoint", type=str, required=True)
    p_eval.add_argument("--data-root", type=str, required=True)
    p_eval.add_argument(
        "--split",
        type=str,
        default="val",
        choices=("train", "val", "test"),
        help="Metadata split to score (primary metric: mean PSNR)",
    )
    p_eval.add_argument("--batch-size", type=int, default=16)
    p_eval.add_argument("--device", type=str, default="auto")

    # --- data-validate ---
    p_validate = sub.add_parser("data-validate", help="Validate dataset integrity")
    p_validate.add_argument("--data-root", type=str, required=True)
    p_validate.add_argument("--split", type=str, default="train")
    p_validate.add_argument("--max-issues", type=int, default=20)
    p_validate.add_argument("--output", type=str, default=None)

    # --- doctor ---
    p_doctor = sub.add_parser("doctor", help="Check runtime/dependency health")
    p_doctor.add_argument("--output", type=str, default=None)

    # --- benchmark ---
    p_bench = sub.add_parser("benchmark", help="Run synthetic throughput/latency benchmark")
    p_bench.add_argument("--device", type=str, default="auto")
    p_bench.add_argument("--batch-size", type=int, default=1)
    p_bench.add_argument("--steps", type=int, default=50)
    p_bench.add_argument("--image-size", type=int, default=256)
    p_bench.add_argument("--output", type=str, default=None)

    # --- profile-realtime ---
    p_profile = sub.add_parser("profile-realtime", help="Compute p50/p95/p99 from *_meta.json files")
    p_profile.add_argument("--meta-dir", type=str, required=True)
    p_profile.add_argument("--output", type=str, default=None)

    # --- data-download ---
    p_download = sub.add_parser("data-download", help="Download GRID dataset via Kaggle API")
    p_download.add_argument("--output-dir", type=str, required=True)
    p_download.add_argument("--dataset", type=str, default=None, help="Kaggle dataset ref owner/name")
    p_download.add_argument("--no-unzip", action="store_true")
    p_download.add_argument("--force", action="store_true")

    # --- data-prepare ---
    p_prepare = sub.add_parser("data-prepare", help="Prepare GRID raw data into framework format")
    p_prepare.add_argument("--input-root", type=str, required=True)
    p_prepare.add_argument("--output-root", type=str, required=True)
    p_prepare.add_argument("--speakers", type=str, default="")
    p_prepare.add_argument("--val-ratio", type=float, default=0.1)
    p_prepare.add_argument("--seed", type=int, default=42)
    p_prepare.add_argument("--face-size", type=int, default=256)
    p_prepare.add_argument("--lip-size", type=int, default=96)
    p_prepare.add_argument("--fps", type=float, default=25.0)
    p_prepare.add_argument("--device", type=str, default="cpu")
    p_prepare.add_argument("--min-frames", type=int, default=20)
    p_prepare.add_argument("--limit", type=int, default=0)
    p_prepare.add_argument("--overwrite", action="store_true")
    p_prepare.add_argument(
        "--split-by-speaker",
        action="store_true",
        help="Put all clips of a speaker in one split (train or val) to reduce identity leakage",
    )

    # --- dataset-verify ---
    p_dv = sub.add_parser(
        "dataset-verify",
        help="Reconstruction check on val/test: static (aug) first frame + clip mel vs GT",
    )
    p_dv.add_argument("--checkpoint", type=str, required=True)
    p_dv.add_argument("--data-root", type=str, required=True)
    p_dv.add_argument("--split", type=str, default="val", choices=("val", "test"))
    p_dv.add_argument("--num-clips", type=int, default=5)
    p_dv.add_argument("--seed", type=int, default=42)
    p_dv.add_argument("--no-augment", action="store_true")
    p_dv.add_argument("--out-dir", type=str, default="verify_recon_out")
    p_dv.add_argument("--device", type=str, default="auto")
    p_dv.add_argument("--batch-size", type=int, default=16)
    p_dv.add_argument("--no-mux", action="store_true")

    # --- apply-batch ---
    p_apply_batch = sub.add_parser("apply-batch", help="Apply a trained model in batch mode")
    p_apply_batch.add_argument("--checkpoint", type=str, required=True)
    p_apply_batch.add_argument("--video", type=str, default=None)
    p_apply_batch.add_argument("--audio", type=str, default=None)
    p_apply_batch.add_argument("--output", type=str, default="output.mp4")
    p_apply_batch.add_argument("--pairs-json", type=str, default=None, help="JSON list with video/audio/output")
    p_apply_batch.add_argument("--device", type=str, default="auto")
    p_apply_batch.add_argument("--use-sr", action="store_true")
    p_apply_batch.add_argument("--sr-backend", type=str, default="gfpgan")
    p_apply_batch.add_argument("--fps", type=float, default=25.0)

    # --- apply-realtime ---
    p_apply_rt = sub.add_parser("apply-realtime", help="Run realtime model apply server")
    p_apply_rt.add_argument("--checkpoint", type=str, required=True)
    p_apply_rt.add_argument("--host", type=str, default="0.0.0.0")
    p_apply_rt.add_argument("--port", type=int, default=8000)
    p_apply_rt.add_argument("--device", type=str, default="auto")
    p_apply_rt.add_argument("--use-sr", action="store_true")
    p_apply_rt.add_argument("--sr-backend", type=str, default="gfpgan")
    p_apply_rt.add_argument("--fps", type=float, default=25.0)
    p_apply_rt.add_argument("--audio-window-ms", type=float, default=200.0)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "train": cmd_train,
        "static-video": cmd_static_video,
        "wav2lip-infer": cmd_wav2lip_infer,
        "wav2lip-user-train": cmd_wav2lip_user_pipeline,
        "infer": cmd_infer,
        "serve": cmd_serve,
        "export": cmd_export,
        "eval": cmd_eval,
        "data-validate": cmd_data_validate,
        "doctor": cmd_doctor,
        "benchmark": cmd_benchmark,
        "profile-realtime": cmd_profile_realtime,
        "data-download": cmd_data_download,
        "data-prepare": cmd_data_prepare,
        "apply-batch": cmd_apply_batch,
        "apply-realtime": cmd_apply_realtime,
        "dataset-verify": cmd_dataset_verify,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

"""Core GAN Trainer with mixed precision, gradient checkpointing and checkpointing."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from ..losses import AdversarialLoss, HuberLoss, PerceptualLoss, SyncLoss
from ..models import (
    AudioEncoder,
    IdentityEncoder,
    LipSyncGenerator,
    PatchDiscriminator,
    SyncNet,
)
from ..optimizers import build_optimizer, build_scheduler
from .callbacks import Callback, ModelCheckpoint, ProgressBar
from .checkpoint import make_metadata, migrate_to_v2, validate_checkpoint_v2
from ..composite import (
    composite_mouth_region,
    lip_roi_slices,
    mouth_composite_kwargs_from_inference,
    pool_audio_sequence,
)
from ..config.schema import merge_inference_defaults
from ..runtime import assert_runtime_compatible, set_deterministic


class LipSyncTrainerCore:
    """Full GAN training loop for the lip-sync model.

    Manages:
    - Generator (AudioEncoder + IdentityEncoder + LipSyncGenerator)
    - Discriminator (PatchDiscriminator + SyncNet)
    - Mixed-precision training (torch.cuda.amp)
    - Gradient checkpointing for memory efficiency
    - Configurable losses, optimizers, schedulers
    - Callback system for logging, early stopping, checkpointing

    Args:
        config: Training configuration dict (mirrors configs/base.yaml).
        device: 'cuda' | 'cpu' | 'auto'.
    """

    def __init__(self, config: dict[str, Any], device: str = "auto") -> None:
        self.config = config
        # Deterministic mode for reproducibility
        if self.config.get("deterministic", False):
            set_deterministic(self.config.get("seed", 42))

        # Runtime compatibility checks (fail-fast with clear diagnostics)
        assert_runtime_compatible(
            require_torchvision=self.config.get("losses", {}).get("w_perceptual", 0) > 0,
            require_torchaudio=False,
            require_cv2=False,
        )

        self.device = self._resolve_device(device)
        self.should_stop = False
        self._loaded_checkpoint_path: Path | None = None
        self._build_models()
        self._build_losses()
        self._build_optimizers()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_models(self) -> None:
        mc = self.config.get("model", {})
        self.audio_encoder = AudioEncoder(
            n_mels=mc.get("n_mels", 80),
            d_model=mc.get("audio_d_model", 256),
            num_heads=mc.get("audio_heads", 4),
            num_layers=mc.get("audio_layers", 6),
            embed_dim=mc.get("audio_embed_dim", 512),
        ).to(self.device)

        self.identity_encoder = IdentityEncoder(
            embed_dim=mc.get("identity_embed_dim", 512),
            pretrained=mc.get("pretrained_identity", True),
        ).to(self.device)

        self.generator = LipSyncGenerator(
            in_channels=mc.get("gen_in_channels", 4),
            base_ch=mc.get("gen_base_ch", 64),
            num_encoder_blocks=mc.get("gen_depth", 4),
            audio_dim=mc.get("audio_embed_dim", 512),
            identity_dim=mc.get("identity_embed_dim", 512),
        ).to(self.device)

        self.discriminator = PatchDiscriminator(
            in_channels=mc.get("disc_in_channels", 3),
            base_ch=mc.get("disc_base_ch", 64),
        ).to(self.device)

        self.syncnet = SyncNet(
            embed_dim=mc.get("sync_embed_dim", 256),
        ).to(self.device)

        # Gradient checkpointing saves memory during training
        if self.config.get("gradient_checkpointing", False):
            self.generator.encoders = nn.ModuleList(
                [torch.utils.checkpoint.checkpoint_wrapper(m) for m in self.generator.encoders]
            )

    # ------------------------------------------------------------------
    # Loss construction
    # ------------------------------------------------------------------

    def _build_losses(self) -> None:
        lc = self.config.get("losses", {})
        self._huber_delta = float(lc.get("huber_delta", 1.0))
        self.recon_loss = HuberLoss(delta=self._huber_delta)

        w_perc = lc.get("w_perceptual", 1.0)
        if w_perc > 0:
            try:
                self.perceptual_loss = PerceptualLoss(
                    layers=lc.get("perceptual_layers", ["relu3_3", "relu4_3"])
                ).to(self.device)
            except Exception as e:
                print(f"[Trainer] PerceptualLoss unavailable ({e}), disabling.")
                self.perceptual_loss = None
                w_perc = 0.0
        else:
            self.perceptual_loss = None

        self.sync_loss = SyncLoss()
        self.adv_loss = AdversarialLoss(mode=lc.get("adv_mode", "hinge"))

        self.loss_weights = {
            "recon": lc.get("w_recon", 10.0),
            "perceptual": w_perc,
            "sync": lc.get("w_sync", 0.5),
            "temporal": lc.get("w_temporal", 0.1),
            "adv": lc.get("w_adv", 1.0),
        }

    def _weighted_recon_and_lip_huber(
        self,
        generated: torch.Tensor,
        gt: torch.Tensor,
        mc_kw: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Huber recon with higher weight inside lip ROI (matches Wav2Lip-style pixel supervision).

        Mean Huber over the full frame masks mouth errors: most pixels match outside the small
        pasted box, so the network can keep high-frequency garbage inside the mouth ROI.
        """
        lc = self.config.get("losses", {})
        boost = float(lc.get("lip_recon_boost", 1.0))
        base = F.huber_loss(
            generated, gt, delta=self._huber_delta, reduction="none"
        )
        _, _, H, W = generated.shape
        if mc_kw.get("composite_scope") == "lip_box" and boost > 1.0:
            y0, y1, x0, x1 = mc_kw["lip_box_fracs"]
            sy, sx = lip_roi_slices(H, W, (y0, y1, x0, x1))
            w = torch.ones_like(base)
            w[:, :, sy, sx] = boost
            l_recon = (base * w).sum() / w.sum().clamp(min=1e-8)
        else:
            l_recon = base.mean()

        if mc_kw.get("composite_scope") == "lip_box":
            y0, y1, x0, x1 = mc_kw["lip_box_fracs"]
            sy, sx = lip_roi_slices(H, W, (y0, y1, x0, x1))
            l_lip_only = F.huber_loss(
                generated[:, :, sy, sx],
                gt[:, :, sy, sx],
                delta=self._huber_delta,
                reduction="mean",
            )
        else:
            mh = max(1, min(H - 1, int(H * self._mouth_blend_frac())))
            l_lip_only = F.huber_loss(
                generated[:, :, mh:, :],
                gt[:, :, mh:, :],
                delta=self._huber_delta,
                reduction="mean",
            )
        return l_recon, l_lip_only

    # ------------------------------------------------------------------
    # Optimizer construction
    # ------------------------------------------------------------------

    def _build_optimizers(self) -> None:
        oc = self.config.get("optimizer", {})
        g_params = (
            list(self.audio_encoder.parameters())
            + list(self.identity_encoder.parameters())
            + list(self.generator.parameters())
            + list(self.syncnet.parameters())
        )
        d_params = list(self.discriminator.parameters())

        opt_name = oc.get("name", "adamw")
        skip = {"name", "gradient_clipping", "d_lr"}
        g_kwargs = {k: v for k, v in oc.items() if k not in skip}
        g_kwargs.setdefault("lr", 2e-4)

        self.opt_g = build_optimizer(g_params, opt_name, **g_kwargs)
        d_lr = oc.get("d_lr", g_kwargs.get("lr", 2e-4))
        self.opt_d = build_optimizer(d_params, opt_name, lr=d_lr)

        sc = self.config.get("scheduler", {})
        if sc.get("name"):
            sched_kwargs = {k: v for k, v in sc.items() if k != "name"}
            self.sched_g = build_scheduler(sc["name"], self.opt_g, **sched_kwargs)
            self.sched_d = build_scheduler(sc["name"], self.opt_d, **sched_kwargs)
        else:
            self.sched_g = None
            self.sched_d = None

        self.scaler_g = GradScaler(enabled=self.config.get("fp16", True))
        self.scaler_d = GradScaler(enabled=self.config.get("fp16", True))

    def _mouth_blend_frac(self) -> float:
        return float(self.config.get("inference", {}).get("mouth_blend_from", 0.42))

    def _audio_pool_mode(self) -> str:
        return self.config.get("inference", {}).get("audio_embed_pool", "last")

    def _mouth_composite_kw(self) -> dict[str, Any]:
        """Same mouth compositing as inference (YAML ``inference`` block).

        Using different train vs infer composite (e.g. lower-half blend vs lip-box paste)
        lets the generator dump arbitrary RGB where blend alpha is low; paste inference
        then shows a noisy mouth rectangle.
        """
        return mouth_composite_kwargs_from_inference(self.config.get("inference", {}))

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------

    def _g_step(self, batch: dict) -> dict[str, float]:
        mel = batch["mel"].to(self.device)           # (B, T, n_mels)
        face = batch["face"].to(self.device)         # (B, 3, H, W)
        ref_face = batch["ref_face"].to(self.device) # (B, 3, H, W)
        gt = batch["gt_frame"].to(self.device)       # (B, 3, H, W)

        # Mask lower half of face for generator input
        masked_face = face.clone()
        masked_face[:, :, face.shape[-2] // 2 :, :] = 0.0
        masked_face_with_ch = torch.cat(
            [masked_face, torch.zeros_like(masked_face[:, :1])], dim=1
        )  # (B, 4, H, W)

        self.opt_g.zero_grad()
        with autocast(enabled=self.config.get("fp16", True)):
            audio_seq = self.audio_encoder(mel)
            audio_emb = pool_audio_sequence(audio_seq, self._audio_pool_mode())
            identity_emb = self.identity_encoder(ref_face)
            rgb, alpha = self.generator(masked_face_with_ch, audio_emb, identity_emb)

            mc_kw = self._mouth_composite_kw()
            generated = composite_mouth_region(
                face, rgb, alpha, self._mouth_blend_frac(), **mc_kw
            )

            # Losses (mouth ROI–weighted recon so val_loss reflects lip quality)
            l_recon, l_lip_huber = self._weighted_recon_and_lip_huber(generated, gt, mc_kw)
            w_adv = float(self.loss_weights.get("adv", 1.0))
            if w_adv > 0.0:
                fake_logits = self.discriminator(generated)
                l_adv = self.adv_loss.generator_loss(fake_logits)
            else:
                l_adv = torch.tensor(0.0, device=self.device)
            l_total = self.loss_weights["recon"] * l_recon + w_adv * l_adv
            l_perc = torch.tensor(0.0, device=self.device)
            if self.perceptual_loss is not None and self.loss_weights["perceptual"] > 0:
                l_perc = self.perceptual_loss(generated, gt)
                l_total = l_total + self.loss_weights["perceptual"] * l_perc

            l_sync = torch.tensor(0.0, device=self.device)
            if batch.get("sync_lips") is not None and self.loss_weights["sync"] > 0:
                sl = batch["sync_lips"].to(self.device)
                mel_ch = mel.unsqueeze(1)
                ae, ve = self.syncnet(mel_ch, sl)
                l_sync = self.sync_loss(ae, ve)
                l_total = l_total + self.loss_weights["sync"] * l_sync

            l_temp = torch.tensor(0.0, device=self.device)
            wt = self.loss_weights["temporal"]
            if wt > 0 and generated.shape[-1] > 1:
                dh_g = generated[:, :, :, 1:] - generated[:, :, :, :-1]
                dh_t = gt[:, :, :, 1:] - gt[:, :, :, :-1]
                l_temp = torch.nn.functional.l1_loss(dh_g, dh_t.detach())
                l_total = l_total + wt * l_temp

            l_mouth_a = torch.tensor(0.0, device=self.device)
            w_ma = float(self.loss_weights.get("w_mouth_alpha", 0.0))
            if w_ma > 0.0:
                _, _, H, W = face.shape
                tgt = float(self.config.get("losses", {}).get("mouth_alpha_target", 0.25))
                scope = mc_kw.get("composite_scope", "lower_half")
                if scope == "lip_box":
                    y0, y1, x0, x1 = mc_kw["lip_box_fracs"]
                    sy, sx = lip_roi_slices(H, W, (y0, y1, x0, x1))
                    a_zone = alpha[:, :, sy, sx]
                else:
                    mh = max(1, min(H - 1, int(H * self._mouth_blend_frac())))
                    a_zone = alpha[:, :, mh:, :]
                l_mouth_a = torch.nn.functional.relu(
                    torch.as_tensor(tgt, device=face.device, dtype=alpha.dtype) - a_zone
                ).mean()
                l_total = l_total + w_ma * l_mouth_a

        self.scaler_g.scale(l_total).backward()
        if self.config.get("gradient_clipping", {}).get("enabled", True):
            self.scaler_g.unscale_(self.opt_g)
            torch.nn.utils.clip_grad_norm_(
                list(self.audio_encoder.parameters())
                + list(self.identity_encoder.parameters())
                + list(self.generator.parameters())
                + list(self.syncnet.parameters()),
                self.config.get("gradient_clipping", {}).get("max_norm", 1.0),
            )
        self.scaler_g.step(self.opt_g)
        self.scaler_g.update()

        return {
            "g_total": l_total.item(),
            "g_recon": l_recon.item(),
            "g_recon_lip": l_lip_huber.item(),
            "g_perc": l_perc.item(),
            "g_adv": l_adv.item(),
            "g_sync": l_sync.item(),
            "g_temp": l_temp.item(),
            "g_mouth_a": l_mouth_a.item(),
        }

    def _d_step(self, batch: dict) -> dict[str, float]:
        mel = batch["mel"].to(self.device)
        face = batch["face"].to(self.device)
        ref_face = batch["ref_face"].to(self.device)
        gt = batch["gt_frame"].to(self.device)

        masked_face = face.clone()
        masked_face[:, :, face.shape[-2] // 2 :, :] = 0.0
        masked_face_4ch = torch.cat([masked_face, torch.zeros_like(masked_face[:, :1])], dim=1)

        self.opt_d.zero_grad()
        with autocast(enabled=self.config.get("fp16", True)):
            with torch.no_grad():
                audio_seq = self.audio_encoder(mel)
                audio_emb = pool_audio_sequence(audio_seq, self._audio_pool_mode())
                identity_emb = self.identity_encoder(ref_face)
                rgb, alpha = self.generator(masked_face_4ch, audio_emb, identity_emb)
                mc_kw = self._mouth_composite_kw()
                generated = composite_mouth_region(
                    face, rgb, alpha, self._mouth_blend_frac(), **mc_kw
                )

            real_logits = self.discriminator(gt)
            fake_logits = self.discriminator(generated.detach())
            l_d = self.adv_loss.discriminator_loss(real_logits, fake_logits)

        self.scaler_d.scale(l_d).backward()
        self.scaler_d.step(self.opt_d)
        self.scaler_d.update()
        return {"d_loss": l_d.item()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        callbacks: list[Callback] | None = None,
        num_workers: int = 4,
    ) -> None:
        """Train for *epochs* epochs.

        Args:
            train_dataset: Training dataset.
            val_dataset: Optional validation dataset.
            epochs: Overrides config value.
            batch_size: Overrides config value.
            callbacks: List of Callback instances.
            num_workers: DataLoader workers.
        """
        from ..data.datasets.base_dataset import BaseLipSyncDataset

        epochs = epochs or self.config.get("epochs", 100)
        bs = batch_size or self.config.get("batch_size", 8)

        train_loader = DataLoader(
            train_dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=getattr(train_dataset, "collate_fn", None),
        )
        val_loader = (
            DataLoader(val_dataset, batch_size=bs, num_workers=num_workers)
            if val_dataset
            else None
        )

        cbs = callbacks or [ProgressBar()]
        if not any(isinstance(c, ModelCheckpoint) for c in cbs):
            cbs.append(ModelCheckpoint())

        for cb in cbs:
            cb.on_train_begin(self)

        for epoch in range(1, epochs + 1):
            if self.should_stop:
                break

            for cb in cbs:
                cb.on_epoch_begin(self, epoch)

            self.audio_encoder.train()
            self.identity_encoder.train()
            self.generator.train()
            self.discriminator.train()
            self.syncnet.train()

            epoch_logs: dict[str, float] = {}
            train_disc = float(self.loss_weights.get("adv", 1.0)) > 0.0
            for batch_idx, batch in enumerate(train_loader):
                d_logs = self._d_step(batch) if train_disc else {"d_loss": 0.0}
                g_logs = self._g_step(batch)
                step_logs = {**d_logs, **g_logs}
                for k, v in step_logs.items():
                    epoch_logs[k] = epoch_logs.get(k, 0.0) + v
                for cb in cbs:
                    cb.on_batch_end(self, batch_idx, step_logs)

            n_batches = len(train_loader)
            epoch_logs = {k: v / n_batches for k, v in epoch_logs.items()}

            if val_loader:
                val_logs = self._validate(val_loader)
                epoch_logs.update(val_logs)

            if self.sched_g:
                self.sched_g.step()
            if self.sched_d:
                self.sched_d.step()

            for cb in cbs:
                cb.on_epoch_end(self, epoch, epoch_logs)

        for cb in cbs:
            cb.on_train_end(self)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict[str, float]:
        self.audio_encoder.eval()
        self.generator.eval()
        self.identity_encoder.eval()
        self.syncnet.eval()
        totals: dict[str, float] = {}
        for batch in loader:
            mel = batch["mel"].to(self.device)
            face = batch["face"].to(self.device)
            ref_face = batch["ref_face"].to(self.device)
            gt = batch["gt_frame"].to(self.device)

            masked_face = face.clone()
            masked_face[:, :, face.shape[-2] // 2 :, :] = 0.0
            masked_face_4ch = torch.cat([masked_face, torch.zeros_like(masked_face[:, :1])], dim=1)

            audio_seq = self.audio_encoder(mel)
            audio_emb = pool_audio_sequence(audio_seq, self._audio_pool_mode())
            identity_emb = self.identity_encoder(ref_face)
            rgb, alpha = self.generator(masked_face_4ch, audio_emb, identity_emb)
            mc_kw = self._mouth_composite_kw()
            generated = composite_mouth_region(
                face, rgb, alpha, self._mouth_blend_frac(), **mc_kw
            )
            val_w, val_lip = self._weighted_recon_and_lip_huber(generated, gt, mc_kw)
            totals["val_loss"] = totals.get("val_loss", 0.0) + val_w.item()
            totals["val_lip_huber"] = totals.get("val_lip_huber", 0.0) + val_lip.item()

        n = len(loader)
        return {k: v / n for k, v in totals.items()}

    def save_checkpoint(self, path: str | Path) -> None:
        """Save full model state to *path*."""
        from copy import deepcopy

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = deepcopy(self.config)
        cfg["inference"] = merge_inference_defaults(cfg.get("inference"))
        payload = {
            "audio_encoder": self.audio_encoder.state_dict(),
            "identity_encoder": self.identity_encoder.state_dict(),
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "syncnet": self.syncnet.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "config": cfg,
            "meta": make_metadata(cfg),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model state from *path*."""
        ckpt = torch.load(str(path), map_location=self.device)
        ckpt = migrate_to_v2(ckpt)
        validate_checkpoint_v2(ckpt)
        self.audio_encoder.load_state_dict(ckpt["audio_encoder"])
        self.identity_encoder.load_state_dict(ckpt["identity_encoder"])
        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        if "syncnet" in ckpt:
            self.syncnet.load_state_dict(ckpt["syncnet"])
        self._loaded_checkpoint_path = Path(path).resolve()
        if "opt_g" in ckpt:
            self.opt_g.load_state_dict(ckpt["opt_g"])
        if "opt_d" in ckpt:
            self.opt_d.load_state_dict(ckpt["opt_d"])

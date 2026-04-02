"""Mouth compositing: lip ROI vs lower half."""
from __future__ import annotations

import torch

from lipsync.config import merge_inference_defaults
from lipsync.composite import composite_mouth_region, mouth_composite_kwargs_from_inference


def test_paste_lower_half() -> None:
    face = torch.zeros(1, 3, 8, 8)
    face[:, :, :, :] = 0.5
    rgb = torch.zeros(1, 3, 8, 8)
    rgb[:, :, :, :] = -0.8
    alpha = torch.zeros(1, 1, 8, 8)
    out = composite_mouth_region(
        face, rgb, alpha, mouth_start_frac=0.5, composite_mode="paste", composite_scope="lower_half"
    )
    assert torch.allclose(out[:, :, :4, :], face[:, :, :4, :])
    assert torch.allclose(out[:, :, 4:, :], rgb[:, :, 4:, :])


def test_lip_box_leaves_face_outside_roi() -> None:
    """Only the central ROI changes; corners of the lower half stay original."""
    face = torch.ones(1, 3, 10, 10) * 0.3
    rgb = torch.ones(1, 3, 10, 10) * -0.9
    alpha = torch.zeros(1, 1, 10, 10)
    box = (0.5, 0.9, 0.3, 0.7)  # central band
    out = composite_mouth_region(
        face,
        rgb,
        alpha,
        0.42,
        composite_mode="paste",
        composite_scope="lip_box",
        lip_box_fracs=box,
    )
    sy = slice(5, 9)
    sx = slice(3, 7)
    assert torch.allclose(out[:, :, sy, sx], rgb[:, :, sy, sx])
    # lower-left corner outside x-range still original
    assert torch.isclose(out[0, 0, 8, 1], torch.tensor(0.3)).item()


def test_alpha_floor_in_blend_lower_half() -> None:
    face = torch.ones(1, 3, 8, 8) * 0.2
    rgb = torch.ones(1, 3, 8, 8) * -0.5
    alpha = torch.zeros(1, 1, 8, 8)
    out0 = composite_mouth_region(
        face, rgb, alpha, 0.5, composite_mode="blend", composite_scope="lower_half"
    )
    out1 = composite_mouth_region(
        face,
        rgb,
        alpha,
        0.5,
        composite_mode="blend",
        composite_scope="lower_half",
        alpha_floor=0.5,
    )
    assert out1[:, :, 4:, :].mean() < out0[:, :, 4:, :].mean()


def test_mouth_composite_kwargs_defaults() -> None:
    kw = mouth_composite_kwargs_from_inference({})
    assert kw["composite_mode"] == "blend"
    assert kw["composite_scope"] == "lip_box"
    assert kw["alpha_floor"] == 0.45
    assert len(kw["lip_box_fracs"]) == 4
    assert kw["lip_roi_feather_px"] == 0


def test_merge_inference_defaults_fills_blend() -> None:
    m = merge_inference_defaults({})
    assert m["mouth_composite_mode"] == "blend"
    assert m["lip_roi_feather_px"] == 6
    assert m["audio_embed_pool"] == "last"


def test_batch_processor_inference_matches_trainer_defaults() -> None:
    """Offline infer must use the same composite defaults as training (blend, not paste)."""
    from lipsync.inference.batch.processor import BatchProcessor

    p = BatchProcessor("__missing__.pt")
    p._apply_inference_config({})
    kw = mouth_composite_kwargs_from_inference({})
    assert p._mouth_composite_mode == kw["composite_mode"]
    assert p._mouth_composite_scope == kw["composite_scope"]
    assert p._lip_box_fracs == kw["lip_box_fracs"]
    assert p._lip_roi_feather_px == kw["lip_roi_feather_px"]
    assert p._mouth_alpha_floor == kw["alpha_floor"]


def test_lip_box_feather_softens_border() -> None:
    """With feather > 0, pixels just inside the ROI mix toward the original face."""
    face = torch.ones(1, 3, 16, 16) * 0.2
    rgb = torch.ones(1, 3, 16, 16) * 0.9
    alpha = torch.zeros(1, 1, 16, 16)
    box = (0.25, 0.75, 0.25, 0.75)
    out_hard = composite_mouth_region(
        face,
        rgb,
        alpha,
        0.42,
        composite_mode="paste",
        composite_scope="lip_box",
        lip_box_fracs=box,
        lip_roi_feather_px=0,
    )
    out_soft = composite_mouth_region(
        face,
        rgb,
        alpha,
        0.42,
        composite_mode="paste",
        composite_scope="lip_box",
        lip_box_fracs=box,
        lip_roi_feather_px=2,
    )
    # feather=2 with 8×8 ROI: center (8,8) is far enough inside for full weight (1)
    assert torch.isclose(out_hard[0, 0, 8, 8], out_soft[0, 0, 8, 8]).item()
    # One row below top ROI edge: hard paste is rgb; feathered mix is strictly between face and rgb
    r = 5
    c = 8
    assert torch.isclose(out_hard[0, 0, r, c], torch.tensor(0.9)).item()
    v = out_soft[0, 0, r, c].item()
    assert 0.2 < v < 0.9

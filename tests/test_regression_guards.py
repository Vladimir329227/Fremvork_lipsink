"""Regression guard tests for speed/shape invariants."""

import torch

from lipsync.models.generator.lipsync_gen import LipSyncGenerator


def test_generator_output_shape_guard():
    gen = LipSyncGenerator(in_channels=4, base_ch=32, num_encoder_blocks=3, audio_dim=256, identity_dim=256)
    rgb, alpha = gen(torch.randn(2, 4, 128, 128), torch.randn(2, 256), torch.randn(2, 256))
    assert rgb.shape == (2, 3, 128, 128)
    assert alpha.shape == (2, 1, 128, 128)


def test_generator_no_nan_guard():
    gen = LipSyncGenerator(in_channels=4, base_ch=32, num_encoder_blocks=3, audio_dim=256, identity_dim=256)
    rgb, alpha = gen(torch.randn(1, 4, 128, 128), torch.randn(1, 256), torch.randn(1, 256))
    assert not torch.isnan(rgb).any()
    assert not torch.isnan(alpha).any()

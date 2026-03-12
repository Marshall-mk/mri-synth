"""Tests for k-space artifact functions."""

import torch

from mri_synth.artifacts.kspace import (
    apply_aliasing,
    apply_kspace_motion_ghosting,
    apply_kspace_spike,
)


class TestMotionGhosting:
    def test_shape_preservation(self, synthetic_volume_no_batch):
        out = apply_kspace_motion_ghosting(synthetic_volume_no_batch, axis=0)
        assert out.shape == synthetic_volume_no_batch.shape

    def test_no_nan_inf(self, synthetic_volume_no_batch):
        out = apply_kspace_motion_ghosting(synthetic_volume_no_batch, axis=1)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_all_axes(self, synthetic_volume_no_batch):
        for axis in range(3):
            out = apply_kspace_motion_ghosting(synthetic_volume_no_batch, axis=axis)
            assert out.shape == synthetic_volume_no_batch.shape


class TestSpike:
    def test_shape_preservation(self, synthetic_volume_no_batch):
        out = apply_kspace_spike(synthetic_volume_no_batch, intensity=5.0)
        assert out.shape == synthetic_volume_no_batch.shape

    def test_no_nan_inf(self, synthetic_volume_no_batch):
        out = apply_kspace_spike(synthetic_volume_no_batch)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_adds_energy(self, synthetic_volume_no_batch):
        out = apply_kspace_spike(synthetic_volume_no_batch, intensity=5.0)
        # Spike adds energy, so total should be >= original
        assert out.sum() >= synthetic_volume_no_batch.sum() * 0.9


class TestAliasing:
    def test_shape_preservation(self, synthetic_volume_no_batch):
        out = apply_aliasing(synthetic_volume_no_batch, axis=0)
        assert out.shape == synthetic_volume_no_batch.shape

    def test_no_nan_inf(self, synthetic_volume_no_batch):
        out = apply_aliasing(synthetic_volume_no_batch, axis=2)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_all_axes(self, synthetic_volume_no_batch):
        for axis in range(3):
            out = apply_aliasing(synthetic_volume_no_batch, axis=axis)
            assert out.shape == synthetic_volume_no_batch.shape

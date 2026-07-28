"""Tests for k-space artifact functions."""

import pytest
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

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_interior_is_untouched(self, axis):
        """Only the wrap bands may change — no global intensity scaling.

        Regression: the old blend divided weights summing to 2 by 1.5, scaling
        every voxel by 4/3 whether or not any wrapped signal reached it.
        """
        n, fold = 32, 0.2
        vol = torch.rand(1, n, n, n)
        out = apply_aliasing(vol, axis=axis, fold_pct=fold)
        band = int(n * fold / 2)
        sa = axis + 1
        interior_in = vol.narrow(sa, band, n - 2 * band)
        interior_out = out.narrow(sa, band, n - 2 * band)
        assert torch.equal(interior_in, interior_out)

    def test_no_wrap_when_anatomy_fits_inside_the_fov(self):
        """An adequate FOV produces no fold-over, as on a real scanner."""
        n = 48
        g = torch.arange(n, dtype=torch.float32)
        zz, yy, xx = torch.meshgrid(g, g, g, indexing="ij")
        r = ((zz - 23) / 12) ** 2 + ((yy - 23) / 11) ** 2 + ((xx - 23) / 11) ** 2
        # Compact support, so "nothing lies outside the FOV" is exactly true
        # rather than true up to a Gaussian tail.
        brain = (torch.exp(-3.0 * r) * (r <= 1.0)).unsqueeze(0)
        assert brain[:, : int(n * 0.1)].max() == 0.0
        out = apply_aliasing(brain, axis=0, fold_pct=0.2)
        assert torch.equal(out, brain)

    def test_wrapped_band_receives_the_opposite_edge(self):
        n, fold = 32, 0.25
        vol = torch.rand(1, n, n, n)
        out = apply_aliasing(vol, axis=0, fold_pct=fold)
        band = int(n * fold / 2)
        added_head = out[:, :band] - vol[:, :band]
        added_tail = out[:, n - band:] - vol[:, n - band:]
        assert torch.allclose(added_head, vol[:, n - band:], atol=1e-6)
        assert torch.allclose(added_tail, vol[:, :band], atol=1e-6)

    def test_does_not_brighten_a_uniform_interior(self):
        vol = torch.full((1, 32, 32, 32), 0.5)
        out = apply_aliasing(vol, axis=1, fold_pct=0.2)
        band = int(32 * 0.2 / 2)
        interior = out.narrow(2, band, 32 - 2 * band)
        assert torch.allclose(interior, torch.full_like(interior, 0.5), atol=1e-6)


class TestArtifactDefaultsMatchConfig:
    def test_spike_default_matches_artifact_config(self):
        """Calling the helper directly must match what the pipeline does."""
        import inspect

        from mri_synth.config import ArtifactConfig

        default = inspect.signature(apply_kspace_spike).parameters[
            "intensity"
        ].default
        assert default == ArtifactConfig().spike_intensity

    def test_default_spike_overlays_rather_than_swamps(self, synthetic_volume_no_batch):
        out = apply_kspace_spike(synthetic_volume_no_batch)
        vol = synthetic_volume_no_batch
        # A stripe overlay perturbs the image; it does not replace it.
        assert out.mean() < vol.mean() * 1.5
        corr = torch.corrcoef(torch.stack([out.flatten(), vol.flatten()]))[0, 1]
        assert corr > 0.9

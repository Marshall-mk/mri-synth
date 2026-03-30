"""Tests for MRIArtifactSimulator."""

import torch

from mri_synth.artifacts.simulator import MRIArtifactSimulator


class TestMRIArtifactSimulator:
    def test_output_shape_with_return_intermediate(self, synthetic_volume):
        sim = MRIArtifactSimulator(
            volume_res=[1.0, 1.0, 1.0],
            target_res=[1.0, 1.0, 1.0],
            preserve_input_shape=True,
            return_intermediate=True,
            enable_obliqueness=False,
        )
        acq_res = torch.tensor([[1.0, 1.0, 5.0]])
        out, fov_masks, true_lr = sim(synthetic_volume, acq_res)
        assert out.shape == synthetic_volume.shape

    def test_output_shape_without_return_intermediate(self, synthetic_volume):
        sim = MRIArtifactSimulator(
            volume_res=[1.0, 1.0, 1.0],
            target_res=[1.0, 1.0, 1.0],
            preserve_input_shape=True,
            return_intermediate=False,
            enable_obliqueness=False,
        )
        acq_res = torch.tensor([[1.0, 1.0, 5.0]])
        out, fov_masks = sim(synthetic_volume, acq_res)
        assert out.shape == synthetic_volume.shape
        assert fov_masks.shape[0] == synthetic_volume.shape[0]
        assert fov_masks.shape[2:] == synthetic_volume.shape[2:]

    def test_fft_downsampling_correct_axis(self, synthetic_volume):
        """FFT cropping should reduce the axis with highest resolution ratio."""
        sim = MRIArtifactSimulator(
            volume_res=[1.0, 1.0, 1.0],
            target_res=[1.0, 1.0, 1.0],
            return_intermediate=True,
            preserve_input_shape=False,
            enable_obliqueness=False,
        )
        acq_res = torch.tensor([[1.0, 1.0, 5.0]])
        out, fov_masks, true_lr = sim(synthetic_volume, acq_res)
        # true_lr should have fewer slices along axis 2 (W)
        assert true_lr.shape[-1] < synthetic_volume.shape[-1]

    def test_no_downsampling_when_isotropic(self, synthetic_volume):
        sim = MRIArtifactSimulator(
            volume_res=[1.0, 1.0, 1.0],
            target_res=[1.0, 1.0, 1.0],
            return_intermediate=True,
            prob_motion=0.0,
            prob_spike=0.0,
            prob_aliasing=0.0,
            prob_noise=0.0,
            enable_obliqueness=False,
        )
        acq_res = torch.tensor([[1.0, 1.0, 1.0]])
        out, fov_masks, true_lr = sim(synthetic_volume, acq_res)
        assert out.shape == synthetic_volume.shape

    def test_fov_masks_binary(self, synthetic_volume):
        """FOV masks should be binary (0=valid, 1=missing)."""
        sim = MRIArtifactSimulator(
            volume_res=[1.0, 1.0, 1.0],
            target_res=[1.0, 1.0, 1.0],
            preserve_input_shape=True,
            return_intermediate=False,
            enable_obliqueness=False,
        )
        acq_res = torch.tensor([[1.0, 1.0, 5.0]])
        _, fov_masks = sim(synthetic_volume, acq_res)
        unique_vals = torch.unique(fov_masks)
        assert all(v in [0.0, 1.0] for v in unique_vals.tolist())

    def test_obliqueness_produces_partial_slices(self):
        """With obliqueness, resampled LR should have partial (zero) regions at extremes."""
        torch.manual_seed(0)
        vol = torch.ones(1, 1, 32, 32, 32)

        sim = MRIArtifactSimulator(
            volume_res=[1.0, 1.0, 1.0],
            target_res=[1.0, 1.0, 1.0],
            preserve_input_shape=True,
            return_intermediate=False,
            enable_obliqueness=True,
            prob_obliqueness=1.0,
            obliqueness_range=15.0,
            prob_motion=0.0,
            prob_spike=0.0,
            prob_aliasing=0.0,
            prob_noise=0.0,
        )
        acq_res = torch.tensor([[1.0, 1.0, 5.0]])
        out, fov_masks = sim(vol, acq_res)
        # Oblique resampling should produce missing regions in the FOV mask
        assert (fov_masks > 0).any(), "Oblique resampling should produce missing regions"

    def test_obliqueness_round_trip_preserves_center(self):
        """Round-trip (aligned→oblique→HR) should preserve brain content in center."""
        torch.manual_seed(0)
        vol = torch.zeros(1, 1, 32, 32, 32)
        vol[:, :, 12:20, 12:20, 12:20] = 1.0

        sim = MRIArtifactSimulator(
            volume_res=[1.0, 1.0, 1.0],
            target_res=[1.0, 1.0, 1.0],
            preserve_input_shape=True,
            return_intermediate=True,
            enable_obliqueness=True,
            prob_obliqueness=1.0,
            obliqueness_range=10.0,
            prob_motion=0.0,
            prob_spike=0.0,
            prob_aliasing=0.0,
            prob_noise=0.0,
        )
        acq_res = torch.tensor([[1.0, 1.0, 5.0]])
        out, fov_masks, true_lr = sim(vol, acq_res)
        # Center of resampled output should still have signal
        center_val = out[0, 0, 14:18, 14:18, 2:4].mean()
        assert center_val > 0.3, (
            f"Center of registered LR should retain brain content (got {center_val:.4f})"
        )

    def test_rotation_angles_stored(self):
        """Rotation angles should be stored in _last_rotation_angles after forward."""
        torch.manual_seed(42)
        sim = MRIArtifactSimulator(
            volume_res=[1.0, 1.0, 1.0],
            target_res=[1.0, 1.0, 1.0],
            preserve_input_shape=True,
            return_intermediate=False,
            enable_obliqueness=True,
            prob_obliqueness=1.0,
            obliqueness_range=15.0,
            prob_motion=0.0,
            prob_spike=0.0,
            prob_aliasing=0.0,
            prob_noise=0.0,
        )
        vol = torch.rand(1, 1, 16, 16, 16)
        sim(vol, torch.tensor([[1.0, 1.0, 5.0]]))
        assert len(sim._last_rotation_angles) == 1
        assert sim._last_rotation_angles[0] is not None
        assert len(sim._last_rotation_angles[0]) == 3  # (rx, ry, rz)

    def test_motion_axis_includes_zero(self):
        """Bug fix: motion axis should include axis 0."""
        sim = MRIArtifactSimulator(
            volume_res=[1.0, 1.0, 1.0],
            target_res=[1.0, 1.0, 1.0],
            return_intermediate=False,
            prob_motion=1.0,
            prob_noise=0.0,
            enable_obliqueness=False,
        )
        for _ in range(100):
            torch.manual_seed(_)
            vol = torch.rand(1, 1, 8, 8, 8)
            sim(vol, torch.tensor([[1.0, 1.0, 1.0]]))

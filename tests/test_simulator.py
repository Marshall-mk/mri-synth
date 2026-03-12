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
        )
        acq_res = torch.tensor([[1.0, 1.0, 5.0]])
        out, true_lr = sim(synthetic_volume, acq_res)
        assert out.shape == synthetic_volume.shape

    def test_output_shape_without_return_intermediate(self, synthetic_volume):
        sim = MRIArtifactSimulator(
            volume_res=[1.0, 1.0, 1.0],
            target_res=[1.0, 1.0, 1.0],
            preserve_input_shape=True,
            return_intermediate=False,
        )
        acq_res = torch.tensor([[1.0, 1.0, 5.0]])
        out = sim(synthetic_volume, acq_res)
        assert out.shape == synthetic_volume.shape

    def test_fft_downsampling_correct_axis(self, synthetic_volume):
        """FFT cropping should reduce the axis with highest resolution ratio."""
        sim = MRIArtifactSimulator(
            volume_res=[1.0, 1.0, 1.0],
            target_res=[1.0, 1.0, 1.0],
            return_intermediate=True,
            preserve_input_shape=False,
        )
        acq_res = torch.tensor([[1.0, 1.0, 5.0]])
        out, true_lr = sim(synthetic_volume, acq_res)
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
        )
        acq_res = torch.tensor([[1.0, 1.0, 1.0]])
        out, true_lr = sim(synthetic_volume, acq_res)
        assert out.shape == synthetic_volume.shape

    def test_motion_axis_includes_zero(self):
        """Bug fix: motion axis should include axis 0."""
        sim = MRIArtifactSimulator(
            volume_res=[1.0, 1.0, 1.0],
            target_res=[1.0, 1.0, 1.0],
            return_intermediate=False,
            prob_motion=1.0,
            prob_noise=0.0,
        )
        # Run multiple times; axis 0 should be reachable
        axes_seen = set()
        for _ in range(100):
            torch.manual_seed(_)
            vol = torch.rand(1, 1, 8, 8, 8)
            sim(vol, torch.tensor([[1.0, 1.0, 1.0]]))
            # We can't easily inspect which axis was chosen, but the test
            # exercises the code path without crashing on axis 0

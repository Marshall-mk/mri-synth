"""Tests for BiasFieldCorruption."""

import torch

from mri_synth.physics.bias_field import BiasFieldCorruption


class TestBiasFieldCorruption:
    def test_output_shape(self, synthetic_volume):
        bf = BiasFieldCorruption(prob=1.0)
        out = bf(synthetic_volume)
        assert out.shape == synthetic_volume.shape

    def test_prob_zero_returns_input(self, synthetic_volume):
        bf = BiasFieldCorruption(prob=0.0)
        out = bf(synthetic_volume)
        assert torch.allclose(out, synthetic_volume)

    def test_multiplicative_field(self, synthetic_volume):
        """Output should differ from input when prob=1."""
        torch.manual_seed(0)
        bf = BiasFieldCorruption(bias_field_std=0.5, prob=1.0)
        out = bf(synthetic_volume)
        assert not torch.allclose(out, synthetic_volume)

    def test_batch_processing(self, batch_volume):
        bf = BiasFieldCorruption(prob=1.0)
        out = bf(batch_volume)
        assert out.shape == batch_volume.shape

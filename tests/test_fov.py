"""Tests for FOV slice dropping."""

import torch

from mri_synth.fov.slice_drop import FOVSliceDrop
from mri_synth.fov.multi_orientation import MultiOrientationFOVDrop, apply_fov_augmentation


class TestFOVSliceDrop:
    def test_dropped_slices_are_zero(self, synthetic_volume):
        dropper = FOVSliceDrop(prob=1.0, min_keep_fraction=0.5, max_keep_fraction=0.7)
        torch.manual_seed(0)
        out, mask = dropper(synthetic_volume, orientation_idx=0)
        # Where mask is 0, output should be 0
        assert (out[mask == 0] == 0).all()

    def test_mask_is_binary(self, synthetic_volume):
        dropper = FOVSliceDrop(prob=1.0)
        _, mask = dropper(synthetic_volume, orientation_idx=1)
        assert set(mask.unique().tolist()).issubset({0.0, 1.0})

    def test_no_drop_when_prob_zero(self, synthetic_volume):
        dropper = FOVSliceDrop(prob=0.0)
        out, mask = dropper(synthetic_volume, orientation_idx=0)
        assert torch.allclose(out, synthetic_volume)
        assert (mask == 1).all()

    def test_output_shape(self, synthetic_volume):
        dropper = FOVSliceDrop(prob=1.0)
        out, mask = dropper(synthetic_volume, orientation_idx=2)
        assert out.shape == synthetic_volume.shape
        assert mask.shape == (1, 1, 32, 32, 32)


class TestMultiOrientationFOVDrop:
    def test_ensure_coverage(self, synthetic_volume):
        """When ensure_coverage=True and all would drop, at least one is kept."""
        stacks = [synthetic_volume.clone() for _ in range(3)]
        dropper = MultiOrientationFOVDrop(
            prob_per_orientation=1.0,
            ensure_coverage=True,
        )
        torch.manual_seed(0)
        dropped, masks = dropper(stacks)
        # At least one mask should be all-ones for each batch element
        any_full = False
        for m in masks:
            if (m == 1).all():
                any_full = True
        assert any_full

    def test_n_stacks(self, synthetic_volume):
        """Works with N != 3 stacks."""
        stacks = [synthetic_volume.clone() for _ in range(5)]
        dropped, masks = apply_fov_augmentation(stacks, prob=0.5)
        assert len(dropped) == 5
        assert len(masks) == 5


class TestApplyFOVAugmentation:
    def test_convenience_function(self, synthetic_volume):
        stacks = [synthetic_volume.clone() for _ in range(3)]
        dropped, masks = apply_fov_augmentation(stacks, prob=0.5)
        assert len(dropped) == 3
        assert len(masks) == 3
        for d, m in zip(dropped, masks):
            assert d.shape == synthetic_volume.shape

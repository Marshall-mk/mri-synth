"""Tests for HRLRDataGenerator pipeline."""

import torch

from mri_synth.fov.slice_drop import FOVSliceDrop
from mri_synth.pipeline import HRLRDataGenerator
from mri_synth.config import GenerationConfig


class TestHRLRDataGenerator:
    def test_produces_n_stacks(self, synthetic_volume):
        gen = HRLRDataGenerator(
            num_stacks=3,
            preserve_input_shape=True,
            return_intermediate=False,
            fov_augmentation_prob=0.0,
        )
        lr_stacks, hr_aug, orient_mask, spatial_masks, interp_masks = gen.generate_paired_data(
            synthetic_volume
        )
        assert len(lr_stacks) == 3
        assert len(spatial_masks) == 3
        assert len(interp_masks) == 3
        assert orient_mask.shape == (1, 3)

    def test_produces_6_stacks(self, synthetic_volume):
        gen = HRLRDataGenerator(
            num_stacks=6,
            preserve_input_shape=True,
            return_intermediate=False,
            fov_augmentation_prob=0.0,
        )
        lr_stacks, hr_aug, orient_mask, spatial_masks, interp_masks = gen.generate_paired_data(
            synthetic_volume
        )
        assert len(lr_stacks) == 6
        assert len(interp_masks) == 6
        assert orient_mask.shape == (1, 6)

    def test_variations_differ(self, synthetic_volume):
        """Different calls produce different stochastic degradations."""
        gen = HRLRDataGenerator(
            num_stacks=3,
            preserve_input_shape=True,
            return_intermediate=False,
            fov_augmentation_prob=0.0,
        )
        torch.manual_seed(0)
        r1 = gen.generate_paired_data(synthetic_volume)
        torch.manual_seed(1)
        r2 = gen.generate_paired_data(synthetic_volume)
        # LR stacks should differ
        assert not torch.allclose(r1[0][0], r2[0][0])

    def test_return_resolution(self, synthetic_volume):
        gen = HRLRDataGenerator(
            num_stacks=3,
            preserve_input_shape=True,
            return_intermediate=False,
            fov_augmentation_prob=0.0,
        )
        result = gen.generate_paired_data(synthetic_volume, return_resolution=True)
        lr_stacks, hr_aug, resolutions, thicknesses, orient_mask, spatial_masks, interp_masks = result
        assert len(resolutions) == 3
        assert resolutions[0].shape == (1, 3)

    def test_return_intermediate(self, synthetic_volume):
        gen = HRLRDataGenerator(
            num_stacks=3,
            preserve_input_shape=True,
            return_intermediate=True,
            fov_augmentation_prob=0.0,
        )
        result = gen.generate_paired_data(
            synthetic_volume, return_resolution=True
        )
        lr_stacks, true_lr_stacks, hr_aug, resolutions, thicknesses, orient_mask, spatial_masks, interp_masks = result
        assert len(true_lr_stacks) == 3

    def test_from_config(self, synthetic_volume):
        cfg = GenerationConfig(
            num_stacks=3,
            preserve_input_shape=True,
            return_intermediate=False,
        )
        cfg.fov.enable = False
        gen = HRLRDataGenerator.from_config(cfg)
        lr_stacks, hr_aug, orient_mask, spatial_masks, interp_masks = gen.generate_paired_data(
            synthetic_volume
        )
        assert len(lr_stacks) == 3

    def test_end_to_end_shapes(self, synthetic_volume):
        """Full pipeline produces valid shapes."""
        gen = HRLRDataGenerator(
            num_stacks=3,
            preserve_input_shape=True,
            return_intermediate=False,
            fov_augmentation_prob=0.0,
        )
        lr_stacks, hr_aug, orient_mask, spatial_masks, interp_masks = gen.generate_paired_data(
            synthetic_volume
        )
        for stack in lr_stacks:
            assert stack.shape == synthetic_volume.shape
        assert hr_aug.shape == synthetic_volume.shape

    def test_bias_and_intensity_consistent_across_stacks(self, synthetic_volume):
        """Bias field and intensity aug should be identical across stacks.

        With all physics artifacts disabled and fixed resolution, the only
        difference between stacks should be the through-plane axis. The
        bias/gamma degradation applied to the HR image must be the same
        for every stack.
        """
        gen = HRLRDataGenerator(
            num_stacks=3,
            prob_motion=0.0,
            prob_spike=0.0,
            prob_aliasing=0.0,
            prob_noise=0.0,
            prob_bias_field=1.0,
            randomise_res=False,
            apply_intensity_aug=True,
            preserve_input_shape=True,
            return_intermediate=False,
            fov_augmentation_prob=0.0,
            clip_to_unit_range=False,
        )

        torch.manual_seed(123)

        # We need to verify the bias+intensity step is shared.
        # Capture the degraded HR before the physics sim by monkey-patching
        # the artifact simulator to be a no-op identity.
        original_forward = gen.artifact_simulator.forward

        captured_inputs = []

        def capturing_forward(image, *args, **kwargs):
            captured_inputs.append(image.clone())
            return original_forward(image, *args, **kwargs)

        gen.artifact_simulator.forward = capturing_forward

        gen.generate_paired_data(synthetic_volume)

        assert len(captured_inputs) == 3
        # All stacks should receive the same degraded image
        for i in range(1, len(captured_inputs)):
            assert torch.allclose(captured_inputs[0], captured_inputs[i]), (
                f"Stack 0 and stack {i} received different bias/intensity inputs"
            )

    def test_interpolation_masks_shape_and_count(self, synthetic_volume):
        """Interpolation masks should match stack count and HR shape."""
        gen = HRLRDataGenerator(
            num_stacks=3,
            preserve_input_shape=True,
            return_intermediate=False,
            fov_augmentation_prob=0.0,
        )
        lr_stacks, hr_aug, orient_mask, spatial_masks, interp_masks = gen.generate_paired_data(
            synthetic_volume
        )
        assert len(interp_masks) == 3
        for mask in interp_masks:
            assert mask.shape == synthetic_volume.shape

    def test_interpolation_masks_binary(self, synthetic_volume):
        """Interpolation masks should contain only 0s and 1s."""
        gen = HRLRDataGenerator(
            num_stacks=3,
            preserve_input_shape=True,
            return_intermediate=False,
            fov_augmentation_prob=0.0,
        )
        _, _, _, _, interp_masks = gen.generate_paired_data(synthetic_volume)
        for mask in interp_masks:
            unique_vals = torch.unique(mask)
            assert all(v in [0.0, 1.0] for v in unique_vals.tolist())

    def test_interpolation_masks_has_both_values(self, synthetic_volume):
        """With anisotropic resolution, masks should have both 0s and 1s."""
        gen = HRLRDataGenerator(
            num_stacks=3,
            preserve_input_shape=True,
            return_intermediate=False,
            fov_augmentation_prob=0.0,
            randomise_res=False,
            max_res_aniso=[5.0, 5.0, 5.0],
        )
        _, _, _, _, interp_masks = gen.generate_paired_data(synthetic_volume)
        for mask in interp_masks:
            assert 0.0 in mask
            assert 1.0 in mask

    def test_interpolation_masks_with_return_resolution(self, synthetic_volume):
        """Interpolation masks should be returned with return_resolution=True."""
        gen = HRLRDataGenerator(
            num_stacks=3,
            preserve_input_shape=True,
            return_intermediate=False,
            fov_augmentation_prob=0.0,
        )
        result = gen.generate_paired_data(synthetic_volume, return_resolution=True)
        lr_stacks, hr_aug, resolutions, thicknesses, orient_mask, spatial_masks, interp_masks = result
        assert len(interp_masks) == 3
        for mask in interp_masks:
            assert mask.shape == synthetic_volume.shape

    def test_fov_force_both_sides_default(self):
        """FOVSliceDrop should default to force_both_sides=True."""
        dropper = FOVSliceDrop()
        assert dropper.force_both_sides is True

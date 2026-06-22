"""Tests for HRLRDataGenerator pipeline."""

import torch

from mri_synth.pipeline import HRLRDataGenerator
from mri_synth.config import GenerationConfig


class TestHRLRDataGenerator:
    def test_produces_n_stacks(self, synthetic_volume):
        gen = HRLRDataGenerator(
            num_stacks=3,
            return_intermediate=False,
            fov_augmentation_prob=0.0,
            enable_obliqueness=False,
        )
        lr_stacks, hr_aug, orient_mask, fov_masks = gen.generate_paired_data(
            synthetic_volume
        )
        assert len(lr_stacks) == 3
        assert len(fov_masks) == 3
        assert orient_mask.shape == (1, 3)

    def test_produces_6_stacks(self, synthetic_volume):
        gen = HRLRDataGenerator(
            num_stacks=6,
            return_intermediate=False,
            fov_augmentation_prob=0.0,
            enable_obliqueness=False,
        )
        lr_stacks, hr_aug, orient_mask, fov_masks = gen.generate_paired_data(
            synthetic_volume
        )
        assert len(lr_stacks) == 6
        assert len(fov_masks) == 6
        assert orient_mask.shape == (1, 6)

    def test_variations_differ(self, synthetic_volume):
        """Different calls produce different stochastic degradations."""
        gen = HRLRDataGenerator(
            num_stacks=3,
            return_intermediate=False,
            fov_augmentation_prob=0.0,
            enable_obliqueness=False,
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
            return_intermediate=False,
            fov_augmentation_prob=0.0,
            enable_obliqueness=False,
        )
        result = gen.generate_paired_data(synthetic_volume, return_resolution=True)
        lr_stacks, hr_aug, resolutions, thicknesses, orient_mask, fov_masks = result
        assert len(resolutions) == 3
        assert resolutions[0].shape == (1, 3)

    def test_return_intermediate(self, synthetic_volume):
        gen = HRLRDataGenerator(
            num_stacks=3,
            return_intermediate=True,
            fov_augmentation_prob=0.0,
            enable_obliqueness=False,
        )
        result = gen.generate_paired_data(
            synthetic_volume, return_resolution=True
        )
        lr_stacks, true_lr_stacks, hr_aug, resolutions, thicknesses, orient_mask, fov_masks = result
        assert len(true_lr_stacks) == 3

    def test_from_config(self, synthetic_volume):
        cfg = GenerationConfig(
            num_stacks=3,
            return_intermediate=False,
        )
        cfg.fov.enable = False
        cfg.fov.enable_obliqueness = False
        gen = HRLRDataGenerator.from_config(cfg)
        lr_stacks, hr_aug, orient_mask, fov_masks = gen.generate_paired_data(
            synthetic_volume
        )
        assert len(lr_stacks) == 3

    def test_end_to_end_shapes(self, synthetic_volume):
        """Full pipeline produces valid shapes."""
        gen = HRLRDataGenerator(
            num_stacks=3,
            return_intermediate=False,
            fov_augmentation_prob=0.0,
            enable_obliqueness=False,
        )
        lr_stacks, hr_aug, orient_mask, fov_masks = gen.generate_paired_data(
            synthetic_volume
        )
        for stack in lr_stacks:
            assert stack.shape == synthetic_volume.shape
        assert hr_aug.shape == synthetic_volume.shape

    def test_bias_and_intensity_consistent_across_stacks(self, synthetic_volume):
        """Bias field and intensity aug should be identical across stacks."""
        gen = HRLRDataGenerator(
            num_stacks=3,
            prob_motion=0.0,
            prob_spike=0.0,
            prob_aliasing=0.0,
            prob_noise=0.0,
            prob_bias_field=1.0,
            randomise_res=False,
            apply_intensity_aug=True,
            return_intermediate=False,
            fov_augmentation_prob=0.0,
            clip_to_unit_range=False,
            enable_obliqueness=False,
        )

        torch.manual_seed(123)

        original_forward = gen.artifact_simulator.forward

        captured_inputs = []

        def capturing_forward(image, *args, **kwargs):
            captured_inputs.append(image.clone())
            return original_forward(image, *args, **kwargs)

        gen.artifact_simulator.forward = capturing_forward

        gen.generate_paired_data(synthetic_volume)

        assert len(captured_inputs) == 3
        for i in range(1, len(captured_inputs)):
            assert torch.allclose(captured_inputs[0], captured_inputs[i]), (
                f"Stack 0 and stack {i} received different bias/intensity inputs"
            )

    def test_fov_masks_shape_and_count(self, synthetic_volume):
        """FOV masks should match stack count and HR spatial shape."""
        gen = HRLRDataGenerator(
            num_stacks=3,
            return_intermediate=False,
            fov_augmentation_prob=0.0,
            enable_obliqueness=False,
        )
        lr_stacks, hr_aug, orient_mask, fov_masks = gen.generate_paired_data(
            synthetic_volume
        )
        assert len(fov_masks) == 3
        for mask in fov_masks:
            # fov_mask shape: (B, 1, D, H, W) — channel dim is 1 for the mask
            assert mask.shape[0] == synthetic_volume.shape[0]  # batch
            assert mask.shape[2:] == synthetic_volume.shape[2:]  # spatial dims

    def test_fov_masks_binary(self, synthetic_volume):
        """FOV masks should contain only 0s and 1s."""
        gen = HRLRDataGenerator(
            num_stacks=3,
            return_intermediate=False,
            fov_augmentation_prob=0.0,
            enable_obliqueness=False,
        )
        _, _, _, fov_masks = gen.generate_paired_data(synthetic_volume)
        for mask in fov_masks:
            unique_vals = torch.unique(mask)
            assert all(v in [0.0, 1.0] for v in unique_vals.tolist())

    def test_fov_masks_all_zeros_without_obliqueness(self, synthetic_volume):
        """Without obliqueness and FOV drop, masks should be all zeros (nothing missing)."""
        gen = HRLRDataGenerator(
            num_stacks=3,
            return_intermediate=False,
            fov_augmentation_prob=0.0,
            enable_obliqueness=False,
        )
        _, _, _, fov_masks = gen.generate_paired_data(synthetic_volume)
        for mask in fov_masks:
            assert (mask == 0).all()

    def test_fov_masks_with_obliqueness_has_ones(self, synthetic_volume):
        """With large obliqueness, masks should have ones (missing regions)."""
        gen = HRLRDataGenerator(
            num_stacks=3,
            return_intermediate=False,
            fov_augmentation_prob=0.0,
            enable_obliqueness=True,
            prob_obliqueness=1.0,
            obliqueness_range=30.0,
            randomise_res=False,
            max_res_aniso=[5.0, 5.0, 5.0],
        )
        torch.manual_seed(42)
        _, _, _, fov_masks = gen.generate_paired_data(synthetic_volume)
        any_has_ones = any(1.0 in mask for mask in fov_masks)
        assert any_has_ones, "Large obliqueness should produce ones (missing voxels) in FOV masks"

    def test_from_config_wires_artifact_intensities(self):
        """noise_std / motion_intensity / spike_intensity / bias_field_std
        from config must reach the simulator (regression: were hardcoded)."""
        cfg = GenerationConfig()
        cfg.artifacts.noise_std = 0.123
        cfg.artifacts.motion_intensity = 0.456
        cfg.artifacts.spike_intensity = 0.789
        cfg.physics.bias_field_std = 0.321
        gen = HRLRDataGenerator.from_config(cfg)
        assert gen.artifact_simulator.noise_std == 0.123
        assert gen.artifact_simulator.motion_intensity == 0.456
        assert gen.artifact_simulator.spike_intensity == 0.789
        assert gen.bias.bias_field_std == 0.321

    def test_structural_only_zeroes_appearance_probs(self):
        """structural_only must silence every appearance corruption."""
        gen = HRLRDataGenerator(
            structural_only=True,
            prob_motion=1.0,
            prob_spike=1.0,
            prob_aliasing=1.0,
            prob_noise=1.0,
            prob_bias_field=1.0,
            apply_intensity_aug=True,
        )
        assert gen.prob_bias_field == 0.0
        assert gen.apply_intensity_aug is False
        assert gen.artifact_simulator.prob_motion == 0.0
        assert gen.artifact_simulator.prob_spike == 0.0
        assert gen.artifact_simulator.prob_aliasing == 0.0
        assert gen.artifact_simulator.prob_noise == 0.0

    def test_structural_only_no_artifacts_applied(self, synthetic_volume):
        """With structural_only, no appearance corruption is ever sampled,
        even when every probability is forced to 1.0."""
        gen = HRLRDataGenerator(
            num_stacks=3,
            structural_only=True,
            prob_motion=1.0,
            prob_spike=1.0,
            prob_aliasing=1.0,
            prob_noise=1.0,
            prob_bias_field=1.0,
            apply_intensity_aug=True,
            fov_augmentation_prob=0.0,
            enable_obliqueness=False,
        )
        gen.generate_paired_data(synthetic_volume)
        d = gen._last_decisions
        assert d["intensity_aug"] is False
        for key in ("bias_field", "motion", "spike", "aliasing", "noise"):
            assert not bool(d[key].any()), f"{key} should never be applied"

    def test_structural_only_from_config(self, synthetic_volume):
        """structural_only must propagate through from_config."""
        cfg = GenerationConfig(structural_only=True)
        cfg.artifacts.prob_noise = 1.0
        cfg.physics.prob_bias_field = 1.0
        gen = HRLRDataGenerator.from_config(cfg)
        assert gen.prob_bias_field == 0.0
        assert gen.artifact_simulator.prob_noise == 0.0

    def test_fov_masks_with_return_resolution(self, synthetic_volume):
        """FOV masks should be returned with return_resolution=True."""
        gen = HRLRDataGenerator(
            num_stacks=3,
            return_intermediate=False,
            fov_augmentation_prob=0.0,
            enable_obliqueness=False,
        )
        result = gen.generate_paired_data(synthetic_volume, return_resolution=True)
        lr_stacks, hr_aug, resolutions, thicknesses, orient_mask, fov_masks = result
        assert len(fov_masks) == 3
        for mask in fov_masks:
            assert mask.shape[0] == synthetic_volume.shape[0]
            assert mask.shape[2:] == synthetic_volume.shape[2:]

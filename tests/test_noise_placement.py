"""Noise must be acquired at native LR resolution, not painted on afterwards.

Adding Rician noise after the resample to the HR grid is wrong in three
separable ways, each covered here: the noise ends up white at HR resolution
instead of correlated at the LR voxel scale, it fills regions the FOV mask
reports as never acquired, and it never reaches the native-resolution stacks
that ``return_intermediate`` hands back.
"""

import pytest
import torch

from mri_synth.pipeline import HRLRDataGenerator

NOISE_STD = 0.08
# Mean of a Rician variable over zero signal is sigma * sqrt(pi / 2).
RICIAN_AIR_MEAN = NOISE_STD * (torch.pi / 2) ** 0.5


@pytest.fixture
def phantom():
    n = 48
    g = torch.arange(n, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(g, g, g, indexing="ij")
    r = ((zz - 23) / 14) ** 2 + ((yy - 23) / 12) ** 2 + ((xx - 23) / 13) ** 2
    return torch.exp(-3.0 * r).unsqueeze(0).unsqueeze(0)


def _generator(**kw):
    params = dict(
        num_stacks=3,
        randomise_res=False,
        min_resolution=[1.0, 1.0, 1.0],
        max_res_aniso=[6.0, 6.0, 6.0],
        prob_noise=1.0,
        noise_std=NOISE_STD,
        prob_bias_field=0.0,
        prob_motion=0.0,
        prob_spike=0.0,
        prob_aliasing=0.0,
        fov_augmentation_prob=0.0,
        enable_obliqueness=False,
        tight_fov=False,
        clip_to_unit_range=False,
    )
    params.update(kw)
    return HRLRDataGenerator(**params)


class TestNoiseIsAcquiredAtNativeResolution:
    def test_native_stacks_carry_noise(self, phantom):
        """return_intermediate stacks must not be silently noise-free."""
        gen = _generator(return_intermediate=True)
        torch.manual_seed(0)
        _, true_lr, _, _, _, _, _ = gen.generate_paired_data(
            phantom, return_resolution=True
        )
        for stack in true_lr:
            air = stack[0, 0][stack[0, 0] < 0.5 * RICIAN_AIR_MEAN * 4]
            assert stack[0, 0].min() >= 0.0, "Rician magnitude cannot be negative"
            assert air.numel() > 0
            # A noiseless stack would have an air floor at ~0.
            assert air.mean().item() > 0.5 * RICIAN_AIR_MEAN

    def test_noise_is_correlated_at_the_lr_voxel_scale(self, phantom):
        """Upsampled noise must not be white at HR resolution."""
        gen = _generator()
        torch.manual_seed(0)
        lr_stacks, _, _, _ = gen.generate_paired_data(phantom)
        through_plane = [2, 1, 0]
        for s, stack in enumerate(lr_stacks):
            vol = stack[0, 0]
            resid = vol - torch.nn.functional.avg_pool3d(
                vol[None, None], 3, 1, 1
            )[0, 0]
            axis = through_plane[s]
            a = resid.narrow(axis, 0, vol.shape[axis] - 1).flatten()
            b = resid.narrow(axis, 1, vol.shape[axis] - 1).flatten()
            corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
            assert corr > 0.5, (
                f"stack {s}: lag-1 noise autocorrelation {corr:.3f} — noise "
                f"looks white at HR resolution, i.e. added after resampling"
            )

    def test_no_noise_where_the_fov_mask_says_missing(self, phantom):
        """Out-of-FOV regions must not be filled with plausible-looking signal."""
        gen = _generator(
            fov_augmentation_prob=1.0,
            fov_min_keep=0.4,
            fov_max_keep=0.5,
            fov_ensure_coverage=False,
        )
        torch.manual_seed(0)
        lr_stacks, _, _, masks = gen.generate_paired_data(phantom)
        for stack, mask in zip(lr_stacks, masks):
            missing = mask[0, 0] == 1
            assert missing.any()
            vals = stack[0, 0][missing]
            # Most missing voxels are exactly zero; the rest are the bilinear
            # ramp at the slab boundary (the image interpolates, the mask
            # rounds), never a Rician noise floor.
            assert (vals == 0).float().mean().item() > 0.7
            assert vals.mean().item() < 0.25 * RICIAN_AIR_MEAN

    def test_dropped_slices_are_exactly_zero(self, phantom):
        """Slices that were never acquired must not pick up a noise floor."""
        gen = _generator(
            return_intermediate=True,
            fov_augmentation_prob=1.0,
            fov_min_keep=0.4,
            fov_max_keep=0.5,
            fov_ensure_coverage=False,
        )
        torch.manual_seed(0)
        _, true_lr, _, _, _, _, _ = gen.generate_paired_data(
            phantom, return_resolution=True
        )
        through_plane = [2, 1, 0]
        for s, stack in enumerate(true_lr):
            vol = stack[0, 0]
            axis = through_plane[s]
            per_slice = vol.abs().sum(dim=[d for d in range(3) if d != axis])
            assert (per_slice == 0).any(), "expected fully dropped slices"

    def test_noise_disabled_leaves_no_floor(self, phantom):
        gen = _generator(prob_noise=0.0, return_intermediate=True)
        torch.manual_seed(0)
        _, true_lr, _, _, _, _, _ = gen.generate_paired_data(
            phantom, return_resolution=True
        )
        for stack in true_lr:
            air = stack[0, 0][stack[0, 0].abs() < 0.05]
            assert air.abs().mean().item() < 0.1 * RICIAN_AIR_MEAN

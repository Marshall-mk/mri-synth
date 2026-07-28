"""Geometric-fidelity tests: LR stacks must stay registered to the HR grid.

These cover the class of bug that a shape/dtype test cannot see — the LR data
landing in the wrong *place* relative to the ground truth. Sub-voxel shifts are
invisible in a shape assertion but are exactly what breaks reconstruction
methods (INRs, super-resolution) that assume LR and HR share a coordinate
system.
"""



import pytest
import torch

from mri_synth.artifacts.simulator import MRIArtifactSimulator
from mri_synth.fov.resampling import affine_resample_3d, build_lr_affine
from mri_synth.physics.slice_profile import SliceProfilePhysics
from mri_synth.pipeline import HRLRDataGenerator


def _centroid(vol: torch.Tensor, axis: int) -> float:
    """Intensity centroid of a (C, D, H, W) volume along a spatial axis."""
    prof = vol[0].sum(dim=[d for d in range(3) if d != axis]).double()
    idx = torch.arange(prof.numel(), dtype=torch.float64)
    return ((prof * idx).sum() / prof.sum()).item()


def _blob(n: int, centre: float, axis: int = 0, sigma: float = 7.0) -> torch.Tensor:
    """Smooth, near-band-limited ridge centred at ``centre`` along ``axis``."""
    g = torch.arange(n, dtype=torch.float32)
    prof = torch.exp(-0.5 * ((g - centre) / sigma) ** 2)
    shape = [1, 1, 1, 1]
    shape[axis + 1] = n
    return prof.view(shape).expand(1, n, n, n).clone()


class TestSliceProfileKernel:
    """The slice-select PSF must blur without translating."""

    @pytest.mark.parametrize("profile", ["boxcar", "gaussian", "trapezoid"])
    @pytest.mark.parametrize("thickness", [3.0, 4.0, 5.0, 9.0])
    def test_kernel_is_centred(self, profile, thickness):
        phys = SliceProfilePhysics(profile_type=profile, edge_width=0.1)
        kernel = phys.get_slice_kernel(
            torch.tensor(thickness), torch.tensor(1.0), torch.device("cpu")
        )
        idx = torch.arange(kernel.numel(), dtype=torch.float64)
        com = (kernel.double() * idx).sum() / kernel.double().sum()
        # conv1d pads by kernel_size // 2, so the kernel's centre of mass must
        # sit on that tap or the whole volume is translated.
        assert abs(com.item() - kernel.numel() // 2) < 1e-4

    @pytest.mark.parametrize("profile", ["boxcar", "gaussian", "trapezoid"])
    def test_kernel_is_symmetric(self, profile):
        phys = SliceProfilePhysics(profile_type=profile, edge_width=0.1)
        kernel = phys.get_slice_kernel(
            torch.tensor(5.0), torch.tensor(1.0), torch.device("cpu")
        )
        assert torch.allclose(kernel, kernel.flip(0), atol=1e-6)

    @pytest.mark.parametrize("thickness", [3.0, 5.0, 9.0])
    def test_blur_does_not_shift_volume(self, thickness):
        phys = SliceProfilePhysics(profile_type="trapezoid", edge_width=0.1)
        n = 64
        img = torch.zeros(1, n, 8, 8)
        img[0, n // 2, 4, 4] = 1.0
        out = phys(
            img,
            resolution=torch.tensor([1.0, 1.0, 1.0]),
            thickness=torch.tensor([thickness, 1.0, 1.0]),
        )
        assert abs(_centroid(out, 0) - n // 2) < 1e-3

    def test_effective_thickness_matches_request(self):
        """A boxcar profile of thickness T must span ~T taps, not T*(K-1)/K."""
        phys = SliceProfilePhysics(profile_type="boxcar")
        kernel = phys.get_slice_kernel(
            torch.tensor(9.0), torch.tensor(1.0), torch.device("cpu")
        )
        # Boxcar over |x| <= 0.5 with x = grid / 9 -> taps at |grid| <= 4.5 -> 9 taps
        assert int((kernel > 0).sum().item()) == 9


class TestLrHrRegistration:
    """The LR affine must describe where the downsampled samples actually are."""

    @pytest.mark.parametrize("req_res", [2.0, 3.0, 4.0, 4.7, 6.0])
    @pytest.mark.parametrize("centre", [40.0, 63.5, 90.0])
    def test_downsample_then_resample_preserves_position(self, req_res, centre):
        n = 128
        hr = _blob(n, centre)
        lr = MRIArtifactSimulator._fft_downsample(hr, 0, req_res)
        m = lr.shape[1]

        hr_affine = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        lr_affine = build_lr_affine(
            hr_affine=hr_affine,
            through_plane_axis=0,
            lr_spacing_tp=n / m,  # effective spacing, not the requested one
            hr_spacing_tp=1.0,
            lr_shape=tuple(lr.shape[1:]),
            hr_shape=(n, n, n),
        )
        back = affine_resample_3d(lr, lr_affine, hr_affine, (n, n, n), mode="bilinear")
        assert abs(_centroid(back, 0) - centre) < 0.05

    def test_build_lr_affine_warns_on_nominal_spacing(self):
        """Passing the requested resolution instead of the realised one is a bug."""
        hr_affine = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        with pytest.warns(UserWarning, match="effective spacing"):
            build_lr_affine(
                hr_affine=hr_affine,
                through_plane_axis=2,
                lr_spacing_tp=4.7,  # nominal; 128/27 = 4.741 is what the crop gives
                hr_spacing_tp=1.0,
                lr_shape=(128, 128, 27),
                hr_shape=(128, 128, 128),
            )

    def test_fft_downsample_grid_is_centred(self):
        """LR samples must straddle the HR FOV symmetrically."""
        n = 128
        # A symmetric object must stay symmetric after downsampling.
        hr = _blob(n, (n - 1) / 2.0)
        lr = MRIArtifactSimulator._fft_downsample(hr, 0, 6.0)
        prof = lr[0].sum(dim=[1, 2])
        prof = prof / prof.max()
        assert torch.allclose(prof, prof.flip(0), atol=1e-5)


class TestResolutionSampling:
    """The configured resolution range must be honoured in both modes."""

    @pytest.mark.parametrize("randomise", [True, False])
    def test_configured_range_is_respected(self, randomise):
        gen = HRLRDataGenerator(
            num_stacks=3,
            randomise_res=randomise,
            min_resolution=[1.0, 1.0, 1.0],
            max_res_aniso=[3.0, 3.0, 3.0],
        )
        resolutions, _ = gen._create_orthogonal_resolutions(1, torch.device("cpu"))
        for res in resolutions:
            assert res.max().item() <= 3.0 + 1e-6, (
                "through-plane resolution exceeded the configured max_res_aniso"
            )
            assert res.min().item() >= 1.0 - 1e-6

    def test_deterministic_mode_uses_configured_max(self):
        """randomise_res=False must give exactly max_res_aniso, not a default."""
        gen = HRLRDataGenerator(
            num_stacks=3,
            randomise_res=False,
            min_resolution=[1.0, 1.0, 1.0],
            max_res_aniso=[4.0, 4.0, 4.0],
        )
        resolutions, _ = gen._create_orthogonal_resolutions(1, torch.device("cpu"))
        through_plane = [2, 1, 0]
        for stack_idx, res in enumerate(resolutions):
            assert res[0, through_plane[stack_idx]].item() == pytest.approx(4.0)


class TestFovCoverage:
    """FOV simulation stays active under structural_only, and the mask tells
    the truth about which voxels the stack actually observed."""

    @pytest.fixture
    def phantom(self):
        n = 64
        g = torch.arange(n, dtype=torch.float32)
        zz, yy, xx = torch.meshgrid(g, g, g, indexing="ij")
        r = ((zz - 31) / 19) ** 2 + ((yy - 31) / 16) ** 2 + ((xx - 31) / 17) ** 2
        return torch.exp(-3.0 * r).unsqueeze(0).unsqueeze(0)

    def _generator(self, **kw):
        params = dict(
            num_stacks=3,
            structural_only=True,
            randomise_res=False,
            min_resolution=[1.0, 1.0, 1.0],
            max_res_aniso=[4.0, 4.0, 4.0],
            fov_augmentation_prob=1.0,
            fov_min_keep=0.4,
            fov_max_keep=0.6,
            fov_ensure_coverage=False,
            enable_obliqueness=False,
            tight_fov=False,
        )
        params.update(kw)
        return HRLRDataGenerator(**params)

    @pytest.mark.parametrize("tight_fov", [False, True])
    def test_slice_drop_reaches_the_mask(self, phantom, tight_fov):
        """A dropped slab must be marked missing, not left looking like signal.

        Regression: with tight_fov=False the all-ones support used to be built
        after the drop, so the mask declared the zeroed slabs valid.
        """
        gen = self._generator(tight_fov=tight_fov)
        torch.manual_seed(3)
        lr_stacks, _, _, masks = gen.generate_paired_data(phantom)
        for stack, mask in zip(lr_stacks, masks):
            assert mask.mean() > 0.05, "FOV drop left no missing region in the mask"
            # Wherever the mask says data is valid, the stack must not be a
            # blanket zero slab.
            valid = mask[0, 0] == 0
            assert stack[0, 0][valid].abs().sum() > 0

    def test_structural_only_keeps_fov_simulation(self, phantom):
        """structural_only silences appearance only — FOV must survive."""
        gen = self._generator(enable_obliqueness=True, prob_obliqueness=1.0,
                              obliqueness_range=12.0, tight_fov=True)
        torch.manual_seed(3)
        _, _, _, masks = gen.generate_paired_data(phantom)
        assert all(m.mean() > 0.05 for m in masks)

    def test_each_orientation_covers_a_different_region(self, phantom):
        """Axial, coronal and sagittal stacks must not share one FOV."""
        gen = self._generator()
        torch.manual_seed(3)
        _, _, _, masks = gen.generate_paired_data(phantom)
        valid = [m[0, 0] == 0 for m in masks]
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            assert not torch.equal(valid[i], valid[j])

        # Each stack's slab must be limited along its own through-plane axis
        # and full along the other two.
        through_plane = [2, 1, 0]
        for s, v in enumerate(valid):
            for axis in range(3):
                per = v.float().mean(dim=[d for d in range(3) if d != axis])
                covered = (per > 0.01).sum().item()
                if axis == through_plane[s]:
                    assert covered < v.shape[axis], (
                        f"stack {s} should be cropped along axis {axis}"
                    )
                else:
                    assert covered == v.shape[axis], (
                        f"stack {s} should be full along in-plane axis {axis}"
                    )

    def test_force_both_sides_false_offsets_the_slab(self, phantom):
        """With one-sided dropping the slabs sit off-centre, not just narrower."""
        gen = self._generator(fov_force_both_sides=False)
        torch.manual_seed(3)
        _, _, _, masks = gen.generate_paired_data(phantom)
        through_plane = [2, 1, 0]
        offsets = []
        for s, m in enumerate(masks):
            v = m[0, 0] == 0
            axis = through_plane[s]
            per = v.float().mean(dim=[d for d in range(3) if d != axis])
            idx = (per > 0.01).nonzero().flatten().float()
            offsets.append(idx.mean().item() - (v.shape[axis] - 1) / 2.0)
        assert any(abs(o) > 2.0 for o in offsets), (
            "one-sided FOV drop should place at least one slab off-centre"
        )

    def test_no_fov_simulation_means_full_coverage(self, phantom):
        gen = self._generator(fov_augmentation_prob=0.0)
        _, _, _, masks = gen.generate_paired_data(phantom)
        assert all((m == 0).all() for m in masks)


class TestEnsureUnionCoverage:
    """`ensure_coverage` must guarantee the union of stacks contains the head.

    Checked against the FOV masks the pipeline actually produces, not against
    the analytic predictor that drives the repair.
    """

    @pytest.fixture
    def big_head(self):
        """A head that reaches close to the FOV edges — the demanding case."""
        n = 72
        g = torch.arange(n, dtype=torch.float32)
        zz, yy, xx = torch.meshgrid(g, g, g, indexing="ij")
        r = ((zz - 35) / 23) ** 2 + ((yy - 35) / 20) ** 2 + ((xx - 35) / 21) ** 2
        return torch.exp(-3.0 * r).unsqueeze(0).unsqueeze(0)

    def _generator(self, **kw):
        params = dict(
            num_stacks=3,
            structural_only=True,
            randomise_res=True,
            min_resolution=[1.0, 1.0, 1.0],
            max_res_aniso=[6.0, 6.0, 6.0],
            fov_augmentation_prob=1.0,
            fov_min_keep=0.35,
            fov_max_keep=0.65,
            fov_force_both_sides=False,
            fov_ensure_coverage=True,
            enable_obliqueness=False,
            tight_fov=False,
        )
        params.update(kw)
        return HRLRDataGenerator(**params)

    @staticmethod
    def _unseen_fraction(phantom, masks):
        head = phantom[0, 0] > 1e-3
        union = torch.zeros_like(head)
        for m in masks:
            union |= m[0, 0] == 0
        return ((head & ~union).sum() / head.sum()).item()

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
    def test_union_contains_the_head(self, big_head, seed):
        gen = self._generator()
        torch.manual_seed(seed)
        _, _, _, masks = gen.generate_paired_data(big_head)
        assert self._unseen_fraction(big_head, masks) == 0.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_union_contains_the_head_with_obliqueness(self, big_head, seed):
        gen = self._generator(enable_obliqueness=True, prob_obliqueness=1.0,
                              obliqueness_range=10.0, tight_fov=True)
        torch.manual_seed(seed)
        _, _, _, masks = gen.generate_paired_data(big_head)
        assert self._unseen_fraction(big_head, masks) == 0.0

    def test_disabled_can_leave_head_unseen(self, big_head):
        """Guards against the test passing for reasons other than the repair."""
        worst = 0.0
        for seed in range(12):
            gen = self._generator(fov_ensure_coverage=False)
            torch.manual_seed(seed)
            _, _, _, masks = gen.generate_paired_data(big_head)
            worst = max(worst, self._unseen_fraction(big_head, masks))
        assert worst > 0.0, (
            "expected some seed to lose head with ensure_coverage disabled; "
            "if not, the coverage test above proves nothing"
        )

    def test_does_not_force_a_stack_to_be_uncropped(self, big_head):
        """The old heuristic gave one stack full coverage; that over-corrects.

        Centred slabs already cover an ellipsoid between them, so the repair
        should leave the sampled cropping alone.
        """
        gen = self._generator(fov_force_both_sides=True)
        torch.manual_seed(0)
        _, _, _, masks = gen.generate_paired_data(big_head)
        assert self._unseen_fraction(big_head, masks) == 0.0
        keeps = [
            float(k[0]) for k in gen._last_structural_plan["fov_keep_fraction"]
        ]
        assert all(k < 1.0 for k in keeps), (
            f"no stack should have been forced to full coverage, got {keeps}"
        )

    def test_repair_only_grows_slabs(self, big_head):
        """Coverage must never be bought by shrinking another stack."""
        gen = self._generator()
        torch.manual_seed(1)
        plan = gen.sample_structural_plan(batch_size=1)
        before = [float(k[0]) for k in plan["fov_keep_fraction"]]
        gen.generate_paired_data(big_head, structural_plan=plan)
        after = [float(k[0]) for k in plan["fov_keep_fraction"]]
        assert all(a >= b - 1e-6 for a, b in zip(after, before))


class TestStructuralOnlyPipeline:
    """Geometry-only mode must yield LR stacks aligned with the HR ground truth."""

    @pytest.fixture
    def phantom(self):
        n = 96
        g = torch.arange(n, dtype=torch.float32)
        zz, yy, xx = torch.meshgrid(g, g, g, indexing="ij")
        ell = ((zz - 44) / 26) ** 2 + ((yy - 50) / 21) ** 2 + ((xx - 47) / 23) ** 2
        return torch.exp(-3.0 * ell).unsqueeze(0).unsqueeze(0)

    def _generator(self, **kw):
        params = dict(
            num_stacks=3,
            structural_only=True,
            randomise_res=False,
            min_resolution=[1.0, 1.0, 1.0],
            max_res_aniso=[7.0, 7.0, 7.0],
            fov_augmentation_prob=0.0,
            enable_obliqueness=False,
            tight_fov=False,
        )
        params.update(kw)
        return HRLRDataGenerator(**params)

    def test_stacks_are_registered_to_hr(self, phantom):
        gen = self._generator()
        lr_stacks, hr, _, _ = gen.generate_paired_data(phantom)
        hr_c = [_centroid(hr[0], a) for a in range(3)]
        for stack in lr_stacks:
            for axis in range(3):
                assert abs(_centroid(stack[0], axis) - hr_c[axis]) < 0.05

    def test_stacks_are_registered_to_each_other(self, phantom):
        """Orthogonal stacks must agree, or an INR gets contradictory evidence."""
        gen = self._generator()
        lr_stacks, _, _, _ = gen.generate_paired_data(phantom)
        for axis in range(3):
            cs = [_centroid(s[0], axis) for s in lr_stacks]
            assert max(cs) - min(cs) < 0.05

    def test_structural_only_is_psf_blur_and_nothing_else(self, phantom):
        """With no downsampling, the only difference from HR may be the PSF.

        Any bias field, gamma or noise would break the exact match against a
        reference that applies the PSF alone.
        """
        gen = self._generator(max_res_aniso=[1.0, 1.0, 1.0])
        lr_stacks, hr, _, _ = gen.generate_paired_data(phantom)
        reference = SliceProfilePhysics(profile_type="trapezoid", edge_width=0.1)(
            hr[0],
            resolution=torch.tensor([1.0, 1.0, 1.0]),
            thickness=torch.tensor([1.0, 1.0, 1.0]),
        )
        for stack in lr_stacks:
            assert torch.allclose(stack[0], reference, atol=1e-4)

    def test_structural_only_preserves_global_intensity(self, phantom):
        """A bias field or gamma shift would move the mean; a blur must not."""
        gen = self._generator()
        lr_stacks, hr, _, _ = gen.generate_paired_data(phantom)
        for stack in lr_stacks:
            assert stack.mean().item() == pytest.approx(hr.mean().item(), rel=2e-3)

    def test_obliqueness_keeps_object_centred(self, phantom):
        """Tilting a stack must rotate about the FOV centre, not translate it."""
        gen = self._generator(
            enable_obliqueness=True, prob_obliqueness=1.0, obliqueness_range=12.0
        )
        torch.manual_seed(7)
        lr_stacks, hr, _, _ = gen.generate_paired_data(phantom)
        hr_c = [_centroid(hr[0], a) for a in range(3)]
        for stack in lr_stacks:
            for axis in range(3):
                # A rotation about the centre leaves a centred ellipsoid's
                # centroid put; a mis-anchored rotation shifts it by voxels.
                assert abs(_centroid(stack[0], axis) - hr_c[axis]) < 1.0

"""Tests for affine-based resampling and FOV mask generation."""

import math

import torch

from mri_synth.fov.resampling import (
    affine_resample_3d,
    apply_fov_slice_drop_native,
    build_lr_affine,
    compute_brain_bbox_support_mask,
    compute_oblique_fov_mask,
    euler_to_rotation_matrix,
    resample_with_fov_mask,
)


class TestEulerToRotationMatrix:
    def test_identity_rotation(self):
        """Zero angles should produce identity matrix."""
        R = euler_to_rotation_matrix(0.0, 0.0, 0.0)
        assert torch.allclose(R, torch.eye(3), atol=1e-6)

    def test_rotation_is_orthogonal(self):
        """R @ R^T should equal identity."""
        R = euler_to_rotation_matrix(0.3, 0.5, 0.1)
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)

    def test_determinant_is_one(self):
        """Proper rotation should have det(R) = 1."""
        R = euler_to_rotation_matrix(0.7, -0.2, 0.4)
        assert abs(torch.linalg.det(R).item() - 1.0) < 1e-5


class TestBuildLrAffine:
    def test_no_rotation_scales_through_plane(self):
        """Without rotation, only the through-plane column should be scaled."""
        hr_affine = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        lr_affine = build_lr_affine(
            hr_affine,
            through_plane_axis=2,
            lr_spacing_tp=5.0,
            hr_spacing_tp=1.0,
            lr_shape=(32, 32, 6),
            hr_shape=(32, 32, 32),
        )
        # Through-plane column (axis 2) should be scaled by 5
        assert abs(lr_affine[2, 2].item() - 5.0) < 1e-5
        # In-plane columns should be unchanged
        assert abs(lr_affine[0, 0].item() - 1.0) < 1e-5
        assert abs(lr_affine[1, 1].item() - 1.0) < 1e-5


class TestAffineResample3d:
    def test_identity_resampling(self):
        """Identity affines and same shape should return input."""
        vol = torch.rand(1, 16, 16, 16)
        affine = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        result = affine_resample_3d(vol, affine, affine, (16, 16, 16), mode="bilinear")
        assert torch.allclose(vol, result, atol=1e-4)

    def test_output_shape(self):
        """Output should match target_shape."""
        vol = torch.rand(1, 10, 10, 10)
        affine = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        result = affine_resample_3d(vol, affine, affine, (20, 20, 20), mode="bilinear")
        assert result.shape == (1, 20, 20, 20)

    def test_nearest_mode_binary(self):
        """Nearest-neighbor on binary input should produce binary output."""
        vol = torch.ones(1, 8, 8, 8)
        src_affine = torch.diag(torch.tensor([2.0, 2.0, 2.0, 1.0]))
        tgt_affine = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        result = affine_resample_3d(vol, src_affine, tgt_affine, (16, 16, 16), mode="nearest")
        unique_vals = torch.unique(result)
        assert all(v in [0.0, 1.0] for v in unique_vals.tolist())


class TestResampleWithFovMask:
    def test_fov_mask_binary(self):
        """FOV mask should be binary (0 or 1)."""
        lr_vol = torch.rand(1, 32, 32, 6)
        hr_affine = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        lr_affine = build_lr_affine(
            hr_affine,
            through_plane_axis=2,
            lr_spacing_tp=5.0,
            hr_spacing_tp=1.0,
            lr_shape=(32, 32, 6),
            hr_shape=(32, 32, 32),
        )
        _, fov_mask = resample_with_fov_mask(lr_vol, lr_affine, hr_affine, (32, 32, 32))
        unique_vals = torch.unique(fov_mask)
        assert all(v in [0.0, 1.0] for v in unique_vals.tolist())

    def test_fov_mask_mostly_zero_aligned(self):
        """Axis-aligned LR->HR should have mostly zeros (valid) with some ones at edges."""
        lr_vol = torch.rand(1, 32, 32, 6)
        hr_affine = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        lr_affine = build_lr_affine(
            hr_affine,
            through_plane_axis=2,
            lr_spacing_tp=5.0,
            hr_spacing_tp=1.0,
            lr_shape=(32, 32, 6),
            hr_shape=(32, 32, 32),
            rotation_angles=None,
        )
        _, fov_mask = resample_with_fov_mask(lr_vol, lr_affine, hr_affine, (32, 32, 32))
        # Center-aligned, the LR FOV covers 6*5=30mm vs HR 32*1=32mm
        # Edges of HR grid outside LR FOV -> fov_mask=1 (missing)
        # Most of the volume should be 0 (valid)
        assert (fov_mask == 0).sum() > (fov_mask == 1).sum()

    def test_fov_mask_has_ones_with_rotation(self):
        """Oblique LR should produce ones (missing) at corners."""
        lr_vol = torch.rand(1, 32, 32, 6)
        hr_affine = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        lr_affine = build_lr_affine(
            hr_affine,
            through_plane_axis=2,
            lr_spacing_tp=5.0,
            hr_spacing_tp=1.0,
            lr_shape=(32, 32, 6),
            hr_shape=(32, 32, 32),
            rotation_angles=(0.3, 0.3, 0.3),
        )
        _, fov_mask = resample_with_fov_mask(lr_vol, lr_affine, hr_affine, (32, 32, 32))
        assert 1.0 in fov_mask, "Oblique rotation should produce ones (missing voxels) in FOV mask"

    def test_support_mask_produces_ones_where_dropped(self):
        """Support mask with zeros should produce 1s (missing) in FOV mask at corresponding HR positions."""
        lr_vol = torch.rand(1, 32, 32, 6)
        hr_affine = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        lr_affine = build_lr_affine(
            hr_affine,
            through_plane_axis=2,
            lr_spacing_tp=5.0,
            hr_spacing_tp=1.0,
            lr_shape=(32, 32, 6),
            hr_shape=(32, 32, 32),
            rotation_angles=None,
        )
        # Support mask: drop first and last 2 slices (keep middle 2 of 6)
        support_mask = torch.ones(1, 32, 32, 6)
        support_mask[:, :, :, :2] = 0.0
        support_mask[:, :, :, 4:] = 0.0

        _, fov_mask_with_support = resample_with_fov_mask(
            lr_vol, lr_affine, hr_affine, (32, 32, 32), support_mask=support_mask,
        )
        _, fov_mask_without = resample_with_fov_mask(
            lr_vol, lr_affine, hr_affine, (32, 32, 32),
        )
        # Support mask should produce significantly more missing (1s)
        assert fov_mask_with_support.sum() > fov_mask_without.sum(), (
            "Support mask with dropped slices should produce more missing voxels"
        )


class TestComputeObliqueFovMask:
    def test_aligned_mask_mostly_zeros(self):
        """Axis-aligned affine should produce mostly-zero mask."""
        hr_affine = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        lr_shape = (32, 32, 6)
        hr_shape = (32, 32, 32)
        lr_affine = build_lr_affine(
            hr_affine, through_plane_axis=2, lr_spacing_tp=5.0,
            hr_spacing_tp=1.0, lr_shape=lr_shape, hr_shape=hr_shape,
            rotation_angles=None,
        )
        mask = compute_oblique_fov_mask(lr_shape, lr_affine, hr_affine, hr_shape)
        assert (mask == 0).sum() > (mask == 1).sum()

    def test_oblique_mask_has_ones(self):
        """Oblique affine should produce ones (missing) at edges."""
        hr_affine = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        lr_shape = (32, 32, 6)
        hr_shape = (32, 32, 32)
        lr_affine = build_lr_affine(
            hr_affine, through_plane_axis=2, lr_spacing_tp=5.0,
            hr_spacing_tp=1.0, lr_shape=lr_shape, hr_shape=hr_shape,
            rotation_angles=(0.3, 0.3, 0.3),
        )
        mask = compute_oblique_fov_mask(lr_shape, lr_affine, hr_affine, hr_shape)
        assert 1.0 in mask

    def test_mask_is_binary(self):
        """FOV mask should only contain 0s and 1s."""
        hr_affine = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        lr_shape = (32, 32, 6)
        hr_shape = (32, 32, 32)
        lr_affine = build_lr_affine(
            hr_affine, through_plane_axis=2, lr_spacing_tp=5.0,
            hr_spacing_tp=1.0, lr_shape=lr_shape, hr_shape=hr_shape,
            rotation_angles=(0.1, -0.2, 0.15),
        )
        mask = compute_oblique_fov_mask(lr_shape, lr_affine, hr_affine, hr_shape)
        unique_vals = torch.unique(mask)
        assert all(v in [0.0, 1.0] for v in unique_vals.tolist())


class TestApplyFovSliceDropNative:
    def test_drops_slices(self):
        """Should zero out edge slices."""
        vol = torch.ones(1, 20, 20, 20)
        result = apply_fov_slice_drop_native(vol, through_plane_axis=0, keep_fraction=0.5)
        # 50% of 20 = 10 slices kept, 10 dropped
        assert (result[:, :5, :, :] == 0).all()  # 5 dropped from left
        assert (result[:, 15:, :, :] == 0).all()  # 5 dropped from right
        assert (result[:, 5:15, :, :] == 1).all()  # middle kept

    def test_full_keep(self):
        """keep_fraction=1.0 should not drop anything."""
        vol = torch.ones(1, 20, 20, 20)
        result = apply_fov_slice_drop_native(vol, through_plane_axis=1, keep_fraction=1.0)
        assert torch.allclose(result, vol)

    def test_drop_from_start_shared(self):
        """Explicit drop_from_start should be deterministic and shared across calls."""
        vol = torch.ones(1, 20, 20, 20)
        # drop_from_start=True -> drops the leading slices on axis 0
        from_start = apply_fov_slice_drop_native(
            vol, through_plane_axis=0, keep_fraction=0.5,
            force_both_sides=False, drop_from_start=True,
        )
        from_end = apply_fov_slice_drop_native(
            vol, through_plane_axis=0, keep_fraction=0.5,
            force_both_sides=False, drop_from_start=False,
        )
        assert (from_start[:, :10, :, :] == 0).all()
        assert (from_start[:, 10:, :, :] == 1).all()
        assert (from_end[:, :10, :, :] == 1).all()
        assert (from_end[:, 10:, :, :] == 0).all()


class TestComputeBrainBboxSupportMask:
    def test_bbox_around_bright_cuboid(self):
        """Mask should be 1 inside the bright cuboid bbox and 0 outside."""
        vol = torch.zeros(1, 32, 32, 32)
        vol[:, 8:24, 6:26, 10:22] = 1.0
        mask = compute_brain_bbox_support_mask(vol, threshold=0.5)
        assert mask.shape == (1, 32, 32, 32)
        # Inside bbox -> 1
        assert (mask[:, 8:24, 6:26, 10:22] == 1).all()
        # Outside bbox -> 0
        assert (mask[:, :8, :, :] == 0).all()
        assert (mask[:, 24:, :, :] == 0).all()
        assert (mask[:, :, :6, :] == 0).all()
        assert (mask[:, :, 26:, :] == 0).all()
        assert (mask[:, :, :, :10] == 0).all()
        assert (mask[:, :, :, 22:] == 0).all()

    def test_empty_foreground_falls_back_to_ones(self):
        """When no voxel exceeds threshold, mask should be all-ones."""
        vol = torch.zeros(1, 16, 16, 16)
        mask = compute_brain_bbox_support_mask(vol, threshold=0.5)
        assert (mask == 1).all()

    def test_margin_expands_bbox(self):
        """Margin should expand the bbox symmetrically (clipped to volume)."""
        vol = torch.zeros(1, 32, 32, 32)
        vol[:, 12:20, 12:20, 12:20] = 1.0
        mask = compute_brain_bbox_support_mask(vol, threshold=0.5, margin=2)
        # Bbox grew by 2 voxels each side -> 10:22
        assert (mask[:, 10:22, 10:22, 10:22] == 1).all()
        assert (mask[:, :10, :, :] == 0).all()

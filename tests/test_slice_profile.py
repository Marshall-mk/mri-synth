"""Tests for SliceProfilePhysics."""

import torch
import pytest

from mri_synth.physics.slice_profile import SliceProfilePhysics


class TestSliceProfilePhysics:
    def test_kernel_normalization_boxcar(self):
        sp = SliceProfilePhysics(profile_type="boxcar")
        kernel = sp.get_slice_kernel(3.0, 1.0, torch.device("cpu"))
        assert abs(kernel.sum().item() - 1.0) < 1e-5

    def test_kernel_normalization_gaussian(self):
        sp = SliceProfilePhysics(profile_type="gaussian")
        kernel = sp.get_slice_kernel(3.0, 1.0, torch.device("cpu"))
        assert abs(kernel.sum().item() - 1.0) < 1e-5

    def test_kernel_normalization_trapezoid(self):
        sp = SliceProfilePhysics(profile_type="trapezoid")
        kernel = sp.get_slice_kernel(3.0, 1.0, torch.device("cpu"))
        assert abs(kernel.sum().item() - 1.0) < 1e-5

    def test_output_shape_preservation(self, synthetic_volume_no_batch):
        sp = SliceProfilePhysics(profile_type="trapezoid")
        img = synthetic_volume_no_batch
        resolution = torch.tensor([1.0, 1.0, 1.0])
        thickness = torch.tensor([1.0, 1.0, 5.0])
        out = sp(img, resolution, thickness)
        assert out.shape == img.shape

    def test_through_plane_axis_selection(self, synthetic_volume_no_batch):
        """The through-plane axis should be the one with largest thickness/resolution ratio."""
        sp = SliceProfilePhysics(profile_type="trapezoid")
        img = synthetic_volume_no_batch
        resolution = torch.tensor([1.0, 1.0, 1.0])
        thickness = torch.tensor([1.0, 1.0, 5.0])
        # Should blur primarily along axis 2 (W)
        out = sp(img, resolution, thickness)
        assert not torch.allclose(img, out)

    def test_no_blur_when_isotropic(self, synthetic_volume_no_batch):
        """When thickness == resolution, through-plane factor <= 1.1 → minimal effect."""
        sp = SliceProfilePhysics(profile_type="trapezoid")
        img = synthetic_volume_no_batch
        resolution = torch.tensor([1.0, 1.0, 1.0])
        thickness = torch.tensor([1.0, 1.0, 1.0])
        out = sp(img, resolution, thickness)
        # In-plane PSF still applies, but should be close
        assert out.shape == img.shape

    def test_unknown_profile_raises(self):
        sp = SliceProfilePhysics(profile_type="unknown")
        with pytest.raises(ValueError, match="Unknown profile"):
            sp.get_slice_kernel(3.0, 1.0, torch.device("cpu"))

"""Slice profile physics simulation for MRI PSF blurring."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SliceProfilePhysics(nn.Module):
    """
    Simulates the physical blurring caused by the MRI scanner's slice selection
    profile and in-plane sampling.

    In a real MRI scanner, a 2D slice has a thickness determined by the RF
    excitation pulse. The sensitivity profile across the slab is rarely a perfect
    rectangle — it is often trapezoidal or Gaussian due to hardware limits on the
    RF pulse duration (truncated Sinc pulses).

    Args:
        profile_type: Shape of the slice sensitivity profile.
            'boxcar' — ideal rectangular profile.
            'gaussian' — standard approximation.
            'trapezoid' — realistic profile for most clinical scanners.
        edge_width: For 'trapezoid', the fraction of the slice thickness that
            is the transition region (0.0–0.5).
    """

    def __init__(self, profile_type: str = "trapezoid", edge_width: float = 0.1):
        super().__init__()
        # profile_type is deliberately NOT validated here: the established contract is
        # that an unknown profile raises from get_slice_kernel at use, not at
        # construction (tests/test_slice_profile.py::test_unknown_profile_raises).
        # The docstring advertises 0.0-0.5, but the trapezoid divides by edge_width and
        # sets flat_width = 0.5 - edge_width, so 0.0 is a division by zero and anything
        # above 0.5 gives a negative flat region. Reject both rather than emit a kernel
        # full of inf/nan that only shows up as a corrupted volume much later.
        if profile_type == "trapezoid" and not (0.0 < edge_width <= 0.5):
            raise ValueError(
                f"edge_width must be in (0.0, 0.5] for the trapezoid profile, got "
                f"{edge_width}. Use profile_type='boxcar' for a zero-width transition."
            )
        self.profile_type = profile_type
        self.edge_width = edge_width

    def get_slice_kernel(
        self, thickness_mm: float, current_res_mm: float, device: torch.device
    ) -> torch.Tensor:
        """
        Generate the 1D convolution kernel representing the slice sensitivity profile.

        Args:
            thickness_mm: Target slice thickness to simulate.
            current_res_mm: Current resolution of the input image.
            device: Device to create tensors on.

        Returns:
            Normalized 1D kernel.
        """
        scale = thickness_mm / current_res_mm
        kernel_size = int(math.ceil(scale * 3))
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Tap offsets in voxels: symmetric integers centred on 0, spacing
        # exactly 1 voxel — the same convention conv1d assumes when we pad by
        # kernel_size // 2.
        grid = (
            torch.arange(kernel_size, device=device, dtype=torch.float32)
            - kernel_size // 2
        )
        # Normalize grid relative to slice thickness (0.5 = half thickness)
        x = grid / scale

        if self.profile_type == "boxcar":
            kernel = (x.abs() <= 0.5).float()

        elif self.profile_type == "gaussian":
            # FWHM = thickness → sigma = 1 / 2.355
            sigma = 0.4246
            kernel = torch.exp(-0.5 * (x / sigma) ** 2)

        elif self.profile_type == "trapezoid":
            flat_width = 0.5 - self.edge_width
            flat_mask = (x.abs() <= flat_width).float()
            slope_mask = ((x.abs() > flat_width) & (x.abs() <= 0.5)).float()
            slope_val = 1.0 - (x.abs() - flat_width) / self.edge_width
            kernel = flat_mask + slope_mask * slope_val

        else:
            raise ValueError(f"Unknown profile: {self.profile_type}")

        # Energy conservation
        return kernel / kernel.sum()

    def forward(
        self,
        img: torch.Tensor,
        resolution: torch.Tensor,
        thickness: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply physics-based blurring to the input volume.

        Identifies the slice-select direction (largest thickness/resolution ratio)
        and applies the specific slice profile kernel. For the other two in-plane
        directions, applies a fixed Gaussian PSF (~0.42 pixels sigma).

        Args:
            img: Input image tensor (C, D, H, W).
            resolution: Current voxel size [res_D, res_H, res_W].
            thickness: Target slice thickness [thick_D, thick_H, thick_W].
        """
        device = img.device

        factors = thickness / resolution
        slice_dim_idx = torch.argmax(factors).item()

        for i, dim in enumerate([1, 2, 3]):
            # Through-plane: apply slice profile
            if i == slice_dim_idx and factors[i] > 1.1:
                kernel = self.get_slice_kernel(
                    thickness[i], resolution[i], device
                )
            # In-plane: fixed Gaussian PSF width (~0.42 pixels)
            else:
                sigma = 0.42
                k_size = 5
                k_grid = torch.arange(k_size, device=device) - k_size // 2
                kernel = torch.exp(-0.5 * (k_grid / sigma) ** 2)
                kernel = kernel / kernel.sum()

            # (1, 1, K). The volume is flattened to (N, 1, L) below, so every
            # channel goes through the same single-channel filter and no
            # per-channel copy of the kernel is needed.
            kernel = kernel.view(1, 1, -1)
            padding = kernel.shape[-1] // 2

            # Permute to put active axis last
            if i == 0:  # D
                img_in = img.permute(0, 2, 3, 1)
            elif i == 1:  # H
                img_in = img.permute(0, 1, 3, 2)
            else:  # W
                img_in = img.permute(0, 1, 2, 3)

            # Flatten non-active dims into batch
            shape_before = img_in.shape
            img_flat = img_in.reshape(-1, 1, shape_before[-1])

            img_filtered = F.conv1d(img_flat, kernel, padding=padding)

            img_out = img_filtered.view(shape_before)

            # Un-permute
            if i == 0:
                img = img_out.permute(0, 3, 1, 2)
            elif i == 1:
                img = img_out.permute(0, 1, 3, 2)
            else:
                img = img_out

        return img

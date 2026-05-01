"""MRI artifact simulation pipeline."""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from mri_synth.fov.resampling import (
    affine_resample_3d,
    apply_fov_slice_drop_native,
    build_lr_affine,
    compute_brain_bbox_support_mask,
    resample_with_fov_mask,
)
from mri_synth.physics.slice_profile import SliceProfilePhysics
from mri_synth.artifacts.kspace import (
    apply_kspace_motion_ghosting,
    apply_kspace_spike,
    apply_aliasing,
)


class MRIArtifactSimulator(nn.Module):
    """
    Physics engine that orchestrates the degradation pipeline.

    Pipeline:
    1. Slice profile (physics-based PSF blurring)
    2. K-space corruptions (motion ghosts, RF spikes)
    3. Aliasing (FOV wrap-around)
    4. Resolution loss (FFT cropping)
    5. FOV slice drop on native LR (optional)
    6. Affine-based resampling to HR grid + FOV mask generation
    7. Thermal noise (Rician)

    Args:
        volume_res: Input HR volume resolution in mm.
        target_res: Target LR resolution in mm.
        output_shape: Fixed output spatial shape, or None.
        prob_motion: Probability of motion ghosting.
        prob_spike: Probability of RF spike.
        prob_aliasing: Probability of aliasing.
        prob_noise: Probability of noise.
        noise_std: Standard deviation of Rician noise.
        motion_intensity: Intensity of motion ghosting.
        spike_intensity: Intensity of spike artifact.
        upsample_mode: Interpolation mode for upsampling.
        return_intermediate: If True, also return true LR before upsample.
        psf_profile_type: Slice profile type ('trapezoid', 'gaussian', 'boxcar').
        psf_edge_width: Edge width for trapezoid profile.
        obliqueness_range: Maximum rotation per axis in degrees.
        enable_obliqueness: If True, apply random oblique rotations.
        prob_obliqueness: Probability of applying obliqueness per stack.
        tight_fov: If True, size the LR scan FOV to the brain bounding box
            so the resulting FOV mask covers the air around the brain in HR
            space (mimics radiographer-sized FOV in real acquisitions).
        tight_fov_threshold: Intensity threshold for foreground detection.
        tight_fov_margin: Extra voxels added around the brain bbox.
    """

    def __init__(
        self,
        volume_res: List[float],
        target_res: List[float],
        output_shape: Optional[List[int]] = None,
        prob_motion: float = 0.2,
        prob_spike: float = 0.1,
        prob_aliasing: float = 0.1,
        prob_noise: float = 0.95,
        noise_std: float = 0.05,
        motion_intensity: float = 1.5,
        spike_intensity: float = 0.04,
        upsample_mode: str = "trilinear",
        return_intermediate: bool = False,
        psf_profile_type: str = "trapezoid",
        psf_edge_width: float = 0.1,
        obliqueness_range: float = 15.0,
        enable_obliqueness: bool = True,
        prob_obliqueness: float = 0.5,
        tight_fov: bool = True,
        tight_fov_threshold: float = 1e-3,
        tight_fov_margin: int = 0,
    ):
        super().__init__()
        self.volume_res = torch.tensor(volume_res, dtype=torch.float32)
        self.target_res = torch.tensor(target_res, dtype=torch.float32)
        self.output_shape = output_shape
        self.prob_motion = prob_motion
        self.prob_spike = prob_spike
        self.prob_aliasing = prob_aliasing
        self.prob_noise = prob_noise
        self.noise_std = noise_std
        self.motion_intensity = motion_intensity
        self.spike_intensity = spike_intensity
        self.upsample_mode = upsample_mode
        self.return_intermediate = return_intermediate
        self.obliqueness_range = obliqueness_range
        self.enable_obliqueness = enable_obliqueness
        self.prob_obliqueness = prob_obliqueness
        self.tight_fov = tight_fov
        self.tight_fov_threshold = tight_fov_threshold
        self.tight_fov_margin = tight_fov_margin
        self.physics_engine = SliceProfilePhysics(
            profile_type=psf_profile_type, edge_width=psf_edge_width
        )

    # ------------------------------------------------------------------
    # Per-step helpers (extracted to keep forward() readable). RNG ordering
    # within forward() is preserved exactly: helpers consume randoms in the
    # same sequence as the inlined code.
    # ------------------------------------------------------------------

    def _apply_kspace_artifacts(
        self,
        img: torch.Tensor,
        b: int,
        enable_motion: Optional[torch.Tensor],
        enable_spike: Optional[torch.Tensor],
        enable_aliasing: Optional[torch.Tensor],
        motion_axis: Optional[torch.Tensor],
        aliasing_axis: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """STEPS 2-3: motion ghosting, RF spike, aliasing."""
        if enable_motion is not None:
            should_apply_motion = enable_motion[b].item()
        else:
            should_apply_motion = torch.rand(1).item() < self.prob_motion
        if should_apply_motion:
            if motion_axis is not None:
                axis = motion_axis[b].item()
            else:
                # Bug fix history: previously randint(1, 3) skipped axis 0.
                axis = torch.randint(0, 3, (1,)).item()
            img = apply_kspace_motion_ghosting(
                img, axis=axis, intensity=self.motion_intensity
            )

        if enable_spike is not None:
            should_apply_spike = enable_spike[b].item()
        else:
            should_apply_spike = torch.rand(1).item() < self.prob_spike
        if should_apply_spike:
            img = apply_kspace_spike(img, intensity=self.spike_intensity)

        if enable_aliasing is not None:
            should_apply_aliasing = enable_aliasing[b].item()
        else:
            should_apply_aliasing = torch.rand(1).item() < self.prob_aliasing
        if should_apply_aliasing:
            if aliasing_axis is not None:
                axis = aliasing_axis[b].item()
            else:
                axis = torch.randint(0, 3, (1,)).item()
            img = apply_aliasing(img, axis=axis, fold_pct=0.15)
        return img

    @staticmethod
    def _fft_downsample(img: torch.Tensor, downsample_axis: int, factor: float) -> torch.Tensor:
        """STEP 4: FFT-based downsampling via k-space cropping along ``downsample_axis``."""
        original_shape = img.shape
        spatial_axis = downsample_axis + 1  # img is (C, D, H, W)
        new_size = int(round(original_shape[spatial_axis] / factor))

        fft_volume = torch.fft.fftn(img, dim=(1, 2, 3))
        fft_volume = torch.fft.fftshift(fft_volume, dim=(1, 2, 3))

        center_idx = original_shape[spatial_axis] // 2
        crop_start = center_idx - new_size // 2
        crop_end = crop_start + new_size

        if downsample_axis == 0:
            cropped_fft = fft_volume[:, crop_start:crop_end, :, :]
        elif downsample_axis == 1:
            cropped_fft = fft_volume[:, :, crop_start:crop_end, :]
        else:
            cropped_fft = fft_volume[:, :, :, crop_start:crop_end]

        cropped_fft = torch.fft.ifftshift(cropped_fft, dim=(1, 2, 3))
        scale_factor = new_size / original_shape[spatial_axis]
        return torch.real(torch.fft.ifftn(cropped_fft, dim=(1, 2, 3))) * scale_factor

    def _sample_obliqueness(
        self, device: torch.device
    ) -> Optional[Tuple[float, float, float]]:
        """Sample (rx, ry, rz) rotation angles in radians, or ``None`` if disabled.

        Consumes 1 random for the prob check and 3 for the angles, mirroring
        the inlined logic so RNG ordering stays stable.
        """
        if not (self.enable_obliqueness and self.obliqueness_range > 0):
            return None
        if torch.rand(1).item() >= self.prob_obliqueness:
            return None
        max_rad = self.obliqueness_range * math.pi / 180.0
        rx = torch.empty(1, device=device).uniform_(-max_rad, max_rad).item()
        ry = torch.empty(1, device=device).uniform_(-max_rad, max_rad).item()
        rz = torch.empty(1, device=device).uniform_(-max_rad, max_rad).item()
        return (rx, ry, rz)

    def _resample_lr_to_hr(
        self,
        img: torch.Tensor,
        hr_support: Optional[torch.Tensor],
        downsample_axis: int,
        target_shape: Tuple[int, int, int],
        acq_res: torch.Tensor,
        fov_drop_applied: bool,
        keep_frac: float,
        fov_force_both_sides: bool,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[float, float, float]]]:
        """STEPS 5-6: native FOV slice drop, obliqueness, affine resample to HR.

        Returns ``(img_hr, fov_mask, true_lr_native, rotation_angles)``. The
        ``true_lr_native`` tensor is whatever the scanner is conceptually
        emitting at native LR resolution — the oblique-LR volume when a
        rotation was sampled, otherwise the axis-aligned LR.
        """
        vol_res = self.volume_res.to(device)
        hr_affine = torch.diag(
            torch.tensor(
                [vol_res[0], vol_res[1], vol_res[2], 1.0], device=device
            )
        )
        lr_native_shape = tuple(img.shape[1:])
        resample_mode = (
            self.upsample_mode if self.upsample_mode != "trilinear" else "bilinear"
        )

        lr_affine_aligned = build_lr_affine(
            hr_affine=hr_affine,
            through_plane_axis=downsample_axis,
            lr_spacing_tp=acq_res[downsample_axis].item(),
            hr_spacing_tp=vol_res[downsample_axis].item(),
            lr_shape=lr_native_shape,
            hr_shape=target_shape,
            rotation_angles=None,
        )

        # Mirror the HR brain bbox support into axis-aligned LR space.
        if hr_support is not None:
            lr_support = affine_resample_3d(
                hr_support, hr_affine, lr_affine_aligned,
                lr_native_shape, mode="nearest",
            )
        else:
            lr_support = None

        # STEP 5: FOV slice drop on native LR (image + support share the
        # same drop pattern so the FOV mask reflects the drop).
        if fov_drop_applied:
            drop_from_start = (
                None
                if fov_force_both_sides
                else bool(torch.rand(1).item() < 0.5)
            )
            img = apply_fov_slice_drop_native(
                img,
                through_plane_axis=downsample_axis,
                keep_fraction=keep_frac,
                force_both_sides=fov_force_both_sides,
                drop_from_start=drop_from_start,
            )
            if lr_support is not None:
                lr_support = apply_fov_slice_drop_native(
                    lr_support,
                    through_plane_axis=downsample_axis,
                    keep_fraction=keep_frac,
                    force_both_sides=fov_force_both_sides,
                    drop_from_start=drop_from_start,
                )

        rotation_angles = self._sample_obliqueness(device)

        if rotation_angles is None:
            # Axis-aligned: image and support already in axis-aligned LR.
            img_hr, fov_mask = resample_with_fov_mask(
                img, lr_affine_aligned, hr_affine,
                target_shape, mode=resample_mode,
                support_mask=lr_support,
            )
            return img_hr, fov_mask, img, None

        # Oblique branch: build the tilted affine and rotate both image and
        # support into oblique LR space before the final HR resample.
        lr_affine_oblique = build_lr_affine(
            hr_affine=hr_affine,
            through_plane_axis=downsample_axis,
            lr_spacing_tp=acq_res[downsample_axis].item(),
            hr_spacing_tp=vol_res[downsample_axis].item(),
            lr_shape=lr_native_shape,
            hr_shape=target_shape,
            rotation_angles=rotation_angles,
        )
        oblique_lr = affine_resample_3d(
            img, lr_affine_aligned, lr_affine_oblique,
            lr_native_shape, mode=resample_mode,
        )
        if lr_support is not None:
            lr_support = affine_resample_3d(
                lr_support, lr_affine_aligned, lr_affine_oblique,
                lr_native_shape, mode="nearest",
            )
        img_hr, fov_mask = resample_with_fov_mask(
            oblique_lr, lr_affine_oblique, hr_affine,
            target_shape, mode=resample_mode,
            support_mask=lr_support,
        )
        return img_hr, fov_mask, oblique_lr, rotation_angles

    def _apply_noise(
        self, img: torch.Tensor, b: int, enable_noise: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """STEP 7: Rician noise."""
        if enable_noise is not None:
            should_apply_noise = enable_noise[b].item()
        else:
            should_apply_noise = (
                self.prob_noise > 0 and torch.rand(1).item() < self.prob_noise
            )
        if should_apply_noise:
            n1 = torch.randn_like(img) * self.noise_std
            n2 = torch.randn_like(img) * self.noise_std
            img = torch.sqrt((img + n1) ** 2 + n2 ** 2)
        return img

    def forward(
        self,
        image: torch.Tensor,
        acquisition_res: torch.Tensor,
        thickness: Optional[torch.Tensor] = None,
        enable_motion: Optional[torch.Tensor] = None,
        enable_spike: Optional[torch.Tensor] = None,
        enable_aliasing: Optional[torch.Tensor] = None,
        enable_noise: Optional[torch.Tensor] = None,
        motion_axis: Optional[torch.Tensor] = None,
        aliasing_axis: Optional[torch.Tensor] = None,
        fov_drop_decisions: Optional[torch.Tensor] = None,
        fov_keep_fractions: Optional[torch.Tensor] = None,
        fov_force_both_sides: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MRI artifact simulation with affine-based resampling.

        Args:
            image: Input volume (B, C, D, H, W).
            acquisition_res: Resolution per batch (B, 3) or (3,).
            thickness: Slice thickness per batch (B, 3) or (3,).
            enable_motion: Pre-sampled bool mask (B,).
            enable_spike: Pre-sampled bool mask (B,).
            enable_aliasing: Pre-sampled bool mask (B,).
            enable_noise: Pre-sampled bool mask (B,).
            motion_axis: Pre-sampled axis (B,).
            aliasing_axis: Pre-sampled axis (B,).
            fov_drop_decisions: Per-batch bool (B,) — True to apply FOV drop.
            fov_keep_fractions: Per-batch float (B,) — fraction of slices to keep.
            fov_force_both_sides: Drop from both ends of the through-plane axis.

        Returns:
            If return_intermediate is False:
                Tuple of (simulated_lr, fov_masks) each (B, C, D, H, W).
            If return_intermediate is True:
                Tuple of (simulated_lr, fov_masks, true_lr_stacks) where
                true_lr_stacks is (B, C, D', H', W') before resampling.
        """
        batch_size = image.shape[0]
        device = image.device
        outputs = []
        fov_mask_outputs = []
        true_lr_outputs = [] if self.return_intermediate else None
        self._last_rotation_angles = []

        for b in range(batch_size):
            img = image[b]  # (C, D, H, W)
            original_input_shape = img.shape[1:]
            acq_res = (
                acquisition_res[b] if acquisition_res.ndim > 1 else acquisition_res
            ).to(device)

            # Capture HR brain bbox support BEFORE PSF blur so the bbox
            # tracks the original tissue extent, not the blurred halo.
            hr_support = (
                compute_brain_bbox_support_mask(
                    img,
                    threshold=self.tight_fov_threshold,
                    margin=self.tight_fov_margin,
                )
                if self.tight_fov
                else None
            )

            if thickness is not None:
                thk = (thickness[b] if thickness.ndim > 1 else thickness).to(device)
            else:
                thk = acq_res

            # STEP 1: PSF blurring
            img = self.physics_engine(
                img, resolution=self.volume_res.to(device), thickness=thk,
            )

            # STEPS 2-3: K-space artifacts (motion, spike, aliasing).
            img = self._apply_kspace_artifacts(
                img, b, enable_motion, enable_spike, enable_aliasing,
                motion_axis, aliasing_axis,
            )

            # STEP 4: Resolution reduction (FFT downsample).
            factors = acq_res / self.volume_res.to(device)
            downsample_axis = torch.argmax(factors).item()
            factor = factors[downsample_axis].item()

            if factor > 1.1:
                img = self._fft_downsample(img, downsample_axis, factor)

                target_shape = (
                    tuple(self.output_shape)
                    if self.output_shape is not None
                    else tuple(original_input_shape)
                )
                fov_drop_applied = (
                    fov_drop_decisions is not None
                    and fov_drop_decisions[b].item()
                    and fov_keep_fractions is not None
                )
                keep_frac = (
                    fov_keep_fractions[b].item() if fov_drop_applied else 0.0
                )

                # STEPS 5-6: native FOV slice drop, obliqueness, resample to HR.
                img, fov_mask, true_lr_native, rotation_angles = (
                    self._resample_lr_to_hr(
                        img, hr_support, downsample_axis, target_shape,
                        acq_res, fov_drop_applied, keep_frac,
                        fov_force_both_sides, device,
                    )
                )
                self._last_rotation_angles.append(rotation_angles)
                if self.return_intermediate:
                    true_lr_outputs.append(true_lr_native.clone().unsqueeze(0))
                fov_mask_outputs.append(fov_mask.unsqueeze(0))
            else:
                # No significant downsampling — no resampling needed.
                self._last_rotation_angles.append(None)
                if self.return_intermediate:
                    true_lr_outputs.append(img.clone().unsqueeze(0))
                fov_mask = torch.zeros(1, *img.shape[1:], device=device)
                fov_mask_outputs.append(fov_mask.unsqueeze(0))

            # STEP 7: Rician noise.
            img = self._apply_noise(img, b, enable_noise)
            outputs.append(img.unsqueeze(0))

        final_output = torch.cat(outputs, dim=0)
        final_fov_masks = torch.cat(fov_mask_outputs, dim=0)

        if self.return_intermediate:
            true_lr_output = torch.cat(true_lr_outputs, dim=0)
            return final_output, final_fov_masks, true_lr_output
        else:
            return final_output, final_fov_masks

"""MRI artifact simulation pipeline."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    5. Thermal noise (Rician)

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
        preserve_input_shape: If True, upsample back to input shape.
        return_intermediate: If True, also return true LR before upsample.
        psf_profile_type: Slice profile type ('trapezoid', 'gaussian', 'boxcar').
        psf_edge_width: Edge width for trapezoid profile.
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
        preserve_input_shape: bool = True,
        return_intermediate: bool = False,
        psf_profile_type: str = "trapezoid",
        psf_edge_width: float = 0.1,
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
        self.preserve_input_shape = preserve_input_shape
        self.return_intermediate = return_intermediate
        self.physics_engine = SliceProfilePhysics(
            profile_type=psf_profile_type, edge_width=psf_edge_width
        )

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
    ) -> torch.Tensor:
        """
        Apply MRI artifact simulation.

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

        Returns:
            Simulated LR volume, or tuple of (upsampled, true_lr) if
            return_intermediate is True.
        """
        batch_size = image.shape[0]
        device = image.device
        outputs = []
        true_lr_outputs = [] if self.return_intermediate else None

        for b in range(batch_size):
            img = image[b]  # (C, D, H, W)
            original_input_shape = img.shape[1:]
            acq_res = (
                acquisition_res[b] if acquisition_res.ndim > 1 else acquisition_res
            )
            acq_res = acq_res.to(device)

            if thickness is not None:
                thk = thickness[b] if thickness.ndim > 1 else thickness
                thk = thk.to(device)
            else:
                thk = acq_res

            # STEP 1: PSF blurring
            img = self.physics_engine(
                img,
                resolution=self.volume_res.to(device),
                thickness=thk,
            )

            # STEP 2: K-space artifacts
            if enable_motion is not None:
                should_apply_motion = enable_motion[b].item()
            else:
                should_apply_motion = torch.rand(1).item() < self.prob_motion

            if should_apply_motion:
                if motion_axis is not None:
                    axis = motion_axis[b].item()
                else:
                    # BUG FIX: was randint(1, 3) — now includes axis 0
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

            # STEP 3: Aliasing
            if enable_aliasing is not None:
                should_apply_aliasing = enable_aliasing[b].item()
            else:
                should_apply_aliasing = torch.rand(1).item() < self.prob_aliasing

            if should_apply_aliasing:
                if aliasing_axis is not None:
                    axis = aliasing_axis[b].item()
                else:
                    # BUG FIX: was randint(1, 3) — now includes axis 0
                    axis = torch.randint(0, 3, (1,)).item()
                img = apply_aliasing(img, axis=axis, fold_pct=0.15)

            # STEP 4: Resolution reduction (FFT downsample)
            factors = acq_res / self.volume_res.to(device)
            downsample_axis = torch.argmax(factors).item()
            factor = factors[downsample_axis].item()

            true_lr_img = None
            if factor > 1.1:
                original_shape = img.shape
                spatial_axis = downsample_axis + 1
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
                img = torch.real(torch.fft.ifftn(cropped_fft, dim=(1, 2, 3)))

                if self.return_intermediate:
                    true_lr_img = img.clone()

                if self.preserve_input_shape:
                    target_shape = original_input_shape
                elif self.output_shape is not None:
                    target_shape = self.output_shape
                else:
                    target_shape = None

                if target_shape is not None and list(img.shape[1:]) != list(
                    target_shape
                ):
                    img = F.interpolate(
                        img.unsqueeze(0),
                        size=target_shape,
                        mode=self.upsample_mode,
                    ).squeeze(0)

            if self.return_intermediate:
                if true_lr_img is not None:
                    true_lr_outputs.append(true_lr_img.unsqueeze(0))
                else:
                    true_lr_outputs.append(img.clone().unsqueeze(0))

            # STEP 5: Noise
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

            outputs.append(img.unsqueeze(0))

        final_output = torch.cat(outputs, dim=0)

        if self.return_intermediate:
            true_lr_output = torch.cat(true_lr_outputs, dim=0)
            return final_output, true_lr_output
        else:
            return final_output

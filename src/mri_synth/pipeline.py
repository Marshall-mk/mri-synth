"""HR→LR data generation pipeline."""

from typing import Dict, List, Optional, Tuple

import torch
from monai.transforms import ScaleIntensityRangePercentiles

from mri_synth.artifacts.simulator import MRIArtifactSimulator
from mri_synth.config import GenerationConfig
from mri_synth.fov.multi_orientation import apply_fov_augmentation
from mri_synth.physics.bias_field import BiasFieldCorruption
from mri_synth.physics.intensity import IntensityAugmentation
from mri_synth.resolution import ResolutionConfig


class HRLRDataGenerator:
    """
    Domain-randomization pipeline using frequency-domain downsampling.

    Generates N orthogonal LR stacks from each HR volume using FFT-based
    k-space cropping for realistic MRI simulation.

    Args:
        atlas_res: Resolution of input HR images in mm.
        target_res: Target output resolution in mm.
        output_shape: Output spatial shape [D, H, W].
        num_stacks: Number of LR stacks to produce (default 3).
        prob_motion: Probability of motion ghosting.
        prob_spike: Probability of RF spike.
        prob_aliasing: Probability of aliasing.
        prob_bias_field: Probability of bias field corruption.
        prob_noise: Probability of noise.
        fov_augmentation_prob: Probability of FOV augmentation.
        min_resolution: Minimum resolution per axis.
        max_res_aniso: Maximum anisotropic resolution per axis.
        randomise_res: If True, randomize acquisition resolution.
        apply_intensity_aug: If True, apply intensity augmentation.
        clip_to_unit_range: If True, clip outputs to [0, 1].
        orientation_dropout_prob: Probability of orientation dropout.
        min_orientations: Minimum orientations to keep after dropout.
        drop_orientations: Specific orientations to always drop.
        upsample_mode: Interpolation mode for upsampling.
        preserve_input_shape: If True, upsample back to input shape.
        return_intermediate: If True, return true LR before upsample.
        psf_profile_type: Slice profile type.
        psf_edge_width: Edge width for trapezoid profile.
        fov_min_keep: Minimum fraction of slices to keep in FOV sim.
        fov_max_keep: Maximum fraction of slices to keep in FOV sim.
        fov_ensure_coverage: If True, ensure complementary FOV coverage.
        fov_force_both_sides: If True, drop from both ends in FOV sim.
    """

    def __init__(
        self,
        atlas_res: list = None,
        target_res: list = None,
        output_shape: list = None,
        num_stacks: int = 3,
        # Probabilities
        prob_motion: float = 0.2,
        prob_spike: float = 0.05,
        prob_aliasing: float = 0.1,
        prob_bias_field: float = 0.5,
        prob_noise: float = 0.8,
        fov_augmentation_prob: float = 0.7,
        # Resolution simulation
        min_resolution: list = None,
        max_res_aniso: list = None,
        randomise_res: bool = True,
        # Toggles
        apply_intensity_aug: bool = False,
        clip_to_unit_range: bool = True,
        # Orientation dropout
        orientation_dropout_prob: float = 0.0,
        min_orientations: int = 1,
        drop_orientations: list = None,
        # Interpolation mode
        upsample_mode: str = "trilinear",
        preserve_input_shape: bool = True,
        # LR stack saving
        return_intermediate: bool = False,
        # PSF configuration
        psf_profile_type: str = "trapezoid",
        psf_edge_width: float = 0.1,
        # FOV configuration
        fov_min_keep: float = 0.40,
        fov_max_keep: float = 0.70,
        fov_ensure_coverage: bool = True,
        fov_force_both_sides: bool = True,
    ):
        if atlas_res is None:
            atlas_res = [1.0, 1.0, 1.0]
        if target_res is None:
            target_res = [1.0, 1.0, 1.0]
        if min_resolution is None:
            min_resolution = [1.0, 1.0, 1.0]
        if max_res_aniso is None:
            max_res_aniso = [9.0, 9.0, 9.0]

        self.atlas_res = atlas_res
        self.target_res = target_res
        self.output_shape = output_shape
        self.num_stacks = num_stacks
        self.randomise_res = randomise_res
        self.apply_intensity_aug = apply_intensity_aug
        self.clip_to_unit_range = clip_to_unit_range

        self.prob_bias_field = prob_bias_field
        self.fov_augmentation_prob = fov_augmentation_prob
        self.fov_min_keep = fov_min_keep
        self.fov_max_keep = fov_max_keep
        self.fov_ensure_coverage = fov_ensure_coverage
        self.fov_force_both_sides = fov_force_both_sides
        self.upsample_mode = upsample_mode
        self.preserve_input_shape = preserve_input_shape
        self.return_intermediate = return_intermediate

        # Orientation dropout
        self.orientation_dropout_prob = orientation_dropout_prob
        self.min_orientations = max(1, min(min_orientations, 3))
        self.drop_orientations = drop_orientations

        if self.drop_orientations is not None:
            if len(self.drop_orientations) >= self.num_stacks:
                raise ValueError("Cannot drop all orientations.")
            if any(idx not in range(self.num_stacks) for idx in self.drop_orientations):
                raise ValueError(
                    f"drop_orientations must contain indices in [0, {self.num_stacks})"
                )

        # Resolution config (replaces SampleResolution nn.Module — Bug Fix #2)
        if randomise_res:
            self.res_config = ResolutionConfig(
                min_resolution=min_resolution,
                max_res_aniso=max_res_aniso,
            )

        # Bias field
        self.bias = BiasFieldCorruption(
            bias_field_std=0.3, bias_scale=0.025, prob=1.0
        )

        # Intensity augmentation
        if apply_intensity_aug:
            self.intensity_aug = IntensityAugmentation(
                clip=False,
                gamma_std=0.5,
                channel_wise=False,
                prob_gamma=0.5,
            )

        # MRI artifact simulator
        self.artifact_simulator = MRIArtifactSimulator(
            volume_res=atlas_res,
            target_res=target_res,
            output_shape=output_shape,
            prob_motion=prob_motion,
            prob_spike=prob_spike,
            prob_aliasing=prob_aliasing,
            prob_noise=prob_noise,
            noise_std=0.02,
            motion_intensity=0.5,
            upsample_mode=upsample_mode,
            preserve_input_shape=preserve_input_shape,
            return_intermediate=return_intermediate,
            psf_profile_type=psf_profile_type,
            psf_edge_width=psf_edge_width,
        )

        self.normalizer = ScaleIntensityRangePercentiles(
            lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        )

    @classmethod
    def from_config(cls, config: GenerationConfig) -> "HRLRDataGenerator":
        """Create a generator from a GenerationConfig."""
        return cls(
            atlas_res=config.atlas_res,
            target_res=config.target_res,
            num_stacks=config.num_stacks,
            prob_motion=config.artifacts.prob_motion,
            prob_spike=config.artifacts.prob_spike,
            prob_aliasing=config.artifacts.prob_aliasing,
            prob_bias_field=config.physics.prob_bias_field,
            prob_noise=config.artifacts.prob_noise,
            fov_augmentation_prob=config.fov.prob if config.fov.enable else 0.0,
            min_resolution=config.min_resolution,
            max_res_aniso=config.max_res_aniso,
            randomise_res=config.randomise_res,
            apply_intensity_aug=config.apply_intensity_aug,
            clip_to_unit_range=config.clip_to_unit_range,
            orientation_dropout_prob=config.orientation_dropout_prob,
            min_orientations=config.min_orientations,
            drop_orientations=config.drop_orientations,
            upsample_mode=config.upsample_mode,
            preserve_input_shape=config.preserve_input_shape,
            return_intermediate=config.return_intermediate,
            psf_profile_type=config.physics.psf_type,
            psf_edge_width=config.physics.edge_width,
            fov_min_keep=config.fov.min_keep,
            fov_max_keep=config.fov.max_keep,
            fov_ensure_coverage=config.fov.ensure_coverage,
            fov_force_both_sides=config.fov.force_both_sides,
        )

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image to [0, 1] using percentile scaling."""
        return self.normalizer(image)

    def _create_orientation_mask(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create orientation mask for dropout (simulating missing views).

        Returns:
            Boolean mask of shape (batch_size, num_stacks).
        """
        mask = torch.ones(batch_size, self.num_stacks, dtype=torch.bool, device=device)

        # Deterministic dropout
        if self.drop_orientations is not None and len(self.drop_orientations) > 0:
            for idx in self.drop_orientations:
                mask[:, idx] = False
            return mask

        # Random dropout
        if self.orientation_dropout_prob > 0.0:
            for b in range(batch_size):
                if torch.rand(1).item() < self.orientation_dropout_prob:
                    num_keep = torch.randint(
                        self.min_orientations, self.num_stacks + 1, (1,)
                    ).item()
                    if num_keep < self.num_stacks:
                        indices = torch.randperm(self.num_stacks)[:num_keep]
                        mask[b, :] = False
                        mask[b, indices] = True

        return mask

    def _create_orthogonal_resolutions(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Create N orthogonal anisotropic resolution configurations.

        For N stacks, cycles through through-plane axes [2, 1, 0] pattern:
        - Stack 0 (Axial):    through-plane axis 2 (S)
        - Stack 1 (Coronal):  through-plane axis 1 (A)
        - Stack 2 (Sagittal): through-plane axis 0 (R)
        - Stack 3+: cycles back through [2, 1, 0, ...]

        Returns:
            Tuple of (resolutions_list, thickness_list), each containing
            num_stacks tensors of shape (batch_size, 3).
        """
        if hasattr(self, "res_config"):
            min_res = torch.tensor(self.res_config.min_resolution, device=device)
            max_res = torch.tensor(self.res_config.max_res_aniso, device=device)
        else:
            min_res = torch.tensor([1.0, 1.0, 1.0], device=device)
            max_res = torch.tensor([9.0, 9.0, 9.0], device=device)

        high_res_value = min_res.min().item()

        # Sample low resolution ONCE per patient
        low_res_samples = []
        for b in range(batch_size):
            if self.randomise_res:
                low_res = (
                    torch.rand(1, device=device).item()
                    * (max_res.max() - high_res_value)
                    + high_res_value
                )
            else:
                low_res = max_res.max().item()
            low_res_samples.append(low_res)

        # Cycle through-plane axes: [2, 1, 0, 2, 1, 0, ...]
        through_plane_axes = [2, 1, 0]

        resolutions = []
        thicknesses = []

        for stack_idx in range(self.num_stacks):
            through_plane_axis = through_plane_axes[stack_idx % 3]

            res_batch = []
            thick_batch = []

            for b in range(batch_size):
                low_res = low_res_samples[b]

                res = torch.zeros(3, device=device)
                thick = torch.zeros(3, device=device)

                for axis in range(3):
                    if axis == through_plane_axis:
                        # Through-plane: low resolution
                        # For stacks beyond the first 3, sample fresh low_res
                        if stack_idx >= 3 and self.randomise_res:
                            fresh_low = (
                                torch.rand(1, device=device).item()
                                * (max_res.max() - high_res_value)
                                + high_res_value
                            )
                            res[axis] = fresh_low
                            thick[axis] = fresh_low
                        else:
                            res[axis] = low_res
                            thick[axis] = low_res
                    else:
                        res[axis] = high_res_value
                        thick[axis] = high_res_value

                res_batch.append(res)
                thick_batch.append(thick)

            resolutions.append(torch.stack(res_batch, dim=0))
            thicknesses.append(torch.stack(thick_batch, dim=0))

        return resolutions, thicknesses

    def generate_paired_data(
        self,
        hr_images: torch.Tensor,
        return_resolution: bool = False,
        sample_info: Optional[Dict] = None,
    ):
        """
        Generate paired LR-HR training data with N orthogonal LR stacks.

        Args:
            hr_images: High-resolution input images (B, C, D, H, W) in RAS.
            return_resolution: If True, also return resolution and thickness.
            sample_info: Optional metadata dict.

        Returns:
            Variable-length tuple depending on return_resolution and
            return_intermediate flags.
        """
        batch_size = hr_images.shape[0]
        device = hr_images.device

        # STEP 1: Normalize HR
        hr_augmented = self._normalize_image(hr_images)

        # STEP 2: Create N orthogonal LR stacks
        resolutions, thicknesses = self._create_orthogonal_resolutions(
            batch_size, device
        )

        # STEP 3: Pre-sample artifact decisions (once per patient)
        apply_bias_field = torch.rand(batch_size, device=device) < self.prob_bias_field
        apply_motion = (
            torch.rand(batch_size, device=device)
            < self.artifact_simulator.prob_motion
        )
        apply_spike = (
            torch.rand(batch_size, device=device)
            < self.artifact_simulator.prob_spike
        )
        apply_aliasing = (
            torch.rand(batch_size, device=device)
            < self.artifact_simulator.prob_aliasing
        )
        apply_noise = (
            torch.rand(batch_size, device=device)
            < self.artifact_simulator.prob_noise
        )

        # BUG FIX: was randint(1, 3) — now includes axis 0
        motion_axis = torch.randint(0, 3, (batch_size,), device=device)
        aliasing_axis = torch.randint(0, 3, (batch_size,), device=device)

        # Apply bias field ONCE (shared across all stacks)
        hr_degraded = hr_augmented.clone()
        for b in range(batch_size):
            if apply_bias_field[b]:
                hr_degraded[b : b + 1] = self.bias(hr_degraded[b : b + 1])

        # Apply intensity augmentation ONCE (shared across all stacks)
        if self.apply_intensity_aug:
            for b in range(batch_size):
                hr_degraded[b : b + 1] = self.intensity_aug(hr_degraded[b : b + 1])

        lr_stacks = []
        true_lr_stacks = []

        for stack_idx in range(self.num_stacks):
            lr_images = hr_degraded.clone()

            # Physics simulation
            resolution = resolutions[stack_idx]
            thickness = thicknesses[stack_idx]

            # BUG FIX #5: use self.return_intermediate consistently
            if self.return_intermediate:
                lr_images, true_lr_images = self.artifact_simulator(
                    lr_images,
                    resolution,
                    thickness,
                    enable_motion=apply_motion,
                    enable_spike=apply_spike,
                    enable_aliasing=apply_aliasing,
                    enable_noise=apply_noise,
                    motion_axis=motion_axis,
                    aliasing_axis=aliasing_axis,
                )
            else:
                lr_images = self.artifact_simulator(
                    lr_images,
                    resolution,
                    thickness,
                    enable_motion=apply_motion,
                    enable_spike=apply_spike,
                    enable_aliasing=apply_aliasing,
                    enable_noise=apply_noise,
                    motion_axis=motion_axis,
                    aliasing_axis=aliasing_axis,
                )

            # Normalize LR to [0, 1] — match HR's simple clamp
            if self.clip_to_unit_range:
                lr_images = torch.clamp(lr_images, 0.0, 1.0)
                if self.return_intermediate:
                    true_lr_images = torch.clamp(true_lr_images, 0.0, 1.0)

            lr_stacks.append(lr_images)
            if self.return_intermediate:
                true_lr_stacks.append(true_lr_images)

        hr_augmented = torch.clamp(hr_augmented, 0.0, 1.0)

        orientation_mask = self._create_orientation_mask(batch_size, device)

        if self.fov_augmentation_prob > 0:
            lr_stacks, spatial_masks = apply_fov_augmentation(
                lr_stacks,
                prob=self.fov_augmentation_prob,
                min_keep=self.fov_min_keep,
                max_keep=self.fov_max_keep,
                ensure_coverage=self.fov_ensure_coverage,
                force_both_sides=self.fov_force_both_sides,
            )
        else:
            spatial_masks = [
                torch.ones(batch_size, 1, *lr_stacks[0].shape[-3:], device=device)
                for _ in range(self.num_stacks)
            ]

        if return_resolution and self.return_intermediate:
            return (
                lr_stacks,
                true_lr_stacks,
                hr_augmented,
                resolutions,
                thicknesses,
                orientation_mask,
                spatial_masks,
            )
        elif return_resolution:
            return (
                lr_stacks,
                hr_augmented,
                resolutions,
                thicknesses,
                orientation_mask,
                spatial_masks,
            )
        else:
            return lr_stacks, hr_augmented, orientation_mask, spatial_masks

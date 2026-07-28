"""HR→LR data generation pipeline."""

from typing import Dict, List, Optional, Tuple

import torch
from monai.transforms import ScaleIntensityRangePercentiles

from mri_synth.artifacts.simulator import MRIArtifactSimulator
from mri_synth.config import GenerationConfig
from mri_synth.physics.bias_field import BiasFieldCorruption
from mri_synth.physics.intensity import IntensityAugmentation
from mri_synth.resolution import ResolutionConfig


class HRLRDataGenerator:
    """
    Domain-randomization pipeline using frequency-domain downsampling.

    Generates N orthogonal LR stacks from each HR volume using FFT-based
    k-space cropping for realistic MRI simulation. Each stack is resampled
    back to the HR grid via affine-based resampling, producing a physically
    correct FOV mask that captures out-of-bounds regions.

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
        bias_field_std: Std of the bias field coefficients.
        noise_std: Std of the additive Rician noise.
        motion_intensity: Intensity of k-space motion ghosting.
        spike_intensity: Intensity of the RF spike artifact.
        structural_only: If True, disable every appearance corruption
            (bias, noise, intensity/gamma, motion, spike, aliasing) so only
            downsampling and FOV transforms are applied.
        apply_intensity_aug: If True, apply intensity augmentation.
        clip_to_unit_range: If True, clip outputs to [0, 1].
        orientation_dropout_prob: Probability of orientation dropout.
        min_orientations: Minimum orientations to keep after dropout.
        drop_orientations: Specific orientations to always drop.
        upsample_mode: Interpolation mode for upsampling.
        return_intermediate: If True, return true LR before upsample.
        psf_profile_type: Slice profile type.
        psf_edge_width: Edge width for trapezoid profile.
        fov_min_keep: Minimum fraction of slices to keep in FOV sim.
        fov_max_keep: Maximum fraction of slices to keep in FOV sim.
        fov_ensure_coverage: If True, ensure complementary FOV coverage.
        fov_force_both_sides: If True, drop from both ends in FOV sim.
        obliqueness_range: Max rotation per axis in degrees for obliqueness.
        enable_obliqueness: If True, simulate oblique acquisitions.
        prob_obliqueness: Probability of applying obliqueness per stack.
        tight_fov: If True, size the LR scan FOV to the brain bbox so the
            FOV mask covers the air around the brain in HR space (mimics
            radiographer-sized FOV in real acquisitions).
        tight_fov_threshold: Intensity threshold for foreground detection.
        tight_fov_margin: Extra voxels around the brain bbox.
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
        # Artifact / corruption intensities
        bias_field_std: float = 0.3,
        noise_std: float = 0.02,
        motion_intensity: float = 0.5,
        spike_intensity: float = 0.04,
        # Geometry-only mode (disable all appearance corruptions)
        structural_only: bool = False,
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
        # Obliqueness
        obliqueness_range: float = 15.0,
        enable_obliqueness: bool = True,
        prob_obliqueness: float = 0.5,
        # Brain-tight FOV (LR scan FOV sized to brain bbox)
        tight_fov: bool = True,
        tight_fov_threshold: float = 1e-3,
        tight_fov_margin: int = 0,
    ):
        if atlas_res is None:
            atlas_res = [1.0, 1.0, 1.0]
        if target_res is None:
            target_res = [1.0, 1.0, 1.0]
        if min_resolution is None:
            min_resolution = [1.0, 1.0, 1.0]
        if max_res_aniso is None:
            max_res_aniso = [9.0, 9.0, 9.0]

        # Geometry-only mode: silence every appearance corruption so only the
        # structural transforms (downsampling + FOV) remain. Downsampling and
        # FOV stay under the control of their own flags (randomise_res, fov_*).
        self.structural_only = structural_only
        if structural_only:
            prob_motion = 0.0
            prob_spike = 0.0
            prob_aliasing = 0.0
            prob_bias_field = 0.0
            prob_noise = 0.0
            apply_intensity_aug = False

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

        # Resolution config (replaces SampleResolution nn.Module — Bug Fix #2).
        #
        # BUG FIX: this was built only when randomise_res=True, so with
        # randomise_res=False _create_orthogonal_resolutions fell through to
        # hardcoded [1,1,1]/[9,9,9] defaults and silently ignored the
        # configured range — i.e. exactly in the deterministic mode you would
        # use for controlled experiments, asking for fixed 3 mm slices got you
        # fixed 9 mm ones. The range is needed in both modes.
        self.res_config = ResolutionConfig(
            min_resolution=min_resolution,
            max_res_aniso=max_res_aniso,
        )

        # Bias field. prob=1.0 because gating is done per-patient in
        # generate_paired_data via self.prob_bias_field.
        self.bias = BiasFieldCorruption(
            bias_field_std=bias_field_std, bias_scale=0.025, prob=1.0
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
            noise_std=noise_std,
            motion_intensity=motion_intensity,
            spike_intensity=spike_intensity,
            upsample_mode=upsample_mode,
            return_intermediate=return_intermediate,
            psf_profile_type=psf_profile_type,
            psf_edge_width=psf_edge_width,
            obliqueness_range=obliqueness_range,
            enable_obliqueness=enable_obliqueness,
            prob_obliqueness=prob_obliqueness,
            tight_fov=tight_fov,
            tight_fov_threshold=tight_fov_threshold,
            tight_fov_margin=tight_fov_margin,
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
            bias_field_std=config.physics.bias_field_std,
            noise_std=config.artifacts.noise_std,
            motion_intensity=config.artifacts.motion_intensity,
            spike_intensity=config.artifacts.spike_intensity,
            structural_only=config.structural_only,
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
            return_intermediate=config.return_intermediate,
            psf_profile_type=config.physics.psf_type,
            psf_edge_width=config.physics.edge_width,
            fov_min_keep=config.fov.min_keep,
            fov_max_keep=config.fov.max_keep,
            fov_ensure_coverage=config.fov.ensure_coverage,
            fov_force_both_sides=config.fov.force_both_sides,
            obliqueness_range=config.fov.obliqueness_range,
            enable_obliqueness=config.fov.enable_obliqueness,
            prob_obliqueness=config.fov.prob_obliqueness,
            tight_fov=config.fov.tight_fov,
            tight_fov_threshold=config.fov.tight_fov_threshold,
            tight_fov_margin=config.fov.tight_fov_margin,
        )

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image to [0, 1] using percentile scaling.

        Percentiles are computed per batch item. ``ScaleIntensityRangePercentiles``
        is a single-sample transform, so handing it the whole (B, C, D, H, W)
        batch would pool intensities across subjects and make each volume's
        normalization depend on the others it happened to be batched with.
        """
        return torch.cat(
            [self.normalizer(image[b : b + 1]) for b in range(image.shape[0])],
            dim=0,
        )

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

    def _compute_fov_drop_decisions(
        self, batch_size: int, device: torch.device
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Pre-compute per-stack FOV drop decisions with coverage guarantees.

        Returns:
            Tuple of (drop_decisions, keep_fractions), each a list of
            num_stacks tensors of shape (batch_size,).
        """
        drop_decisions = []
        keep_fractions = []

        for _ in range(self.num_stacks):
            drop_decisions.append(torch.zeros(batch_size, dtype=torch.bool, device=device))
            keep_fractions.append(torch.ones(batch_size, dtype=torch.float32, device=device))

        if self.fov_augmentation_prob <= 0:
            return drop_decisions, keep_fractions

        for b in range(batch_size):
            batch_drops = []
            for s in range(self.num_stacks):
                should_drop = torch.rand(1).item() < self.fov_augmentation_prob
                batch_drops.append(should_drop)

            # No coverage handling here: forcing one stack to stay uncropped
            # (the old approach) both over-corrects — three centred slabs
            # usually already cover the head between them — and under-corrects,
            # since it never checks the head at all. The real guarantee lives in
            # HRLRDataGenerator._enforce_union_coverage, which measures coverage
            # against the actual anatomy.
            for s in range(self.num_stacks):
                drop_decisions[s][b] = batch_drops[s]
                if batch_drops[s]:
                    frac = (
                        torch.rand(1).item()
                        * (self.fov_max_keep - self.fov_min_keep)
                        + self.fov_min_keep
                    )
                    keep_fractions[s][b] = frac

        return drop_decisions, keep_fractions

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
        min_res = torch.tensor(self.res_config.min_resolution, device=device)
        max_res = torch.tensor(self.res_config.max_res_aniso, device=device)

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

    def sample_structural_plan(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        support_masks: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Sample one acquisition geometry, reusable across several volumes.

        Everything that defines *where and how* the stacks are sampled —
        per-stack resolution and slice thickness, FOV slice-drop decision, keep
        fraction and dropped end, and obliqueness tilt — is drawn once and
        returned as a plain dict. Feeding the same plan to
        :meth:`generate_paired_data` for several co-registered volumes (the T1,
        T2 and FLAIR of one subject) makes their stack ``i`` share one
        acquisition: same orientation, same thickness, same coverage, same tilt.

        Appearance corruptions (bias field, noise, gamma) are deliberately *not*
        part of the plan — real repeat acquisitions share a prescription, not a
        noise realisation. In ``structural_only`` mode none of them apply anyway.

        Args:
            batch_size: Number of volumes per call to ``generate_paired_data``.
            device: Device for the sampled tensors.
            support_masks: Optional (B, 1, D, H, W) tight-FOV support masks to
                pin into the plan. Without this each volume derives its own
                brain bbox from its own intensities, which differs between
                contrasts and desynchronises their FOV masks. See
                :func:`mri_synth.fov.resampling.compute_shared_support_mask`.

        Returns:
            Dict accepted by ``generate_paired_data(structural_plan=...)``.
        """
        device = device or torch.device("cpu")
        resolutions, thicknesses = self._create_orthogonal_resolutions(
            batch_size, device
        )
        fov_drop, fov_keep = self._compute_fov_drop_decisions(batch_size, device)

        drop_from_start = []
        obliqueness = []
        for _ in range(self.num_stacks):
            drop_from_start.append(
                torch.rand(batch_size, device=device) < 0.5
            )
            obliqueness.append(
                [
                    self.artifact_simulator.sample_obliqueness(device)
                    for _ in range(batch_size)
                ]
            )

        return {
            "resolutions": resolutions,
            "thicknesses": thicknesses,
            "fov_drop": fov_drop,
            "fov_keep_fraction": fov_keep,
            "fov_drop_from_start": drop_from_start,
            "obliqueness": obliqueness,
            "support_masks": support_masks,
        }

    def _enforce_union_coverage(
        self, plan: Dict, hr_images: torch.Tensor
    ) -> None:
        """Widen FOV slabs until the stacks jointly observe the whole head.

        ``fov.ensure_coverage`` promises that the *union* of the stacks contains
        the subject's entire head. Individual stacks may still miss most of it;
        what must not happen is a head voxel that no stack observed, since the
        HR target would then contain anatomy absent from every input.

        Predicts each stack's observed region from the sampled geometry and
        grows the keep fraction of whichever stack recovers the most unseen head
        per step, so cropping is relaxed only where it actually loses anatomy.
        The plan is updated in place, so contrasts sharing a plan stay matched.
        """
        if not self.fov_ensure_coverage or self.fov_augmentation_prob <= 0:
            return

        from mri_synth.fov.coverage import coarse_foreground, ensure_union_coverage

        sim = self.artifact_simulator
        hr_shape = tuple(hr_images.shape[2:])
        hr_spacing = [float(v) for v in sim.volume_res.tolist()]
        shared_support = plan.get("support_masks")

        for b in range(hr_images.shape[0]):
            # With a shared prescription (multi-contrast), judge coverage
            # against that one head definition so every contrast repairs
            # identically instead of drifting on its own intensities.
            source = (
                shared_support[b]
                if shared_support is not None
                else hr_images[b]
            )
            threshold = 0.5 if shared_support is not None else sim.tight_fov_threshold
            foreground, stride = coarse_foreground(source, threshold=threshold)

            geometries, keeps, flags, starts = [], [], [], []
            for s in range(self.num_stacks):
                dropped = bool(plan["fov_drop"][s][b].item())
                geometries.append(
                    sim.resolve_stack_geometry(
                        hr_shape,
                        plan["resolutions"][s][b],
                        plan["obliqueness"][s][b],
                    )
                )
                flags.append(dropped)
                keeps.append(
                    float(plan["fov_keep_fraction"][s][b].item()) if dropped else 1.0
                )
                starts.append(bool(plan["fov_drop_from_start"][s][b].item()))

            new_keeps, new_flags, _ = ensure_union_coverage(
                foreground=foreground,
                stride=stride,
                geometries=geometries,
                keep_fractions=keeps,
                drop_flags=flags,
                drop_from_start=starts,
                force_both_sides=self.fov_force_both_sides,
                hr_shape=hr_shape,
                hr_spacing=hr_spacing,
                device=hr_images.device,
            )

            for s in range(self.num_stacks):
                plan["fov_keep_fraction"][s][b] = new_keeps[s]
                plan["fov_drop"][s][b] = new_flags[s]

    def generate_paired_data(
        self,
        hr_images: torch.Tensor,
        return_resolution: bool = False,
        sample_info: Optional[Dict] = None,
        structural_plan: Optional[Dict] = None,
    ):
        """
        Generate paired LR-HR training data with N orthogonal LR stacks.

        Args:
            hr_images: High-resolution input images (B, C, D, H, W) in RAS.
            return_resolution: If True, also return resolution and thickness.
            sample_info: Optional metadata dict.
            structural_plan: Optional acquisition geometry from
                :meth:`sample_structural_plan`. Pass the same plan for several
                co-registered volumes to give them identical stack geometry.
                When omitted, a fresh geometry is sampled inline.

        Returns:
            Variable-length tuple depending on return_resolution and
            return_intermediate flags. FOV masks (single list) replace the
            old spatial_masks and interpolation_masks.
        """
        batch_size = hr_images.shape[0]
        device = hr_images.device

        # STEP 1: Normalize HR
        hr_augmented = self._normalize_image(hr_images)

        # STEP 2: Resolve the acquisition geometry. A caller-supplied plan pins
        # it so co-registered volumes (contrasts of one subject) come out with
        # matching stacks; otherwise a fresh one is drawn here.
        if structural_plan is None:
            structural_plan = self.sample_structural_plan(batch_size, device)

        # Grow FOV slabs where the sampled cropping would leave head unseen by
        # every stack. Mutates the plan in place, so contrasts sharing a plan
        # inherit the same repaired geometry.
        self._enforce_union_coverage(structural_plan, hr_augmented)

        resolutions = structural_plan["resolutions"]
        thicknesses = structural_plan["thicknesses"]

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

        fov_drop_decisions = structural_plan["fov_drop"]
        fov_keep_fractions = structural_plan["fov_keep_fraction"]

        # Record the sampled decisions so callers (e.g. the CLI) can log which
        # corruptions were actually applied. Per-patient flags are shared
        # across stacks; FOV drop is per-stack.
        self._last_decisions = {
            "bias_field": apply_bias_field,
            "intensity_aug": self.apply_intensity_aug,
            "motion": apply_motion,
            "spike": apply_spike,
            "aliasing": apply_aliasing,
            "noise": apply_noise,
            "motion_axis": motion_axis,
            "aliasing_axis": aliasing_axis,
            "fov_drop": fov_drop_decisions,
            "fov_keep_fraction": fov_keep_fractions,
        }
        self._last_structural_plan = structural_plan

        lr_stacks = []
        true_lr_stacks = []
        fov_masks = []
        self._rotation_angles_per_stack = []
        # Per-stack, per-batch-item LR-voxel -> HR-voxel 4x4 matrices. Callers
        # that save native-resolution stacks need these to place them in world
        # space; deriving the affine from the requested resolution alone is not
        # enough (the k-space crop rounds the spacing, and the stack is
        # corner-aligned to the HR grid and may be rotated).
        self._lr_to_hr_voxel_per_stack = []

        for stack_idx in range(self.num_stacks):
            lr_images = hr_degraded.clone()

            # Physics simulation
            resolution = resolutions[stack_idx]
            thickness = thicknesses[stack_idx]

            sim_out = self.artifact_simulator(
                lr_images,
                resolution,
                thickness,
                enable_motion=apply_motion,
                enable_spike=apply_spike,
                enable_aliasing=apply_aliasing,
                enable_noise=apply_noise,
                motion_axis=motion_axis,
                aliasing_axis=aliasing_axis,
                fov_drop_decisions=fov_drop_decisions[stack_idx],
                fov_keep_fractions=fov_keep_fractions[stack_idx],
                fov_force_both_sides=self.fov_force_both_sides,
                fov_drop_from_start=structural_plan["fov_drop_from_start"][stack_idx],
                obliqueness_angles=structural_plan["obliqueness"][stack_idx],
                support_masks=structural_plan.get("support_masks"),
            )
            if self.return_intermediate:
                lr_images, stack_fov_masks, true_lr_images = sim_out
            else:
                lr_images, stack_fov_masks = sim_out

            self._rotation_angles_per_stack.append(
                self.artifact_simulator._last_rotation_angles
            )
            self._lr_to_hr_voxel_per_stack.append(
                self.artifact_simulator._last_lr_to_hr_voxel_matrices
            )

            # Normalize LR to [0, 1] — match HR's simple clamp
            if self.clip_to_unit_range:
                lr_images = torch.clamp(lr_images, 0.0, 1.0)
                if self.return_intermediate:
                    true_lr_images = torch.clamp(true_lr_images, 0.0, 1.0)

            lr_stacks.append(lr_images)
            fov_masks.append(stack_fov_masks)
            if self.return_intermediate:
                true_lr_stacks.append(true_lr_images)

        hr_augmented = torch.clamp(hr_augmented, 0.0, 1.0)

        orientation_mask = self._create_orientation_mask(batch_size, device)

        if return_resolution and self.return_intermediate:
            return (
                lr_stacks,
                true_lr_stacks,
                hr_augmented,
                resolutions,
                thicknesses,
                orientation_mask,
                fov_masks,
            )
        elif return_resolution:
            return (
                lr_stacks,
                hr_augmented,
                resolutions,
                thicknesses,
                orientation_mask,
                fov_masks,
            )
        else:
            return lr_stacks, hr_augmented, orientation_mask, fov_masks

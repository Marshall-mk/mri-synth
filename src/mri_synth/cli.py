"""Typer CLI for MRI synthesis."""

from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(
    name="mri-synth",
    help="Physics-based MRI simulation: generate synthetic LR stacks from HR volumes.",
)


def _collect_inputs(input_path: Path) -> List[Path]:
    """Collect NIfTI files from a path (file or directory)."""
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        niftis = sorted(
            list(input_path.glob("*.nii.gz")) + list(input_path.glob("*.nii"))
        )
        if not niftis:
            typer.echo(f"No NIfTI files found in {input_path}", err=True)
            raise typer.Exit(1)
        return niftis
    else:
        typer.echo(f"Input path does not exist: {input_path}", err=True)
        raise typer.Exit(1)


def _collect_contrast_groups(input_path: Path) -> List[tuple]:
    """Group input NIfTIs into ``(subject_name, [contrast files])`` pairs.

    Two layouts are accepted:

    * ``input/sub-01/{T1,T2}.nii.gz`` — one group per subject subdirectory.
    * ``input/{T1,T2}.nii.gz`` — a single subject whose contrasts sit directly
      in the given directory.
    """
    if input_path.is_file():
        typer.echo(
            "--multi-contrast expects a directory of volumes, not a single file.",
            err=True,
        )
        raise typer.Exit(1)
    if not input_path.is_dir():
        typer.echo(f"Input path does not exist: {input_path}", err=True)
        raise typer.Exit(1)

    def _niftis(d: Path) -> List[Path]:
        return sorted(list(d.glob("*.nii.gz")) + list(d.glob("*.nii")))

    subject_dirs = sorted(d for d in input_path.iterdir() if d.is_dir())
    groups = [(d.name, _niftis(d)) for d in subject_dirs]
    groups = [(name, files) for name, files in groups if files]

    if not groups:
        files = _niftis(input_path)
        if not files:
            typer.echo(f"No NIfTI files found in {input_path}", err=True)
            raise typer.Exit(1)
        groups = [(input_path.name, files)]

    return groups


def _contrast_name(path: Path) -> str:
    """Strip .nii/.nii.gz to get a contrast label."""
    return path.name[: -len(".nii.gz")] if path.name.endswith(".nii.gz") else path.stem


def _applied_artifacts_meta(decisions: dict) -> dict:
    """Summarise which corruptions were sampled for batch item 0."""
    return {
        "bias_field": bool(decisions["bias_field"][0].item()),
        "intensity_aug": bool(decisions["intensity_aug"]),
        "motion": bool(decisions["motion"][0].item()),
        "spike": bool(decisions["spike"][0].item()),
        "aliasing": bool(decisions["aliasing"][0].item()),
        "noise": bool(decisions["noise"][0].item()),
    }


def _coverage_meta(generator) -> dict:
    """Fraction of head observed by NO stack, after the coverage repair.

    0.0 means the union-coverage guarantee held. A non-zero value means the head
    could not be covered at this obliqueness even with cropping fully relaxed --
    the ground truth then contains anatomy no input saw, and a reconstruction is
    being scored on invented structure. Recorded so a consumer can filter such
    subjects instead of discovering it from a warning that scrolled past.
    """
    unc = getattr(generator, "_coverage_uncovered", None)
    if not unc:
        return {}
    return {"uncovered_head_fraction": round(float(unc.get(0, 0.0)), 6)}


def _save_stacks(
    out_dir: Path,
    generator,
    cfg,
    lr_stacks,
    fov_masks,
    true_lr_stacks,
    resolutions,
    thicknesses,
    affine,
) -> List[dict]:
    """Write every stack (plus FOV mask and optional native-res volume).

    Returns the per-stack metadata entries.
    """
    import math

    from mri_synth.fov.resampling import (
        crop_native_to_kept_slab,
        shift_affine_origin,
    )
    from mri_synth.io import (
        get_fov_mask_filename,
        get_native_stack_filename,
        get_stack_filename,
        save_volume,
    )

    decisions = generator._last_decisions
    stack_metas = []

    for s_idx in range(cfg.num_stacks):
        stack_file = get_stack_filename(s_idx)
        save_volume(lr_stacks[s_idx].squeeze(0), out_dir / stack_file, affine)

        stack_meta = {
            "file": stack_file,
            "resolution": resolutions[s_idx].squeeze(0).tolist(),
            "thickness": thicknesses[s_idx].squeeze(0).tolist(),
        }

        fov_dropped = bool(decisions["fov_drop"][s_idx][0].item())
        stack_meta["fov_dropped"] = fov_dropped
        stack_meta["fov_keep_fraction"] = (
            round(decisions["fov_keep_fraction"][s_idx][0].item(), 4)
            if fov_dropped
            else None
        )

        rot = generator._rotation_angles_per_stack[s_idx][0]
        stack_meta["obliqueness_deg"] = (
            [round(math.degrees(a), 3) for a in rot] if rot is not None else None
        )

        fov_mask_file = get_fov_mask_filename(s_idx)
        save_volume(fov_masks[s_idx].squeeze(0), out_dir / fov_mask_file, affine)
        stack_meta["fov_mask_file"] = fov_mask_file

        if cfg.save_native_res and generator.return_intermediate:
            native_file = get_native_stack_filename(s_idx)
            # Compose the HR world affine with the exact LR-voxel -> HR-voxel
            # matrix the simulator used, so the native stack lands in world
            # space precisely where its data was sampled from. Rebuilding it
            # from the requested resolution instead would drop the grid origin,
            # ignore the rounding of the k-space crop, and rotate about the
            # world origin rather than the FOV centre.
            lr_to_hr = (
                generator._lr_to_hr_voxel_per_stack[s_idx][0].detach().cpu().numpy()
            )
            native_affine = affine @ lr_to_hr
            native_vol = true_lr_stacks[s_idx].squeeze(0)

            # CROP the unacquired slices out instead of shipping them as zeros.
            # apply_fov_slice_drop_native zeroes them, which the HR path needs (the
            # FOV mask is derived from those zeros), but a native stack on disk is an
            # ACQUISITION: a shorter slab means fewer slices, not slices of zeros. A
            # consumer cannot tell fabricated zeros from measured background and
            # will fit them as data.
            axis, lo, hi = generator._kept_slab_per_stack[s_idx][0]
            if (lo, hi) != (0, native_vol.shape[axis + 1]):
                sl = [slice(None)] * native_vol.ndim
                sl[axis + 1] = slice(lo, hi)
                native_vol = native_vol[tuple(sl)].contiguous()
                native_affine = shift_affine_origin(native_affine, axis, lo)
            # What was ACTUALLY kept. fov_keep_fraction is the requested (post-repair)
            # value; kept_slice_bounds rounds it to whole slices, so the two differ by
            # up to half a slice. Consumers should trust these fields, not the fraction.
            n_native = int(true_lr_stacks[s_idx].squeeze(0).shape[axis + 1])
            stack_meta["native_kept_slices"] = [int(lo), int(hi)]
            stack_meta["native_total_slices"] = n_native
            stack_meta["native_through_plane_axis"] = int(axis)
            stack_meta["fov_keep_fraction_realised"] = round((hi - lo) / n_native, 4)

            save_volume(native_vol, out_dir / native_file, native_affine)
            stack_meta["native_file"] = native_file

        stack_metas.append(stack_meta)

    return stack_metas


def _generate_multi_contrast(input_path, output_dir, cfg, generator, manifest, torch):
    """Simulate every contrast of each subject under one shared geometry.

    All contrasts of a subject are handed the same
    :meth:`HRLRDataGenerator.sample_structural_plan` result, so stack ``i`` has
    the same orientation, slice thickness, FOV coverage fraction and oblique
    tilt in every contrast — the T1 and T2 sagittal stacks describe the same
    physical acquisition. The tight-FOV support is likewise computed once from
    all contrasts together, since a bbox derived per contrast would differ.

    Output layout::

        output/<subject>/hr_<contrast>.nii.gz
        output/<subject>/variation_000/<contrast>/stack_0_axial.nii.gz
        output/<subject>/variation_000/metadata.json
    """
    from mri_synth.fov.resampling import compute_shared_support_mask
    from mri_synth.io import load_volume, resample_to_spacing, save_volume, write_metadata

    groups = _collect_contrast_groups(input_path)
    typer.echo(
        f"Processing {len(groups)} subject(s) in multi-contrast mode, "
        f"{cfg.num_variations} variation(s) each, {cfg.num_stacks} stacks each."
    )

    for grp_idx, (subject, files) in enumerate(groups):
        names = [_contrast_name(f) for f in files]
        typer.echo(
            f"  [{grp_idx + 1}/{len(groups)}] {subject}: {len(files)} contrast(s) "
            f"({', '.join(names)})"
        )

        tensors, affines = [], []
        for f in files:
            t, a, _ = load_volume(f)
            t, a = resample_to_spacing(t, a, cfg.atlas_res)
            tensors.append(t.unsqueeze(0).to(cfg.device))
            affines.append(a)

        # A shared geometry is only meaningful on a shared grid.
        shapes = {tuple(t.shape[2:]) for t in tensors}
        if len(shapes) > 1:
            typer.echo(
                f"    Skipping {subject}: contrasts must be co-registered onto one "
                f"grid to share an acquisition geometry, but got shapes {shapes}. "
                f"Register them (or resample to a common grid) first.",
                err=True,
            )
            continue

        affine = affines[0]
        subject_dir = output_dir / subject
        subject_dir.mkdir(parents=True, exist_ok=True)

        # One FOV prescription for the whole session, from every contrast's
        # foreground, rather than a per-contrast bbox that would desynchronise
        # the FOV masks.
        support = None
        if cfg.fov.tight_fov:
            support = compute_shared_support_mask(
                [t[0] for t in tensors],
                threshold=cfg.fov.tight_fov_threshold,
                margin=cfg.fov.tight_fov_margin,
            ).unsqueeze(0)

        subj_manifest = {"name": subject, "contrasts": names, "variations": []}

        for var_idx in range(cfg.num_variations):
            var_dir = subject_dir / f"variation_{var_idx:03d}"
            var_dir.mkdir(parents=True, exist_ok=True)

            plan = generator.sample_structural_plan(
                batch_size=1,
                device=torch.device(cfg.device),
                support_masks=support,
            )

            var_meta = {
                "structural_only": cfg.structural_only,
                "shared_geometry": True,
                "contrasts": {},
            }

            for name, tensor in zip(names, tensors):
                result = generator.generate_paired_data(
                    tensor, return_resolution=True, structural_plan=plan
                )
                if generator.return_intermediate:
                    (lr_stacks, true_lr_stacks, hr_aug, resolutions,
                     thicknesses, _, fov_masks) = result
                else:
                    (lr_stacks, hr_aug, resolutions,
                     thicknesses, _, fov_masks) = result
                    true_lr_stacks = None

                if var_idx == 0:
                    save_volume(
                        hr_aug.squeeze(0), subject_dir / f"hr_{name}.nii.gz", affine
                    )

                contrast_dir = var_dir / name
                contrast_dir.mkdir(parents=True, exist_ok=True)

                var_meta["contrasts"][name] = {
                    "applied_artifacts": _applied_artifacts_meta(
                        generator._last_decisions
                    ),
                    **_coverage_meta(generator),
                    "stacks": _save_stacks(
                        contrast_dir, generator, cfg, lr_stacks, fov_masks,
                        true_lr_stacks, resolutions, thicknesses, affine,
                    ),
                }

            write_metadata(var_dir / "metadata.json", var_meta)
            subj_manifest["variations"].append(str(var_dir))

        manifest["volumes"].append(subj_manifest)


@app.command()
def generate(
    input: Path = typer.Option(..., "--input", "-i", help="Input NIfTI file or directory"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory"),
    num_stacks: int = typer.Option(3, "--num-stacks", help="Number of LR stacks per variation"),
    num_variations: int = typer.Option(1, "--num-variations", "-n", help="Number of variations per volume"),
    psf_type: str = typer.Option("trapezoid", "--psf-type", help="PSF profile type: boxcar, gaussian, trapezoid"),
    enable_bias_field: bool = typer.Option(True, "--enable-bias-field/--no-bias-field", help="Enable bias field corruption"),
    fov_enable: bool = typer.Option(True, "--fov-enable/--no-fov-enable", help="Enable FOV simulation"),
    noise_std: float = typer.Option(0.02, "--noise-std", help="Noise standard deviation"),
    min_res: Optional[List[float]] = typer.Option(None, "--min-res", help="Minimum resolution per axis (3 values)"),
    max_res_aniso: Optional[List[float]] = typer.Option(None, "--max-res-aniso", help="Maximum anisotropic resolution (3 values)"),
    device: str = typer.Option("cpu", "--device", help="Device: cpu or cuda"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed"),
    save_native_res: bool = typer.Option(False, "--save-native-res/--no-save-native-res", help="Save native-resolution LR stacks (pre-upsample)"),
    clip_to_unit_range: bool = typer.Option(True, "--clip-to-unit-range/--no-clip-to-unit-range", help="Clip outputs to [0, 1]"),
    apply_intensity_aug: bool = typer.Option(False, "--apply-intensity-aug/--no-intensity-aug", help="Apply intensity augmentation"),
    structural_only: bool = typer.Option(False, "--structural-only/--no-structural-only", help="Geometry-only mode: disable bias/noise/intensity/motion/spike/aliasing, keep only downsampling + FOV"),
    randomise_res: bool = typer.Option(True, "--randomise-res/--no-randomise-res", help="Randomize acquisition resolution"),
    return_intermediate: bool = typer.Option(False, "--return-intermediate/--no-return-intermediate", help="Return native-resolution LR (pre-upsample)"),
    upsample_mode: str = typer.Option("trilinear", "--upsample-mode", help="Interpolation mode for upsampling"),
    obliqueness_range: float = typer.Option(15.0, "--obliqueness-range", help="Max obliqueness rotation per axis in degrees"),
    enable_obliqueness: bool = typer.Option(True, "--enable-obliqueness/--no-obliqueness", help="Enable oblique acquisition simulation"),
    prob_obliqueness: float = typer.Option(0.5, "--prob-obliqueness", help="Probability of applying obliqueness per stack"),
    tight_fov: bool = typer.Option(True, "--tight-fov/--no-tight-fov", help="Size LR scan FOV to brain bbox (mimics radiographer-sized FOV)"),
    tight_fov_threshold: float = typer.Option(1e-3, "--tight-fov-threshold", help="Foreground intensity threshold for brain bbox detection"),
    tight_fov_margin: int = typer.Option(0, "--tight-fov-margin", help="Voxel margin around the brain bbox"),
    multi_contrast: bool = typer.Option(False, "--multi-contrast/--no-multi-contrast", help="Treat the input as groups of co-registered contrasts per subject; all contrasts of a subject share one acquisition geometry"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file (overrides CLI flags)"),
):
    """Generate synthetic LR MRI stacks from HR volumes."""
    import torch

    from mri_synth.config import GenerationConfig
    from mri_synth.io import (
        create_output_structure,
        load_volume,
        resample_to_spacing,
        save_volume,
        write_manifest,
        write_metadata,
    )
    from mri_synth.pipeline import HRLRDataGenerator

    # Build config
    if config is not None:
        cfg = GenerationConfig.from_yaml(str(config))
    else:
        cfg = GenerationConfig(
            num_stacks=num_stacks,
            num_variations=num_variations,
            min_resolution=min_res or [1.0, 1.0, 1.0],
            max_res_aniso=max_res_aniso or [9.0, 9.0, 9.0],
            device=device,
            seed=seed,
            clip_to_unit_range=clip_to_unit_range,
            apply_intensity_aug=apply_intensity_aug,
            structural_only=structural_only,
            randomise_res=randomise_res,
            return_intermediate=return_intermediate,
            upsample_mode=upsample_mode,
        )
        cfg.physics.psf_type = psf_type
        cfg.physics.prob_bias_field = 0.5 if enable_bias_field else 0.0
        cfg.artifacts.noise_std = noise_std
        cfg.fov.enable = fov_enable
        cfg.fov.obliqueness_range = obliqueness_range
        cfg.fov.enable_obliqueness = enable_obliqueness
        cfg.fov.prob_obliqueness = prob_obliqueness
        cfg.fov.tight_fov = tight_fov
        cfg.fov.tight_fov_threshold = tight_fov_threshold
        cfg.fov.tight_fov_margin = tight_fov_margin
        cfg.save_native_res = save_native_res

    # --save-native-res requires return_intermediate to generate true LR stacks
    if cfg.save_native_res and not cfg.return_intermediate:
        cfg.return_intermediate = True

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    generator = HRLRDataGenerator.from_config(cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"volumes": [], "config": cfg.model_dump(), "multi_contrast": multi_contrast}

    if multi_contrast:
        _generate_multi_contrast(
            input, output_dir, cfg, generator, manifest, torch
        )
        write_manifest(output_dir, manifest)
        typer.echo(f"Done. Output written to {output_dir}")
        return

    input_files = _collect_inputs(input)

    typer.echo(f"Processing {len(input_files)} volume(s), {cfg.num_variations} variation(s) each, {cfg.num_stacks} stacks each.")

    for vol_idx, vol_path in enumerate(input_files):
        vol_name = vol_path.stem.replace(".nii", "")
        typer.echo(f"  [{vol_idx + 1}/{len(input_files)}] {vol_name}")

        tensor, affine, header = load_volume(vol_path)
        tensor, affine = resample_to_spacing(tensor, affine, cfg.atlas_res)
        tensor = tensor.unsqueeze(0).to(cfg.device)  # (1, C, D, H, W)

        dirs = create_output_structure(output_dir, vol_name, cfg.num_variations)

        vol_manifest = {"name": vol_name, "variations": []}

        for var_idx in range(cfg.num_variations):
            var_dir = dirs["variation_dirs"][var_idx]

            result = generator.generate_paired_data(
                tensor, return_resolution=True
            )

            if generator.return_intermediate:
                lr_stacks, true_lr_stacks, hr_aug, resolutions, thicknesses, orient_mask, fov_masks = result
            else:
                lr_stacks, hr_aug, resolutions, thicknesses, orient_mask, fov_masks = result

            # Save normalized HR (once — identical across variations)
            if var_idx == 0:
                save_volume(hr_aug.squeeze(0), dirs["volume_dir"] / "hr.nii.gz", affine)

            # Record which corruptions were actually applied for this variation.
            # Per-patient flags are shared across stacks (batch index 0 here).
            var_meta = {
                "structural_only": cfg.structural_only,
                "applied_artifacts": _applied_artifacts_meta(
                    generator._last_decisions
                ),
                **_coverage_meta(generator),
                "stacks": _save_stacks(
                    var_dir, generator, cfg, lr_stacks, fov_masks,
                    true_lr_stacks if generator.return_intermediate else None,
                    resolutions, thicknesses, affine,
                ),
            }

            write_metadata(var_dir / "metadata.json", var_meta)
            vol_manifest["variations"].append(str(var_dir))

        manifest["volumes"].append(vol_manifest)

    write_manifest(output_dir, manifest)
    typer.echo(f"Done. Output written to {output_dir}")


if __name__ == "__main__":
    app()

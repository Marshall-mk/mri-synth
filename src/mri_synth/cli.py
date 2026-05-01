"""Typer CLI for MRI synthesis."""

import sys
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


@app.command()
def generate(
    input: Path = typer.Option(..., "--input", "-i", help="Input NIfTI file or directory"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory"),
    num_stacks: int = typer.Option(3, "--num-stacks", help="Number of LR stacks per variation"),
    num_variations: int = typer.Option(1, "--num-variations", "-n", help="Number of variations per volume"),
    psf_type: str = typer.Option("trapezoid", "--psf-type", help="PSF profile type: boxcar, gaussian, trapezoid"),
    enable_bias_field: bool = typer.Option(True, "--enable-bias-field/--no-bias-field", help="Enable bias field corruption"),
    enable_fov_sim: bool = typer.Option(True, "--enable-fov-sim/--no-fov-sim", help="Enable FOV simulation"),
    noise_std: float = typer.Option(0.02, "--noise-std", help="Noise standard deviation"),
    min_res: Optional[List[float]] = typer.Option(None, "--min-res", help="Minimum resolution per axis (3 values)"),
    max_res_aniso: Optional[List[float]] = typer.Option(None, "--max-res-aniso", help="Maximum anisotropic resolution (3 values)"),
    device: str = typer.Option("cpu", "--device", help="Device: cpu or cuda"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed"),
    save_native_res: bool = typer.Option(False, "--save-native-res/--no-save-native-res", help="Save native-resolution LR stacks (pre-upsample)"),
    clip_to_unit_range: bool = typer.Option(True, "--clip-to-unit-range/--no-clip-to-unit-range", help="Clip outputs to [0, 1]"),
    apply_intensity_aug: bool = typer.Option(False, "--apply-intensity-aug/--no-intensity-aug", help="Apply intensity augmentation"),
    randomise_res: bool = typer.Option(True, "--randomise-res/--no-randomise-res", help="Randomize acquisition resolution"),
    return_intermediate: bool = typer.Option(False, "--return-intermediate/--no-return-intermediate", help="Return native-resolution LR (pre-upsample)"),
    upsample_mode: str = typer.Option("trilinear", "--upsample-mode", help="Interpolation mode for upsampling"),
    obliqueness_range: float = typer.Option(15.0, "--obliqueness-range", help="Max obliqueness rotation per axis in degrees"),
    enable_obliqueness: bool = typer.Option(True, "--enable-obliqueness/--no-obliqueness", help="Enable oblique acquisition simulation"),
    prob_obliqueness: float = typer.Option(0.5, "--prob-obliqueness", help="Probability of applying obliqueness per stack"),
    tight_fov: bool = typer.Option(True, "--tight-fov/--no-tight-fov", help="Size LR scan FOV to brain bbox (mimics radiographer-sized FOV)"),
    tight_fov_threshold: float = typer.Option(1e-3, "--tight-fov-threshold", help="Foreground intensity threshold for brain bbox detection"),
    tight_fov_margin: int = typer.Option(0, "--tight-fov-margin", help="Voxel margin around the brain bbox"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file (overrides CLI flags)"),
):
    """Generate synthetic LR MRI stacks from HR volumes."""
    import torch

    from mri_synth.config import GenerationConfig
    from mri_synth.fov.resampling import euler_to_rotation_matrix
    from mri_synth.io import (
        create_output_structure,
        get_fov_mask_filename,
        get_native_stack_filename,
        get_stack_filename,
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
            randomise_res=randomise_res,
            return_intermediate=return_intermediate,
            upsample_mode=upsample_mode,
        )
        cfg.physics.psf_type = psf_type
        cfg.physics.prob_bias_field = 0.5 if enable_bias_field else 0.0
        cfg.artifacts.noise_std = noise_std
        cfg.fov.enable = enable_fov_sim
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
    input_files = _collect_inputs(input)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"volumes": [], "config": cfg.model_dump()}

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

            var_meta = {"stacks": []}
            for s_idx in range(cfg.num_stacks):
                stack_file = get_stack_filename(s_idx)
                save_volume(
                    lr_stacks[s_idx].squeeze(0),
                    var_dir / stack_file,
                    affine,
                )

                stack_meta = {
                    "file": stack_file,
                    "resolution": resolutions[s_idx].squeeze(0).tolist(),
                    "thickness": thicknesses[s_idx].squeeze(0).tolist(),
                }

                fov_mask_file = get_fov_mask_filename(s_idx)
                save_volume(
                    fov_masks[s_idx].squeeze(0),
                    var_dir / fov_mask_file,
                    affine,
                )
                stack_meta["fov_mask_file"] = fov_mask_file

                if cfg.save_native_res and generator.return_intermediate:
                    native_file = get_native_stack_filename(s_idx)
                    # Scale affine to reflect actual LR voxel spacing
                    native_affine = affine.copy()
                    res = resolutions[s_idx].squeeze(0)  # (3,) acquisition res in mm
                    for axis in range(3):
                        scale = res[axis].item() / cfg.atlas_res[axis]
                        native_affine[:3, axis] *= scale
                    # Apply obliqueness rotation to native affine
                    rot_angles = generator._rotation_angles_per_stack[s_idx][0]
                    if rot_angles is not None:
                        rx, ry, rz = rot_angles
                        R = euler_to_rotation_matrix(rx, ry, rz).numpy()
                        native_affine[:3, :3] = R @ native_affine[:3, :3]
                    save_volume(
                        true_lr_stacks[s_idx].squeeze(0),
                        var_dir / native_file,
                        native_affine,
                    )
                    stack_meta["native_file"] = native_file

                var_meta["stacks"].append(stack_meta)

            write_metadata(var_dir / "metadata.json", var_meta)
            vol_manifest["variations"].append(str(var_dir))

        manifest["volumes"].append(vol_manifest)

    write_manifest(output_dir, manifest)
    typer.echo(f"Done. Output written to {output_dir}")


if __name__ == "__main__":
    app()

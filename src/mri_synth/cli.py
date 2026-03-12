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
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file (overrides CLI flags)"),
):
    """Generate synthetic LR MRI stacks from HR volumes."""
    import torch

    from mri_synth.config import GenerationConfig
    from mri_synth.io import (
        create_output_structure,
        get_native_stack_filename,
        get_stack_filename,
        load_volume,
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
        )
        cfg.physics.psf_type = psf_type
        cfg.physics.prob_bias_field = 0.5 if enable_bias_field else 0.0
        cfg.artifacts.noise_std = noise_std
        cfg.fov.enable = enable_fov_sim
        cfg.save_native_res = save_native_res

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
        tensor = tensor.unsqueeze(0).to(cfg.device)  # (1, C, D, H, W)

        dirs = create_output_structure(output_dir, vol_name, cfg.num_variations)

        # Save HR
        save_volume(tensor.squeeze(0), dirs["volume_dir"] / "hr.nii.gz", affine)

        vol_manifest = {"name": vol_name, "variations": []}

        for var_idx in range(cfg.num_variations):
            var_dir = dirs["variation_dirs"][var_idx]

            result = generator.generate_paired_data(
                tensor, return_resolution=True
            )

            if generator.return_intermediate:
                lr_stacks, true_lr_stacks, hr_aug, resolutions, thicknesses, orient_mask, spatial_masks = result
            else:
                lr_stacks, hr_aug, resolutions, thicknesses, orient_mask, spatial_masks = result

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

                if cfg.save_native_res and generator.return_intermediate:
                    native_file = get_native_stack_filename(s_idx)
                    save_volume(
                        true_lr_stacks[s_idx].squeeze(0),
                        var_dir / native_file,
                        affine,
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

"""NIfTI I/O and output directory management."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch


def load_volume(path: Union[str, Path]) -> Tuple[torch.Tensor, np.ndarray, nib.Nifti1Header]:
    """
    Load a NIfTI volume.

    Args:
        path: Path to a .nii or .nii.gz file.

    Returns:
        Tuple of (tensor, affine, header) where tensor is (C, D, H, W).
    """
    path = Path(path)
    nii = nib.load(str(path))
    data = np.asarray(nii.dataobj, dtype=np.float32)

    # Add channel dim if needed
    if data.ndim == 3:
        data = data[np.newaxis, ...]

    tensor = torch.from_numpy(data)
    return tensor, nii.affine, nii.header


def save_volume(
    tensor: torch.Tensor,
    path: Union[str, Path],
    affine: Optional[np.ndarray] = None,
) -> None:
    """
    Save a tensor as a NIfTI file.

    Args:
        tensor: Volume tensor. If 4D (C, D, H, W) with C=1, squeezes the
            channel dim. If 3D, saves directly.
        path: Output path (.nii or .nii.gz).
        affine: 4x4 affine matrix. Defaults to identity.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = tensor.detach().cpu().numpy()
    if data.ndim == 4 and data.shape[0] == 1:
        data = data[0]

    if affine is None:
        affine = np.eye(4)

    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, str(path))


def create_output_structure(
    output_dir: Union[str, Path],
    volume_name: str,
    num_variations: int,
) -> Dict[str, Path]:
    """
    Create the output directory tree for a volume.

    Structure::

        output_dir/
          volume_name/
            hr.nii.gz
            variation_000/
              stack_0_axial.nii.gz
              ...
              metadata.json

    Args:
        output_dir: Root output directory.
        volume_name: Name of the volume (used as subdirectory name).
        num_variations: Number of variation subdirectories to create.

    Returns:
        Dict mapping 'volume_dir' and 'variation_dirs' to paths.
    """
    output_dir = Path(output_dir)
    vol_dir = output_dir / volume_name
    vol_dir.mkdir(parents=True, exist_ok=True)

    variation_dirs = []
    for v in range(num_variations):
        var_dir = vol_dir / f"variation_{v:03d}"
        var_dir.mkdir(parents=True, exist_ok=True)
        variation_dirs.append(var_dir)

    return {
        "volume_dir": vol_dir,
        "variation_dirs": variation_dirs,
    }


_ORIENTATION_NAMES = ["axial", "coronal", "sagittal"]


def get_stack_filename(stack_idx: int) -> str:
    """Get the filename for a stack by index."""
    orient = _ORIENTATION_NAMES[stack_idx % len(_ORIENTATION_NAMES)]
    return f"stack_{stack_idx}_{orient}.nii.gz"


def get_native_stack_filename(stack_idx: int) -> str:
    """Get the filename for a native-resolution stack by index."""
    orient = _ORIENTATION_NAMES[stack_idx % len(_ORIENTATION_NAMES)]
    return f"stack_{stack_idx}_{orient}_native.nii.gz"


def write_metadata(
    path: Union[str, Path],
    metadata: Dict,
) -> None:
    """Write metadata dict as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def write_manifest(
    output_dir: Union[str, Path],
    manifest: Dict,
) -> None:
    """Write manifest.json to the output directory."""
    write_metadata(Path(output_dir) / "manifest.json", manifest)

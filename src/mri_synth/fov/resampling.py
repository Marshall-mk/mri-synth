"""Affine-based 3D resampling and FOV mask generation.

Provides the "dummy mask trick": resample an all-ones volume from LR native
space to the HR target grid. Voxels that fall outside the LR FOV become 0,
producing a binary mask of valid data regions.
"""

import math
import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from monai.transforms.utils import generate_spatial_bounding_box


def euler_to_rotation_matrix(
    rx: float, ry: float, rz: float, device: torch.device = None
) -> torch.Tensor:
    """Convert Euler angles (radians) to a 3x3 rotation matrix.

    Uses extrinsic Rz @ Ry @ Rx convention.

    Args:
        rx: Rotation around x-axis in radians.
        ry: Rotation around y-axis in radians.
        rz: Rotation around z-axis in radians.
        device: Torch device for the output.

    Returns:
        3x3 rotation matrix.
    """
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    # fmt: off
    R = torch.tensor([
        [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
        [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
        [-sy,     sx * cy,                cx * cy               ],
    ], dtype=torch.float32, device=device)
    # fmt: on
    return R


def build_lr_affine(
    hr_affine: torch.Tensor,
    through_plane_axis: int,
    lr_spacing_tp: float,
    hr_spacing_tp: float,
    lr_shape: Tuple[int, int, int],
    hr_shape: Tuple[int, int, int],
    rotation_angles: Optional[Tuple[float, float, float]] = None,
) -> torch.Tensor:
    """Build the LR native-space affine matrix.

    Starts from the HR affine, scales the through-plane column to reflect the
    LR voxel spacing, anchors the LR grid onto the HR grid, and optionally
    applies a rotation (obliqueness) about the LR FOV centre.

    **Grid placement.** The LR and HR FOV centres are made to coincide, which
    matches the centred sampling grid that
    ``MRIArtifactSimulator._fft_downsample`` produces.

    ``lr_spacing_tp`` must be the *effective* spacing actually realised by the
    k-space crop, ``hr_spacing_tp * hr_shape[tp] / lr_shape[tp]``, not the
    nominal resolution that was requested. The two differ because the crop
    rounds to a whole number of samples (a requested 4.7 mm over 128 slices
    yields 27 slices, i.e. 4.741 mm); encoding the nominal value instead
    stretches the stack against the HR grid. A mismatch is reported as a
    warning.

    Args:
        hr_affine: 4x4 HR grid affine matrix.
        through_plane_axis: Index (0, 1, or 2) of the through-plane axis.
        lr_spacing_tp: Effective LR voxel spacing along the through-plane
            axis (mm).
        hr_spacing_tp: HR voxel spacing along the through-plane axis (mm).
        lr_shape: Spatial shape (D, H, W) of the native LR volume.
        hr_shape: Spatial shape (D, H, W) of the HR target grid. Used to
            verify that ``lr_spacing_tp`` matches the realised sampling.
        rotation_angles: Optional (rx, ry, rz) in radians for obliqueness.

    Returns:
        4x4 LR affine matrix.
    """
    device = hr_affine.device
    T_lr = hr_affine.clone()

    # Scale through-plane column to LR spacing
    scale = lr_spacing_tp / hr_spacing_tp
    T_lr[:3, through_plane_axis] = T_lr[:3, through_plane_axis] * scale

    # Guard rail: the affine only describes the data if the spacing it encodes
    # is the one the k-space crop actually realised.
    realised = hr_shape[through_plane_axis] / lr_shape[through_plane_axis]
    if abs(realised - scale) > 1e-3 * max(1.0, realised):
        warnings.warn(
            f"build_lr_affine: lr_spacing_tp implies a through-plane scale of "
            f"{scale:.4f} but the LR/HR shapes along axis {through_plane_axis} "
            f"imply {realised:.4f}. The affine will not describe where the LR "
            f"samples actually lie, misregistering the stack against the HR "
            f"grid. Pass the effective spacing "
            f"(hr_spacing_tp * hr_shape[tp] / lr_shape[tp]).",
            stacklevel=2,
        )

    # Apply rotation (obliqueness). Combined with the centre alignment below
    # this rotates the stack about its own FOV centre.
    if rotation_angles is not None:
        rx, ry, rz = rotation_angles
        R = euler_to_rotation_matrix(rx, ry, rz, device=device)
        T_lr[:3, :3] = R @ T_lr[:3, :3]

    # Align centers: compute world-space center of HR and LR FOVs
    hr_center = torch.tensor(
        [(s - 1) / 2.0 for s in hr_shape], dtype=torch.float32, device=device
    )
    lr_center = torch.tensor(
        [(s - 1) / 2.0 for s in lr_shape], dtype=torch.float32, device=device
    )

    # World-space positions of the centers
    hr_center_world = hr_affine[:3, :3] @ hr_center + hr_affine[:3, 3]
    lr_center_world = T_lr[:3, :3] @ lr_center + T_lr[:3, 3]

    # Shift LR origin so centers coincide
    T_lr[:3, 3] += hr_center_world - lr_center_world

    return T_lr


def affine_resample_3d(
    volume: torch.Tensor,
    source_affine: torch.Tensor,
    target_affine: torch.Tensor,
    target_shape: Tuple[int, int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resample a 3D volume between coordinate systems using affine matrices.

    For each voxel in the target grid, computes the corresponding source-space
    coordinate via ``M = inv(T_source) @ T_target`` and samples using
    ``grid_sample``.

    Args:
        volume: Source volume (C, D, H, W).
        source_affine: 4x4 affine of the source volume.
        target_affine: 4x4 affine of the target grid.
        target_shape: Spatial dimensions (D, H, W) of the target grid.
        mode: Interpolation mode — ``'bilinear'`` or ``'nearest'``.

    Returns:
        Resampled volume (C, *target_shape).
    """
    device = volume.device
    src_shape = volume.shape[1:]  # (D_s, H_s, W_s)
    tgt_D, tgt_H, tgt_W = target_shape

    # Mapping: target voxel -> source voxel
    M = torch.linalg.inv(source_affine) @ target_affine  # (4, 4)

    # Build meshgrid of target voxel coordinates
    grid_d = torch.arange(tgt_D, dtype=torch.float32, device=device)
    grid_h = torch.arange(tgt_H, dtype=torch.float32, device=device)
    grid_w = torch.arange(tgt_W, dtype=torch.float32, device=device)
    # (D, H, W) grids
    gd, gh, gw = torch.meshgrid(grid_d, grid_h, grid_w, indexing="ij")

    # Homogeneous coordinates: (D*H*W, 4)
    ones = torch.ones_like(gd)
    coords = torch.stack([gd, gh, gw, ones], dim=-1)  # (D, H, W, 4)
    coords_flat = coords.reshape(-1, 4)  # (N, 4)

    # Transform to source voxel space: (4, 4) @ (4, N) -> (4, N)
    src_coords = (M @ coords_flat.T).T[:, :3]  # (N, 3) — (d, h, w)

    # Normalize to [-1, 1] for grid_sample (align_corners=True)
    # grid_sample expects (x, y, z) = (W, H, D) ordering
    src_d = src_coords[:, 0]
    src_h = src_coords[:, 1]
    src_w = src_coords[:, 2]

    norm_w = 2.0 * src_w / (src_shape[2] - 1) - 1.0 if src_shape[2] > 1 else src_w * 0.0
    norm_h = 2.0 * src_h / (src_shape[1] - 1) - 1.0 if src_shape[1] > 1 else src_h * 0.0
    norm_d = 2.0 * src_d / (src_shape[0] - 1) - 1.0 if src_shape[0] > 1 else src_d * 0.0

    # grid_sample 5D: input (N, C, D, H, W), grid (N, D, H, W, 3) with last dim = (x, y, z) = (W, H, D)
    grid = torch.stack([norm_w, norm_h, norm_d], dim=-1)  # (N, 3)
    grid = grid.reshape(1, tgt_D, tgt_H, tgt_W, 3)  # (1, D, H, W, 3)

    # Add batch dim to volume: (1, C, D_s, H_s, W_s)
    vol_5d = volume.unsqueeze(0)

    # padding_mode="zeros" is load-bearing: voxels mapped outside the source
    # volume return 0, which is what makes the dummy/support-mask FOV trick
    # in resample_with_fov_mask work — those zeros become 1s in the FOV mask.
    resampled = F.grid_sample(
        vol_5d, grid, mode=mode, padding_mode="zeros", align_corners=True
    )

    return resampled.squeeze(0)  # (C, D, H, W)


def resample_with_fov_mask(
    lr_volume: torch.Tensor,
    lr_affine: torch.Tensor,
    hr_affine: torch.Tensor,
    hr_shape: Tuple[int, int, int],
    mode: str = "bilinear",
    support_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resample an LR volume to the HR grid and compute the FOV mask.

    Uses the "dummy mask trick": resamples a mask volume alongside the
    actual image. Out-of-bounds voxels become 0, producing a binary FOV
    validity mask. If ``support_mask`` is provided it replaces the default
    all-ones dummy, encoding which LR voxels survived the FOV slice drop.

    Args:
        lr_volume: Native LR volume (C, D', H', W').
        lr_affine: 4x4 affine of the LR volume.
        hr_affine: 4x4 affine of the HR target grid.
        hr_shape: Spatial dimensions (D, H, W) of the HR grid.
        mode: Interpolation mode for the image (``'bilinear'``).
        support_mask: Optional (1, D', H', W') binary mask where 1 means the
            voxel is valid. When ``None``, an all-ones mask is used.

    Returns:
        Tuple of (resampled_image, fov_mask), each (C, D, H, W).
        ``fov_mask`` is 1 where voxels are **missing** (out-of-bounds in the
        LR stack or dropped by FOV slice drop), 0 where valid LR data exists.
    """
    # Resample image
    resampled_image = affine_resample_3d(
        lr_volume, lr_affine, hr_affine, hr_shape, mode=mode
    )

    # Resample mask with nearest-neighbor
    # The raw result is 1 where valid, 0 where out-of-bounds or dropped.
    # Invert so that 1 = missing, 0 = valid.
    if support_mask is not None:
        dummy = support_mask
    else:
        dummy = torch.ones(1, *lr_volume.shape[1:], device=lr_volume.device)
    validity_mask = affine_resample_3d(
        dummy, lr_affine, hr_affine, hr_shape, mode="nearest"
    )
    fov_mask = 1.0 - validity_mask

    return resampled_image, fov_mask


def apply_fov_slice_drop_native(
    volume: torch.Tensor,
    through_plane_axis: int,
    keep_fraction: float,
    force_both_sides: bool = True,
    drop_from_start: Optional[bool] = None,
) -> torch.Tensor:
    """Drop slices from a native-resolution LR volume along the through-plane axis.

    Zeros out slices at the edges of the volume, simulating incomplete FOV
    coverage. This should be applied *before* resampling to the HR grid so
    the FOV mask naturally captures the dropped regions.

    Args:
        volume: Native LR volume (C, D, H, W).
        through_plane_axis: Spatial axis index (0=D, 1=H, 2=W).
        keep_fraction: Fraction of slices to keep (0, 1].
        force_both_sides: If True, drop from both ends equally.
        drop_from_start: When ``force_both_sides=False``, controls which end
            to drop. ``None`` samples randomly. Pass an explicit boolean to
            share the same side across paired calls (e.g. image + support).

    Returns:
        Volume with dropped slices zeroed out (same shape).
    """
    output = volume.clone()
    # spatial axis in (C, D, H, W) is offset by 1
    axis = through_plane_axis + 1
    axis_size = volume.shape[axis]

    # Drop from one side; sample if not provided so paired calls share it.
    if drop_from_start is None and not force_both_sides:
        drop_from_start = torch.rand(1).item() < 0.5

    lo, hi = kept_slice_bounds(
        axis_size, keep_fraction, force_both_sides, drop_from_start
    )
    if lo == 0 and hi == axis_size:
        return output

    if lo > 0:
        slices = [slice(None)] * volume.ndim
        slices[axis] = slice(0, lo)
        output[tuple(slices)] = 0
    if hi < axis_size:
        slices = [slice(None)] * volume.ndim
        slices[axis] = slice(hi, axis_size)
        output[tuple(slices)] = 0

    return output


def kept_slice_bounds(
    n_slices: int,
    keep_fraction: float,
    force_both_sides: bool = True,
    drop_from_start: Optional[bool] = None,
) -> Tuple[int, int]:
    """Which native LR slices ``apply_fov_slice_drop_native`` keeps: ``[lo, hi)``.

    Factored out so coverage prediction and the drop itself cannot drift apart.

    Args:
        n_slices: Number of slices along the through-plane axis.
        keep_fraction: Fraction of slices to keep (0, 1].
        force_both_sides: If True, the kept slab stays centred.
        drop_from_start: When one-sided, True drops the leading slices.

    Returns:
        ``(lo, hi)`` half-open range of kept slice indices.
    """
    # round, not truncate: int() floors, so the realised fraction was always <= the
    # sampled one (23 slices at 0.567 kept 13 = 0.565) and the recorded
    # fov_keep_fraction systematically overstated what was actually kept.
    n_keep = max(1, min(n_slices, int(round(n_slices * keep_fraction))))
    n_drop = n_slices - n_keep
    if n_drop <= 0:
        return 0, n_slices
    if force_both_sides:
        drop_left = n_drop // 2
        return drop_left, n_slices - (n_drop - drop_left)
    if drop_from_start:
        return n_drop, n_slices
    return 0, n_keep


def crop_native_to_kept_slab(
    volume: torch.Tensor,
    through_plane_axis: int,
    keep_fraction: float,
    force_both_sides: bool = True,
    drop_from_start: Optional[bool] = None,
) -> Tuple[torch.Tensor, int]:
    """Physically remove the dropped slices from a native LR volume.

    :func:`apply_fov_slice_drop_native` ZEROES the dropped slices rather than
    removing them, which is correct for the HR path: the zeros are resampled to
    the HR grid and the FOV mask is derived from them. It is wrong for a native
    stack written to disk as an acquisition. A scanner that images a shorter
    slab produces *fewer slices*, not slices of zeros, and a consumer has no way
    to tell fabricated zeros from measured background — it will treat them as
    data and be trained to reproduce them.

    Returns ``(cropped, lo)``; ``lo`` is the index of the first kept slice, which
    the caller must use to shift the affine origin so the cropped stack still
    lands in the same world position.
    """
    axis = through_plane_axis + 1                      # (C, D, H, W)
    n_slices = volume.shape[axis]
    lo, hi = kept_slice_bounds(
        n_slices, keep_fraction, force_both_sides, drop_from_start
    )
    if lo == 0 and hi == n_slices:
        return volume, 0
    slices = [slice(None)] * volume.ndim
    slices[axis] = slice(lo, hi)
    return volume[tuple(slices)].contiguous(), lo


def shift_affine_origin(affine, through_plane_axis: int, lo: int):
    """Move an affine's origin forward by ``lo`` voxels along one axis.

    Cropping the first ``lo`` slices moves the array's first voxel; without this
    the cropped stack is written at the wrong world position and no longer
    overlaps the other stacks correctly.
    """
    if lo == 0:
        return affine
    import numpy as np

    out = np.array(affine, dtype=float).copy()
    offset = np.zeros(3)
    offset[through_plane_axis] = lo
    out[:3, 3] = out[:3, 3] + out[:3, :3] @ offset
    return out


def compute_brain_bbox_support_mask(
    image: torch.Tensor,
    threshold: float = 1e-3,
    margin: Union[int, Sequence[int]] = 0,
) -> torch.Tensor:
    """Build a bbox-shaped binary support mask around the foreground of ``image``.

    Uses MONAI's :func:`generate_spatial_bounding_box` to find the axis-aligned
    bounding box of voxels above ``threshold``, then returns a (1, D, H, W)
    binary mask that is 1 inside the bbox and 0 outside.

    Mimics how a radiographer sizes the LR scan FOV around the brain in real
    acquisitions: the LR slab tightly contains the brain, so HR voxels falling
    outside the slab show up as "missing" in the FOV mask after resampling.

    Args:
        image: Input volume (C, D, H, W) on any device.
        threshold: Voxel intensity above which is treated as foreground.
        margin: Extra voxels added to each spatial dim of the bbox. Single int
            applies to all dims; a sequence of 3 specifies per-axis margins.

    Returns:
        Binary mask (1, D, H, W) on the same device/dtype as ``image``: 1
        inside the bbox, 0 outside. If no foreground is found, returns a
        mask of all-ones (preserves the all-ones-dummy default behaviour).
    """
    if image.ndim != 4:
        raise ValueError(f"Expected (C, D, H, W); got shape {tuple(image.shape)}")

    spatial_shape = image.shape[1:]
    box_start, box_end = generate_spatial_bounding_box(
        image, select_fn=lambda x: x > threshold, margin=margin
    )

    # Empty foreground -> fall back to all-ones so we don't silently mask
    # everything; warn so a misconfigured threshold is visible.
    if all(s == 0 for s in box_start) and all(e == 0 for e in box_end):
        warnings.warn(
            f"compute_brain_bbox_support_mask: no voxels above threshold "
            f"{threshold!r} found; tight_fov is effectively disabled for "
            f"this volume.",
            stacklevel=2,
        )
        return torch.ones(1, *spatial_shape, dtype=image.dtype, device=image.device)

    mask = torch.zeros(1, *spatial_shape, dtype=image.dtype, device=image.device)
    d0 = max(0, int(box_start[0]))
    h0 = max(0, int(box_start[1]))
    w0 = max(0, int(box_start[2]))
    d1 = min(spatial_shape[0], int(box_end[0]))
    h1 = min(spatial_shape[1], int(box_end[1]))
    w1 = min(spatial_shape[2], int(box_end[2]))
    mask[:, d0:d1, h0:h1, w0:w1] = 1.0
    return mask


def compute_shared_support_mask(
    images: Sequence[torch.Tensor],
    threshold: float = 1e-3,
    margin: Union[int, Sequence[int]] = 0,
) -> torch.Tensor:
    """Build one tight-FOV support mask covering the foreground of every image.

    A radiographer prescribes a single FOV for a session, so the T1, T2 and
    FLAIR of one subject share it. Deriving the bbox per volume instead would
    give each contrast a slightly different box — different tissues are bright
    in different sequences — and their FOV masks would disagree even though the
    stacks were meant to be identical in geometry.

    Takes the union of the per-image bounding boxes, so the shared FOV contains
    every contrast's foreground.

    Args:
        images: Non-empty sequence of (C, D, H, W) volumes sharing one grid.
        threshold: Voxel intensity above which is treated as foreground.
        margin: Extra voxels added to each spatial dim of each bbox.

    Returns:
        Binary mask (1, D, H, W): 1 inside the shared bbox, 0 outside.

    Raises:
        ValueError: If ``images`` is empty or the volumes disagree in shape.
    """
    images = list(images)
    if not images:
        raise ValueError("compute_shared_support_mask requires at least one image.")

    shapes = {tuple(img.shape[1:]) for img in images}
    if len(shapes) > 1:
        raise ValueError(
            f"All volumes must share a spatial grid to share a FOV; got {shapes}."
        )

    masks = [
        compute_brain_bbox_support_mask(img, threshold=threshold, margin=margin)
        for img in images
    ]
    union = masks[0].clone()
    for m in masks[1:]:
        union = torch.maximum(union, m.to(union.device))

    # The union of boxes is not itself a box; re-fit one so the shared FOV
    # stays the rectangular slab a scanner would actually prescribe.
    return compute_brain_bbox_support_mask(union, threshold=0.5, margin=0)

"""Union-coverage guarantee for FOV-cropped stacks.

``fov.ensure_coverage`` means: the union of the stacks must contain the whole
head. A single stack is free to miss most of it, but between them the stacks
have to see every foreground voxel — otherwise the ground truth contains
anatomy that no input observed, and a reconstruction is being asked to invent
it (and then scored on the invention).

Coverage is predicted analytically from the sampled acquisition geometry before
any volume is simulated: each stack's observed region is the set of HR voxels
that land inside its native LR grid *and* inside its kept slab, which is a pair
of parallel half-spaces once the stack's affine (including obliqueness) is
known. When the union falls short, the offending stacks' keep fractions are
grown just enough to close the gap.
"""

import warnings
from typing import List, Optional, Sequence, Tuple

import torch

from mri_synth.fov.resampling import build_lr_affine, kept_slice_bounds


class StackGeometry:
    """Everything needed to predict one stack's observed region.

    Args:
        through_plane_axis: Spatial axis (0, 1, 2) the stack is thick along.
        lr_shape: Native LR spatial shape (D, H, W).
        lr_spacing_tp: Effective through-plane spacing in mm.
        hr_spacing_tp: HR spacing along the same axis in mm.
        rotation_angles: Obliqueness ``(rx, ry, rz)`` in radians, or None.
    """

    __slots__ = (
        "through_plane_axis",
        "lr_shape",
        "lr_spacing_tp",
        "hr_spacing_tp",
        "rotation_angles",
    )

    def __init__(
        self,
        through_plane_axis: int,
        lr_shape: Tuple[int, int, int],
        lr_spacing_tp: float,
        hr_spacing_tp: float,
        rotation_angles: Optional[Tuple[float, float, float]] = None,
    ):
        self.through_plane_axis = through_plane_axis
        self.lr_shape = tuple(lr_shape)
        self.lr_spacing_tp = lr_spacing_tp
        self.hr_spacing_tp = hr_spacing_tp
        self.rotation_angles = rotation_angles


def coarse_foreground(
    image: torch.Tensor,
    threshold: float = 1e-3,
    stride: int = 4,
    relative: bool = True,
    percentile: float = 99.5,
    sanity_range: Tuple[float, float] = (0.05, 0.60),
) -> Tuple[torch.Tensor, int]:
    """Downsample a foreground mask conservatively.

    Uses max-pooling, so a coarse cell is foreground if *any* voxel inside it
    is — coverage is then never claimed for a cell that contains unseen head.

    THRESHOLDING IS RELATIVE BY DEFAULT. An absolute cutoff only means anything
    if every volume occupies the same intensity range, and the failure when it
    does not is silent and asymmetric: a dim subject yields a mask smaller than
    its head, the coverage guarantee is then satisfied against that smaller
    head, and stacks are cropped past anatomy that the threshold excluded; a
    bright subject yields a mask that swells into noise, and the repair loop
    grows keep fractions to cover background, destroying the FOV variation the
    augmentation exists to create. Scaling the cutoff by a high percentile of
    the volume's own intensities makes it track each volume's scale.

    Args:
        image: (C, D, H, W) volume.
        threshold: Absolute cutoff, used when ``relative`` is False.
        stride: Coarse cell size in voxels.
        relative: Interpret ``threshold`` as a FRACTION of the volume's own
            ``percentile`` level rather than an absolute intensity. On data already
            normalised so that percentile is ~1.0 this is identical to the absolute
            behaviour, which is why the default cutoff is unchanged; on data at any
            other scale it tracks the volume instead of silently mis-masking it.
        percentile: Percentile of positive voxels used as the reference level.
        sanity_range: Accepted (min, max) fraction of the volume that may be
            foreground. Outside it the threshold is wrong for this volume and a
            warning is raised — see below.

    Returns:
        ``(mask, stride)`` where mask is a bool tensor of shape
        ``ceil(spatial / stride)``.
    """
    cut = threshold
    if relative:
        pos = image[image > 0]
        if pos.numel() > 0:
            flat = pos.flatten().float()
            # torch.quantile caps at 2**24 elements; subsample above that.
            if flat.numel() > 2 ** 24:
                idx = torch.randperm(flat.numel(), device=flat.device)[: 2 ** 24]
                flat = flat[idx]
            ref = float(torch.quantile(flat, percentile / 100.0))
            if ref > 0:
                cut = max(threshold * ref, torch.finfo(torch.float32).tiny)
    fg_full = (image > cut).any(dim=0, keepdim=True)

    # A head is a sizeable but not dominant fraction of the volume. A mask at
    # 0.1% or 95% means the cutoff is wrong for this volume, and nothing
    # downstream would notice -- the coverage guarantee would simply be computed
    # against the wrong anatomy.
    frac = float(fg_full.float().mean())
    lo, hi = sanity_range
    if not (lo <= frac <= hi):
        warnings.warn(
            f"fov.coarse_foreground: foreground is {frac:.1%} of the volume, outside "
            f"the expected {lo:.0%}-{hi:.0%}. Cutoff {cut:.4g} "
            f"({'relative' if relative else 'absolute'}) is probably wrong for this "
            f"volume; the coverage guarantee will be computed against the wrong head.",
            stacklevel=2,
        )

    fg = fg_full.float().unsqueeze(0)
    if stride > 1:
        fg = torch.nn.functional.max_pool3d(
            fg, kernel_size=stride, stride=stride, ceil_mode=True
        )
    return fg[0, 0] > 0.5, stride


def stack_observed_mask(
    geometry: StackGeometry,
    slab: Tuple[int, int],
    hr_shape: Sequence[int],
    hr_spacing: Sequence[float],
    coarse_shape: Sequence[int],
    stride: int,
    device: torch.device = None,
) -> torch.Tensor:
    """Coarse cells lying *entirely* inside the region this stack observes.

    A HR voxel is observed when its nearest LR sample falls inside the native
    grid and inside the kept slab — which, before rounding, is an intersection
    of half-spaces and therefore convex. A convex region contains a box exactly
    when it contains all eight of its corners, so evaluating the corner lattice
    and AND-ing gives a per-cell answer with no sampling error.

    Testing cell *centres* instead would let a cell whose centre is observed but
    whose corner holds unseen head pass as covered, which is how a coverage
    guarantee ends up leaking a fraction of a percent of the head.
    """
    device = device or torch.device("cpu")
    hr_affine = torch.diag(
        torch.tensor([*hr_spacing, 1.0], dtype=torch.float32, device=device)
    )
    lr_affine = build_lr_affine(
        hr_affine=hr_affine,
        through_plane_axis=geometry.through_plane_axis,
        lr_spacing_tp=geometry.lr_spacing_tp,
        hr_spacing_tp=geometry.hr_spacing_tp,
        lr_shape=geometry.lr_shape,
        hr_shape=tuple(hr_shape),
        rotation_angles=geometry.rotation_angles,
    )

    # HR voxel -> LR voxel
    M = torch.linalg.inv(lr_affine) @ hr_affine

    # Corner lattice: one more point per axis than there are cells, clamped to
    # the last voxel (erring outward, i.e. conservatively).
    grids = [
        torch.clamp(
            torch.arange(c + 1, dtype=torch.float32, device=device) * stride,
            max=float(hr_shape[i] - 1),
        )
        for i, c in enumerate(coarse_shape)
    ]
    gd, gh, gw = torch.meshgrid(*grids, indexing="ij")
    ones = torch.ones_like(gd)
    coords = torch.stack([gd, gh, gw, ones], dim=-1).reshape(-1, 4)
    lr = (M @ coords.T).T[:, :3]

    inside = torch.ones(lr.shape[0], dtype=torch.bool, device=device)
    for axis in range(3):
        nearest = torch.round(lr[:, axis])
        inside &= (nearest >= 0) & (nearest <= geometry.lr_shape[axis] - 1)

    lo, hi = slab
    tp = torch.round(lr[:, geometry.through_plane_axis])
    inside &= (tp >= lo) & (tp < hi)

    c = inside.reshape(coarse_shape[0] + 1, coarse_shape[1] + 1, coarse_shape[2] + 1)
    return (
        c[:-1, :-1, :-1] & c[1:, :-1, :-1] & c[:-1, 1:, :-1] & c[:-1, :-1, 1:]
        & c[1:, 1:, :-1] & c[1:, :-1, 1:] & c[:-1, 1:, 1:] & c[1:, 1:, 1:]
    )


def ensure_union_coverage(
    foreground: torch.Tensor,
    stride: int,
    geometries: List[StackGeometry],
    keep_fractions: List[float],
    drop_flags: List[bool],
    drop_from_start: List[bool],
    force_both_sides: bool,
    hr_shape: Sequence[int],
    hr_spacing: Sequence[float],
    step: float = 0.05,
    max_iters: int = 80,
    device: torch.device = None,
) -> Tuple[List[float], List[bool], float]:
    """Grow FOV slabs until the stacks jointly observe the whole head.

    Each round the stack whose next growth step recovers the most unseen head
    is widened, so the sampled cropping is preserved wherever it is already
    harmless and relaxed only where it actually loses anatomy.

    Args:
        foreground: Coarse bool head mask from :func:`coarse_foreground`.
        stride: Coarse cell size used to build ``foreground``.
        geometries: Per-stack geometry.
        keep_fractions: Per-stack sampled keep fraction.
        drop_flags: Per-stack "FOV cropping applied" flags.
        drop_from_start: Per-stack dropped end (one-sided mode only).
        force_both_sides: Whether slabs stay centred.
        hr_shape: HR spatial shape.
        hr_spacing: HR voxel spacing in mm.
        step: Keep-fraction increment per repair round.
        max_iters: Safety bound on repair rounds.
        device: Torch device.

    Returns:
        ``(keep_fractions, drop_flags, uncovered_fraction)`` after repair. A
        non-zero ``uncovered_fraction`` means the head cannot be covered at
        this obliqueness/resolution even with no cropping at all.
    """
    device = device or foreground.device
    coarse_shape = tuple(foreground.shape)
    n_fg = int(foreground.sum().item())
    if n_fg == 0:
        return keep_fractions, drop_flags, 0.0

    keep = list(keep_fractions)
    flags = list(drop_flags)

    def observed(idx: int, k: float) -> torch.Tensor:
        geom = geometries[idx]
        n_slices = geom.lr_shape[geom.through_plane_axis]
        slab = kept_slice_bounds(
            n_slices, k, force_both_sides, drop_from_start[idx]
        )
        return stack_observed_mask(
            geom, slab, hr_shape, hr_spacing, coarse_shape, stride, device
        )

    masks = [observed(s, keep[s]) for s in range(len(geometries))]

    for _ in range(max_iters):
        union = masks[0].clone()
        for m in masks[1:]:
            union |= m
        uncovered = foreground & ~union
        n_unc = int(uncovered.sum().item())
        if n_unc == 0:
            return keep, flags, 0.0

        best_idx, best_mask, best_gain, best_k = None, None, 0, None
        for s in range(len(geometries)):
            if keep[s] >= 1.0:
                continue
            k_new = min(1.0, keep[s] + step)
            m_new = observed(s, k_new)
            gain = int((uncovered & m_new).sum().item())
            if gain > best_gain:
                best_idx, best_mask, best_gain, best_k = s, m_new, gain, k_new

        if best_idx is None:
            # Nothing left to widen, or widening recovers nothing.
            if all(k >= 1.0 for k in keep):
                break
            # No single step helps; widen every croppable stack so we keep
            # making progress instead of stalling.
            progressed = False
            for s in range(len(geometries)):
                if keep[s] < 1.0:
                    keep[s] = min(1.0, keep[s] + step)
                    masks[s] = observed(s, keep[s])
                    progressed = True
            if not progressed:
                break
            continue

        keep[best_idx] = best_k
        masks[best_idx] = best_mask
        if best_k >= 1.0:
            flags[best_idx] = False

    union = masks[0].clone()
    for m in masks[1:]:
        union |= m
    uncovered = foreground & ~union
    frac = float(uncovered.sum().item()) / n_fg
    if frac > 0:
        warnings.warn(
            f"fov.ensure_coverage: {frac:.1%} of the head is observed by no "
            f"stack even with FOV cropping fully relaxed. The loss is coming "
            f"from obliqueness rotating anatomy out of the LR grid, not from "
            f"slice dropping — reduce fov.obliqueness_range or pad the volume.",
            stacklevel=2,
        )
    return keep, flags, frac

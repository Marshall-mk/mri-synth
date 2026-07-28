"""K-space artifact helpers: motion ghosting, spikes, and aliasing."""

import numpy as np
import torch


def apply_kspace_motion_ghosting(
    volume: torch.Tensor,
    axis: int,
    intensity: float = 0.5,
    num_ghosts: int = 2,
) -> torch.Tensor:
    """
    Simulate motion artifacts (ghosting) in k-space.

    Patient movement during acquisition causes positional inconsistencies
    in the frequency data, manifesting as faint copies of the anatomy
    propagated along the phase encoding direction.
    """
    k_space = torch.fft.fftn(volume, dim=(1, 2, 3))
    k_space = torch.fft.fftshift(k_space, dim=(1, 2, 3))
    dims = volume.shape[1:]
    phase_axis_len = dims[axis]
    indices = torch.arange(phase_axis_len, device=volume.device)
    phase_error = torch.exp(
        1j
        * intensity
        * torch.sin(2 * np.pi * num_ghosts * indices / phase_axis_len)
    )
    view_shape = [1, 1, 1, 1]
    view_shape[axis + 1] = phase_axis_len
    phase_error = phase_error.view(*view_shape)
    k_space_corrupted = k_space * phase_error
    k_space_corrupted = torch.fft.ifftshift(k_space_corrupted, dim=(1, 2, 3))
    return torch.abs(torch.fft.ifftn(k_space_corrupted, dim=(1, 2, 3)))


def apply_kspace_spike(
    volume: torch.Tensor, intensity: float = 0.04
) -> torch.Tensor:
    """
    Simulate RF spikes (zipper artifacts).

    Stray RF interference appears as a high-intensity spike at a specific
    point in k-space, producing periodic stripes when reconstructed.

    Args:
        volume: Input volume (C, D, H, W).
        intensity: Spike amplitude as a fraction of peak k-space magnitude.
            Defaults to ``ArtifactConfig.spike_intensity`` so calling this
            directly matches what the pipeline does. It previously defaulted to
            ``5.0`` — 125x the configured value, enough to swamp the anatomy
            rather than overlay stripes on it.

    Returns:
        Volume with a k-space spike applied (same shape).
    """
    k_space = torch.fft.fftn(volume, dim=(1, 2, 3))
    C, D, H, W = volume.shape
    rd = torch.randint(0, D, (1,))
    rh = torch.randint(0, H, (1,))
    rw = torch.randint(0, W, (1,))
    spike_val = torch.max(torch.abs(k_space)) * intensity
    k_space[:, rd, rh, rw] += spike_val
    return torch.abs(torch.fft.ifftn(k_space, dim=(1, 2, 3)))


def apply_aliasing(
    volume: torch.Tensor, axis: int, fold_pct: float = 0.2
) -> torch.Tensor:
    """
    Simulate wrap-around aliasing (fold-over artifacts).

    When the FOV is smaller than the anatomy along the phase encoding
    direction, signal from beyond one edge re-enters at the opposite edge.
    Only those wrap bands gain signal; the interior of the image is untouched,
    and anatomy that sits comfortably inside the FOV produces no fold-over at
    all — as in a real acquisition with an adequate FOV.

    BUG FIX: this used to roll the *whole* volume and blend it as
    ``(volume + 0.5 * roll(+s) + 0.5 * roll(-s)) / 1.5``. Those weights sum to
    2 but were divided by 1.5, so every voxel — including regions no wrapped
    signal reaches — was scaled by 4/3. Enabling aliasing brightened the image
    by 33% and pushed values past 1.0, which ``clip_to_unit_range`` then
    saturated. It also attenuated nothing and displaced everything, which reads
    as a triple exposure rather than fold-over.

    Args:
        volume: Input volume (C, D, H, W).
        axis: Spatial axis (0, 1, 2) acting as the phase encoding direction.
        fold_pct: Fraction of the extent that lies outside the FOV, split
            between the two ends.

    Returns:
        Volume with wrapped signal added at both ends (same shape).
    """
    spatial_axis = axis + 1
    n = volume.shape[spatial_axis]
    band = min(int(n * fold_pct / 2), n // 2)

    out = volume.clone()
    if band <= 0:
        return out

    head = volume.narrow(spatial_axis, 0, band)
    tail = volume.narrow(spatial_axis, n - band, band)
    # Anatomy past the near edge reappears at the far edge, and vice versa.
    out.narrow(spatial_axis, n - band, band).add_(head)
    out.narrow(spatial_axis, 0, band).add_(tail)
    return out

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
    volume: torch.Tensor, intensity: float = 5.0
) -> torch.Tensor:
    """
    Simulate RF spikes (zipper artifacts).

    Stray RF interference appears as a high-intensity spike at a specific
    point in k-space, producing periodic stripes when reconstructed.
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

    When the FOV is smaller than the anatomy in the phase encoding direction,
    signal from outside the FOV wraps around to the opposite side.
    """
    spatial_axis = axis + 1
    original_size = volume.shape[spatial_axis]
    shift = int(original_size * fold_pct / 2)
    wrapped = (
        torch.roll(volume, shifts=shift, dims=spatial_axis) * 0.5
        + torch.roll(volume, shifts=-shift, dims=spatial_axis) * 0.5
    )
    return (volume + wrapped) / 1.5

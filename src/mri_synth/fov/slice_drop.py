"""FOV slice dropping simulation."""

from typing import Dict, Tuple

import torch
import torch.nn as nn


class FOVSliceDrop(nn.Module):
    """
    Simulates incomplete FOV by dropping slices from the through-plane axis.

    Mimics real clinical scenarios where stacks may miss slices at the edges
    of the anatomy (e.g., vertex/skull base for axial stacks).

    Should be applied to UPSAMPLED LR stacks — simply zeros out slices.

    Args:
        prob: Probability of applying FOV drop to each orientation.
        min_keep_fraction: Minimum fraction of slices to keep.
        max_keep_fraction: Maximum fraction to keep.
        force_both_sides: If True, always drop from both ends.
        orientation_to_axis: Mapping from orientation index to through-plane
            axis in (B, C, D, H, W) format. Defaults to the standard 3-stack
            mapping {0: 4, 1: 3, 2: 2}. Override for N > 3 stacks.
    """

    def __init__(
        self,
        prob: float = 0.4,
        min_keep_fraction: float = 0.5,
        max_keep_fraction: float = 0.95,
        force_both_sides: bool = True,
        orientation_to_axis: Dict[int, int] = None,
    ):
        super().__init__()
        self.prob = prob
        self.min_keep_fraction = min_keep_fraction
        self.max_keep_fraction = max_keep_fraction
        self.force_both_sides = force_both_sides
        # Default: Axial(0)→W(4), Coronal(1)→H(3), Sagittal(2)→D(2)
        if orientation_to_axis is not None:
            self.orientation_to_axis = orientation_to_axis
        else:
            self.orientation_to_axis = {0: 4, 1: 3, 2: 2}

    def forward(
        self,
        image: torch.Tensor,
        orientation_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply FOV slice dropping.

        Args:
            image: Input volume (B, C, D, H, W).
            orientation_idx: Index into orientation_to_axis mapping.

        Returns:
            cropped_image: Image with some slices zeroed out (same shape).
            spatial_mask: Binary mask (B, 1, D, H, W), 1=valid, 0=dropped.
        """
        B, C, D, H, W = image.shape
        device = image.device

        output = image.clone()
        spatial_mask = torch.ones(B, 1, D, H, W, device=device)

        axis = self.orientation_to_axis[orientation_idx % len(self.orientation_to_axis)]
        axis_size = image.shape[axis]

        for b in range(B):
            if torch.rand(1).item() > self.prob:
                continue

            keep_fraction = (
                torch.rand(1).item()
                * (self.max_keep_fraction - self.min_keep_fraction)
                + self.min_keep_fraction
            )
            n_keep = max(1, int(axis_size * keep_fraction))
            n_drop = axis_size - n_keep

            if n_drop == 0:
                continue

            if self.force_both_sides:
                drop_mode = 2
            else:
                drop_mode = torch.randint(0, 3, (1,)).item()

            if drop_mode == 0:
                drop_start, drop_end = 0, n_drop
            elif drop_mode == 1:
                drop_start, drop_end = axis_size - n_drop, axis_size
            else:
                drop_each = n_drop // 2
                drop_start_left, drop_end_left = 0, drop_each
                drop_start_right = axis_size - (n_drop - drop_each)
                drop_end_right = axis_size

            if axis == 2:  # D
                if drop_mode == 2:
                    output[b, :, :drop_each, :, :] = 0
                    output[b, :, drop_start_right:, :, :] = 0
                    spatial_mask[b, :, :drop_each, :, :] = 0
                    spatial_mask[b, :, drop_start_right:, :, :] = 0
                else:
                    output[b, :, drop_start:drop_end, :, :] = 0
                    spatial_mask[b, :, drop_start:drop_end, :, :] = 0

            elif axis == 3:  # H
                if drop_mode == 2:
                    output[b, :, :, :drop_each, :] = 0
                    output[b, :, :, drop_start_right:, :] = 0
                    spatial_mask[b, :, :, :drop_each, :] = 0
                    spatial_mask[b, :, :, drop_start_right:, :] = 0
                else:
                    output[b, :, :, drop_start:drop_end, :] = 0
                    spatial_mask[b, :, :, drop_start:drop_end, :] = 0

            elif axis == 4:  # W
                if drop_mode == 2:
                    output[b, :, :, :, :drop_each] = 0
                    output[b, :, :, :, drop_start_right:] = 0
                    spatial_mask[b, :, :, :, :drop_each] = 0
                    spatial_mask[b, :, :, :, drop_start_right:] = 0
                else:
                    output[b, :, :, :, drop_start:drop_end] = 0
                    spatial_mask[b, :, :, :, drop_start:drop_end] = 0

        return output, spatial_mask

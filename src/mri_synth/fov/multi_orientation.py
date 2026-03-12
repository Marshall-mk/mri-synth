"""Multi-orientation FOV drop with coverage guarantees."""

from typing import List, Tuple

import torch
import torch.nn as nn

from mri_synth.fov.slice_drop import FOVSliceDrop


class MultiOrientationFOVDrop(nn.Module):
    """
    Apply FOV slice dropping to multiple orientation stacks.

    Ensures that not ALL orientations lose the same region — at least one
    orientation should cover each anatomical region (when ensure_coverage
    is True).

    Generalized for N stacks (not limited to 3).

    Args:
        prob_per_orientation: Probability of dropping slices per orientation.
        min_keep_fraction: Minimum fraction of slices to keep.
        max_keep_fraction: Maximum fraction of slices to keep.
        ensure_coverage: If True, ensures at least one orientation is kept
            fully intact per batch element.
        force_both_sides: If True, always drop from both ends.
    """

    def __init__(
        self,
        prob_per_orientation: float = 0.4,
        min_keep_fraction: float = 0.5,
        max_keep_fraction: float = 0.95,
        ensure_coverage: bool = True,
        force_both_sides: bool = True,
    ):
        super().__init__()
        self.prob = prob_per_orientation
        self.min_keep = min_keep_fraction
        self.max_keep = max_keep_fraction
        self.ensure_coverage = ensure_coverage
        self.force_both_sides = force_both_sides

        self.dropper = FOVSliceDrop(
            prob=1.0,  # We control probability at this level
            min_keep_fraction=min_keep_fraction,
            max_keep_fraction=max_keep_fraction,
            force_both_sides=force_both_sides,
        )

    def forward(
        self,
        lr_stacks: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Apply FOV dropping to a list of orientation stacks.

        Args:
            lr_stacks: List of N tensors, each (B, C, D, H, W).

        Returns:
            dropped_stacks: List of N tensors with FOV dropping applied.
            spatial_masks: List of N mask tensors (B, 1, D, H, W).
        """
        num_stacks = len(lr_stacks)
        B = lr_stacks[0].shape[0]
        device = lr_stacks[0].device

        dropped_stacks = []
        spatial_masks = []

        # Decide which orientations to drop for each batch element
        batch_drop_decisions = []
        for b in range(B):
            drop_decisions = [
                torch.rand(1).item() < self.prob for _ in range(num_stacks)
            ]

            # If ensure_coverage and all would be dropped, keep one
            if self.ensure_coverage and all(drop_decisions):
                keep_idx = torch.randint(0, num_stacks, (1,)).item()
                drop_decisions[keep_idx] = False

            batch_drop_decisions.append(drop_decisions)

        # Apply dropping per orientation
        for orient_idx in range(num_stacks):
            stack = lr_stacks[orient_idx].clone()
            B, C, D, H, W = stack.shape
            mask = torch.ones(B, 1, D, H, W, device=device)

            for b in range(B):
                if batch_drop_decisions[b][orient_idx]:
                    single_stack = stack[b : b + 1]
                    dropped, single_mask = self.dropper(single_stack, orient_idx)
                    stack[b : b + 1] = dropped
                    mask[b : b + 1] = single_mask

            dropped_stacks.append(stack)
            spatial_masks.append(mask)

        return dropped_stacks, spatial_masks


def apply_fov_augmentation(
    lr_stacks: List[torch.Tensor],
    prob: float = 0.4,
    min_keep: float = 0.5,
    max_keep: float = 0.95,
    ensure_coverage: bool = True,
    force_both_sides: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Convenience function to apply FOV augmentation to LR stacks.

    Args:
        lr_stacks: List of N tensors (one per orientation).
        prob: Probability of dropping slices per orientation.
        min_keep: Minimum fraction of slices to keep.
        max_keep: Maximum fraction of slices to keep.
        ensure_coverage: If True, ensure complementary coverage.
        force_both_sides: If True, drop from both ends.

    Returns:
        augmented_stacks: Stacks with FOV dropping.
        spatial_masks: Corresponding validity masks.
    """
    augmenter = MultiOrientationFOVDrop(
        prob_per_orientation=prob,
        min_keep_fraction=min_keep,
        max_keep_fraction=max_keep,
        ensure_coverage=ensure_coverage,
        force_both_sides=force_both_sides,
    )
    return augmenter(lr_stacks)

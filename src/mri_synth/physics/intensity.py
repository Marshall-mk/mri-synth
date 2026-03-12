"""Intensity augmentation (clipping and gamma correction)."""

from typing import Tuple, Union

import torch
import torch.nn as nn


class IntensityAugmentation(nn.Module):
    """
    Simulates variations in MRI contrast mechanisms and sensor dynamics.

    1. **Clipping** — dynamic range limits of the MRI receiver/ADC.
    2. **Gamma correction** — power-law transform approximating different
       T1/T2 weightings.

    Args:
        clip: Clipping bounds (float, tuple, or False).
        gamma_std: Standard deviation of gamma parameter.
        channel_wise: If True, apply different gamma per channel.
        prob_gamma: Probability of applying gamma correction.
    """

    def __init__(
        self,
        clip: Union[float, Tuple[float, float], bool] = 300,
        gamma_std: float = 0.5,
        channel_wise: bool = False,
        prob_gamma: float = 0.95,
    ):
        super().__init__()
        self.clip = clip
        self.gamma_std = gamma_std
        self.channel_wise = channel_wise
        self.prob_gamma = prob_gamma

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply intensity augmentations to input volumes.

        Args:
            image: Input tensor of shape (B, C, D, H, W).

        Returns:
            Augmented tensor of shape (B, C, D, H, W).
        """
        batch_size, n_channels = image.shape[:2]
        ndims = len(image.shape) - 2
        device = image.device

        if self.clip:
            if isinstance(self.clip, (int, float)):
                image = torch.clamp(image, 0, self.clip)
            else:
                image = torch.clamp(image, self.clip[0], self.clip[1])

        if self.gamma_std > 0 and torch.rand(1).item() < self.prob_gamma:
            if self.channel_wise:
                gamma = (
                    torch.randn(batch_size, n_channels, *([1] * ndims), device=device)
                    * self.gamma_std
                )
            else:
                gamma = (
                    torch.randn(batch_size, 1, *([1] * ndims), device=device)
                    * self.gamma_std
                )

            image = torch.pow(image.clamp(min=1e-7), torch.exp(gamma))

        return image

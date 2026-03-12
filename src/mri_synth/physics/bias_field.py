"""Bias field (B1 inhomogeneity) simulation."""

import torch
import torch.nn as nn
from monai.transforms import Resize


class BiasFieldCorruption(nn.Module):
    """
    Simulates MRI bias field (B1 inhomogeneity) artifacts.

    The bias field is modeled as a multiplicative smooth field that varies
    over the image volume, simulating RF coil sensitivity variations.

    Args:
        bias_field_std: Standard deviation of the bias field coefficients.
        bias_scale: Scale factor for the bias field resolution.
        prob: Probability of applying this corruption.
    """

    def __init__(
        self,
        bias_field_std: float = 0.3,
        bias_scale: float = 0.025,
        prob: float = 0.98,
    ):
        super().__init__()
        self.bias_field_std = bias_field_std
        self.bias_scale = bias_scale
        self.prob = prob

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply a multiplicative bias field to the input volume.

        Args:
            image: Input tensor of shape (B, C, D, H, W).

        Returns:
            Corrupted tensor of shape (B, C, D, H, W).
        """
        if torch.rand(1).item() > self.prob:
            return image

        batch_size = image.shape[0]
        spatial_shape = image.shape[2:]
        device = image.device
        outputs = []

        for b in range(batch_size):
            img = image[b : b + 1]

            bias_shape = [max(1, int(s * self.bias_scale)) for s in spatial_shape]
            bias_coeffs = (
                torch.randn(1, 1, *bias_shape, device=device) * self.bias_field_std
            )

            resize_transform = Resize(spatial_size=spatial_shape, mode="trilinear")
            bias_field = resize_transform(bias_coeffs.squeeze(0)).unsqueeze(0)

            bias_field = torch.exp(bias_field)
            img = img * bias_field
            outputs.append(img)

        return torch.cat(outputs, dim=0)

"""Resolution sampling configuration."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ResolutionConfig:
    """
    Configuration for MRI acquisition resolution parameters.

    Holds resolution range parameters used by HRLRDataGenerator to create
    orthogonal anisotropic resolution configurations.

    Args:
        min_resolution: Minimum (highest quality) resolution in mm per axis.
        max_res_aniso: Maximum anisotropic resolution in mm per axis.
        prob_min: Probability of using minimum resolution.
    """

    min_resolution: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    max_res_aniso: List[float] = field(default_factory=lambda: [9.0, 9.0, 9.0])
    prob_min: float = 0.05

    def sample_through_plane_resolution(self, axis: int, generator=None) -> float:
        """Sample a through-plane resolution for ``axis``.

        With probability ``prob_min`` the axis is acquired at its best resolution
        (an isotropic-ish stack); otherwise it is drawn uniformly between the
        minimum and the anisotropic maximum.
        """
        import torch

        lo = float(self.min_resolution[axis])
        hi = float(self.max_res_aniso[axis])
        if hi <= lo:
            return lo
        u = torch.rand(1, generator=generator).item()
        if u < self.prob_min:
            return lo
        return lo + torch.rand(1, generator=generator).item() * (hi - lo)

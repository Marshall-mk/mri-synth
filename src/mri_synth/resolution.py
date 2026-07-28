"""Resolution sampling configuration."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ResolutionConfig:
    """
    Configuration for MRI acquisition resolution parameters.

    Holds resolution range parameters used by HRLRDataGenerator to create
    orthogonal anisotropic resolution configurations.

    Note: The original SampleResolution.forward() method was dead code (never
    called) — only min_res and max_res_aniso were accessed. This dataclass
    replaces that nn.Module.

    Args:
        min_resolution: Minimum (highest quality) resolution in mm per axis.
        max_res_aniso: Maximum anisotropic resolution in mm per axis.
        prob_min: Probability of using minimum resolution.
    """

    min_resolution: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    max_res_aniso: List[float] = field(default_factory=lambda: [9.0, 9.0, 9.0])
    prob_min: float = 0.05

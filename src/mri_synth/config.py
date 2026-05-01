"""Pydantic configuration models for MRI synthesis."""

from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field


class PhysicsConfig(BaseModel):
    """Configuration for physics-based simulation."""

    psf_type: Literal["boxcar", "gaussian", "trapezoid"] = "trapezoid"
    edge_width: float = 0.1
    bias_field_std: float = 0.3
    prob_bias_field: float = 0.5


class ArtifactConfig(BaseModel):
    """Configuration for artifact simulation."""

    prob_motion: float = 0.2
    prob_spike: float = 0.05
    prob_aliasing: float = 0.1
    prob_noise: float = 0.8
    noise_std: float = 0.02
    motion_intensity: float = 0.5
    spike_intensity: float = 0.04


class FOVConfig(BaseModel):
    """Configuration for FOV augmentation and obliqueness simulation."""

    enable: bool = True
    prob: float = 0.7
    min_keep: float = 0.40
    max_keep: float = 0.70
    ensure_coverage: bool = True
    force_both_sides: bool = True
    obliqueness_range: float = 15.0
    enable_obliqueness: bool = True
    prob_obliqueness: float = 0.5
    tight_fov: bool = True
    tight_fov_threshold: float = 1e-3
    tight_fov_margin: int = 0


class GenerationConfig(BaseModel):
    """Top-level configuration for MRI synthesis generation."""

    num_stacks: int = 3
    num_variations: int = 1
    atlas_res: List[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    target_res: List[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    min_resolution: List[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    max_res_aniso: List[float] = Field(default_factory=lambda: [9.0, 9.0, 9.0])
    randomise_res: bool = True
    apply_intensity_aug: bool = False
    clip_to_unit_range: bool = True
    upsample_mode: str = "trilinear"
    return_intermediate: bool = False
    save_native_res: bool = False
    orientation_dropout_prob: float = 0.0
    min_orientations: int = 1
    drop_orientations: Optional[List[int]] = None
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    fov: FOVConfig = Field(default_factory=FOVConfig)
    device: str = "cpu"
    seed: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: str) -> "GenerationConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

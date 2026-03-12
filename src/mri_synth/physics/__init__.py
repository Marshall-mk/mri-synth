"""Physics-based MRI simulation components."""

from mri_synth.physics.slice_profile import SliceProfilePhysics
from mri_synth.physics.bias_field import BiasFieldCorruption
from mri_synth.physics.intensity import IntensityAugmentation

__all__ = ["SliceProfilePhysics", "BiasFieldCorruption", "IntensityAugmentation"]

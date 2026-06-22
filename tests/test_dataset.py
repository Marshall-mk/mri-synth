"""Tests for GeneratorDataset wrappers."""

import pytest

from mri_synth.dataset import GeneratorDataset
from mri_synth.pipeline import HRLRDataGenerator


def test_rejects_return_intermediate_generator():
    """Native-res stacks aren't batch-collatable; the dataset must reject
    a generator created with return_intermediate=True instead of crashing
    later on a tuple-unpack/collate error."""
    gen = HRLRDataGenerator(num_stacks=3, return_intermediate=True)
    with pytest.raises(ValueError, match="return_intermediate"):
        GeneratorDataset(base_dataset=[], generator=gen)


def test_accepts_standard_generator():
    """A normal generator (return_intermediate=False) is accepted."""
    gen = HRLRDataGenerator(num_stacks=3, return_intermediate=False)
    ds = GeneratorDataset(base_dataset=[], generator=gen)
    assert ds.generator is gen

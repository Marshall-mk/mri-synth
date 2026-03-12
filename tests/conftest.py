"""Shared test fixtures."""

import pytest
import torch


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def synthetic_volume():
    """Synthetic 32x32x32 single-channel volume."""
    torch.manual_seed(42)
    return torch.rand(1, 1, 32, 32, 32)


@pytest.fixture
def synthetic_volume_no_batch():
    """Synthetic 32x32x32 single-channel volume without batch dim."""
    torch.manual_seed(42)
    return torch.rand(1, 32, 32, 32)


@pytest.fixture
def batch_volume():
    """Batch of 2 synthetic volumes."""
    torch.manual_seed(42)
    return torch.rand(2, 1, 32, 32, 32)

"""
FlashKernel — pytest configuration.

Registers custom markers and provides shared fixtures.
"""

import pytest
import torch


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "cuda: test requires CUDA GPU")
    config.addinivalue_line("markers", "slow: test takes > 10 seconds")


@pytest.fixture(scope="session")
def device():
    """Returns 'cuda' if available, else skips."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def device_name():
    """Returns the name of the CUDA device."""
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)

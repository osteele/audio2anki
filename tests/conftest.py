"""Pytest configuration file."""

import pytest
from rich.progress import Progress


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")


@pytest.fixture
def progress() -> Progress:
    """Progress bar for testing."""
    return Progress()

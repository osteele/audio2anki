"""Pytest configuration file."""

from collections.abc import Generator
from pathlib import Path

import pytest
from rich.progress import Progress

from audio2anki import cache


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")


@pytest.fixture
def progress() -> Progress:
    """Progress bar for testing."""
    return Progress()


@pytest.fixture
def test_cache_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary directory for cache and initialize cache to use it."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    cache._cache = cache.FileCache(cache_dir)  # type: ignore
    yield cache_dir
    cache.clear_cache()

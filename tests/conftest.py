"""Pytest configuration file."""

import os
from collections.abc import Generator
from pathlib import Path
from typing import TypedDict

import pytest
from rich.progress import Progress

from audio2anki import cache


class CacheTestEnv(TypedDict):
    """Type definition for test environment."""

    config_dir: Path
    cache_dir: Path


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


@pytest.fixture
def test_env(tmp_path: Path) -> Generator[CacheTestEnv, None, None]:
    """Create a temporary environment for config and cache directories."""
    # Create temporary directories
    test_dir = tmp_path / "audio2anki_test"
    config_home = test_dir / "config"
    cache_home = test_dir / "cache"

    # Store original environment variables
    original_env = {
        "XDG_CONFIG_HOME": os.environ.get("XDG_CONFIG_HOME"),
        "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME"),
    }

    # Set environment variables for test
    os.environ["XDG_CONFIG_HOME"] = str(config_home)
    os.environ["XDG_CACHE_HOME"] = str(cache_home)

    yield {
        "config_dir": config_home,
        "cache_dir": cache_home,
    }

    # Restore original environment variables
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

"""Pytest configuration file."""

import os
from collections.abc import Generator
from pathlib import Path
from typing import TypedDict

import pytest
from pydub import AudioSegment
from rich.progress import Progress

from audio2anki import cache


class CacheTestEnv(TypedDict):
    """Type definition for test environment."""

    config_dir: Path
    temp_dir: Path


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
    cache._cache = cache.TempDirCache(keep_files=True)  # type: ignore
    yield cache_dir
    cache.cleanup_cache()


@pytest.fixture
def test_env(tmp_path: Path) -> Generator[CacheTestEnv, None, None]:
    """Create a temporary environment for config directory."""
    # Create temporary directories
    test_dir = tmp_path / "audio2anki_test"
    config_home = test_dir / "config"

    # Store original environment variables
    original_env = {
        "XDG_CONFIG_HOME": os.environ.get("XDG_CONFIG_HOME"),
    }

    # Set environment variables for test
    os.environ["XDG_CONFIG_HOME"] = str(config_home)

    # Initialize a test cache
    from audio2anki.cache import TempDirCache, init_cache

    cache = init_cache(keep_files=True)

    # Get the temp_dir - safe cast since we know init_cache returns a TempDirCache
    temp_cache = cache if isinstance(cache, TempDirCache) else TempDirCache(keep_files=True)

    yield {
        "config_dir": config_home,
        "temp_dir": temp_cache.temp_dir,
    }

    # Restore original environment variables
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    # Clean up the cache
    from audio2anki.cache import cleanup_cache

    cleanup_cache()


@pytest.fixture
def test_audio_file(tmp_path: Path) -> Path:
    """Create a valid test audio file."""
    audio = AudioSegment.silent(duration=1000)  # 1 second of silence
    file_path = tmp_path / "test.mp3"
    audio.export(str(file_path), format="mp3")
    return file_path

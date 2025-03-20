"""Tests for the temporary cache module."""

import os
import tempfile
from collections.abc import Generator

import pytest

from audio2anki.cache import TempDirCache, cleanup_cache, get_artifact_path, init_cache, store_artifact


@pytest.fixture
def temp_input_file() -> Generator[str, None, None]:
    """Create a temporary input file for testing."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"test data")
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_cache() -> Generator[TempDirCache, None, None]:
    """Create a temporary cache for testing."""
    cache = TempDirCache(keep_files=True)  # Keep files for testing
    yield cache
    cache.cleanup()


def test_temp_dir_creation(temp_cache: TempDirCache) -> None:
    """Test that the temporary directory is created."""
    assert temp_cache.temp_dir.exists()
    assert temp_cache.temp_dir.is_dir()
    assert "audio2anki_" in str(temp_cache.temp_dir)


def test_get_path(temp_cache: TempDirCache) -> None:
    """Test getting artifact paths."""
    path = temp_cache.get_path("transcript", "srt")
    assert str(path).endswith("transcript.srt")
    assert str(temp_cache.temp_dir) in str(path)

    # Test with extension that already has a leading dot
    path = temp_cache.get_path("audio", ".mp3")
    assert str(path).endswith("audio.mp3")


def test_store_artifact(temp_cache: TempDirCache) -> None:
    """Test storing an artifact in the cache."""
    test_data = b"test artifact data"
    path = temp_cache.store("audio", test_data, "mp3")

    # Check file exists and has correct content
    assert path.exists()
    with open(path, "rb") as f:
        assert f.read() == test_data


def test_cleanup(temp_cache: TempDirCache) -> None:
    """Test cleanup with keep_files=False."""
    temp_dir = temp_cache.temp_dir

    # Override keep_files to False for this test
    temp_cache.keep_files = False

    # Store an artifact
    temp_cache.store("audio", b"test data", "mp3")

    # Verify directory exists
    assert temp_dir.exists()

    # Clean up
    temp_cache.cleanup()

    # Verify directory is removed
    assert not temp_dir.exists()


def test_keep_files(temp_cache: TempDirCache) -> None:
    """Test that files are kept when keep_files=True."""
    # Store an artifact
    temp_cache.store("audio", b"test data", "mp3")

    # Clean up with keep_files=True (default for this fixture)
    temp_cache.cleanup()

    # Verify directory still exists
    assert temp_cache.temp_dir.exists()

    # Manually clean up for the test
    if temp_cache.temp_dir.exists():
        import shutil

        shutil.rmtree(temp_cache.temp_dir)


def test_global_functions() -> None:
    """Test the global cache functions."""
    # Initialize a new cache
    init_cache(keep_files=True)
    try:
        # Get path for an artifact
        path = get_artifact_path("audio", "mp3")
        assert str(path).endswith("audio.mp3")

        # Store an artifact
        stored_path = store_artifact("audio", b"test global data", "mp3")
        assert stored_path.exists()

        # Verify content
        with open(stored_path, "rb") as f:
            assert f.read() == b"test global data"
    finally:
        # Clean up
        cleanup_cache()

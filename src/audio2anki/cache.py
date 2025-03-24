"""Temporary cache module for storing pipeline artifacts during a single run."""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Protocol

# Use module-level logger
logger = logging.getLogger(__name__)


class Cache(Protocol):
    """Protocol for cache implementations."""

    deck_path: Path | None
    temp_dir: Path

    def get_path(self, artifact_name: str, extension: str) -> Path:
        """
        Get the path to an artifact file.

        Args:
            artifact_name: The name of the artifact (e.g., "transcribe", "translate")
            extension: File extension for the artifact (e.g., "mp3", "srt")

        Returns:
            Path to the artifact file
        """
        ...

    def store(self, artifact_name: str, data: bytes, extension: str) -> Path:
        """
        Store data in the cache.

        Args:
            artifact_name: The name of the artifact (e.g., "transcribe", "translate")
            extension: File extension for the artifact (e.g., "mp3", "srt")
            data: Data to store

        Returns:
            Path to the stored artifact file
        """
        ...

    def cleanup(self) -> None:
        """Clean up the cache by removing the temporary directory."""
        ...


class TempDirCache(Cache):
    """Temporary directory-based cache implementation for single pipeline runs."""

    def __init__(self, keep_files: bool = False):
        """
        Initialize a temporary directory cache.

        Args:
            keep_files: Whether to keep the temporary directory after cleanup
        """
        self.temp_dir = Path(tempfile.mkdtemp(prefix="audio2anki_"))
        self.keep_files = keep_files
        self.deck_path: Path | None = None
        logger.debug(f"Created temporary cache directory: {self.temp_dir}")

    def get_path(self, artifact_name: str, extension: str) -> Path:
        """
        Get the path to an artifact file.

        Args:
            artifact_name: The name of the artifact
            extension: File extension for the artifact (without dot)

        Returns:
            Path to the artifact file
        """
        if not extension.startswith("."):
            extension = f".{extension}"
        return self.temp_dir / f"{artifact_name}{extension}"

    def store(self, artifact_name: str, data: bytes, extension: str) -> Path:
        """
        Store data in the cache.

        Args:
            artifact_name: The name of the artifact
            extension: File extension for the artifact (with or without dot)
            data: Data to store

        Returns:
            Path to the stored artifact file
        """
        path = self.get_path(artifact_name, extension)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def cleanup(self) -> None:
        """
        Clean up the cache by removing the temporary directory.

        If keep_files is True, the directory is not removed.
        """
        if self.keep_files:
            logger.info(f"Keeping temporary cache directory: {self.temp_dir}")
            return

        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.debug(f"Removed temporary cache directory: {self.temp_dir}")
            except Exception as e:
                logger.error(f"Failed to clean up temporary directory: {e}")


# Global cache instance
_cache: Cache | None = None


def init_cache(keep_files: bool = False) -> Cache:
    """
    Initialize the cache system with a new temporary directory.

    Args:
        keep_files: Whether to keep the temporary files after cleanup

    Returns:
        The initialized cache instance
    """
    global _cache
    _cache = TempDirCache(keep_files=keep_files)
    return _cache


def get_cache() -> Cache:
    """
    Get the current cache instance, initializing if needed.

    Returns:
        The current cache instance
    """
    if _cache is None:
        return init_cache()
    return _cache


def get_artifact_path(artifact_name: str, extension: str) -> Path:
    """
    Get the path to an artifact file.

    Args:
        artifact_name: The name of the artifact
        extension: File extension for the artifact

    Returns:
        Path to the artifact file
    """
    return get_cache().get_path(artifact_name, extension)


def store_artifact(artifact_name: str, data: bytes, extension: str) -> Path:
    """
    Store an artifact in the cache.

    Args:
        artifact_name: The name of the artifact
        data: Data to store
        extension: File extension for the artifact

    Returns:
        Path to the stored artifact file
    """
    return get_cache().store(artifact_name, data, extension)


def cleanup_cache() -> None:
    """
    Clean up the cache by removing the temporary directory.

    This function should always be called at the end of a pipeline run,
    but will respect the keep_files flag set during initialization.
    """
    global _cache
    if _cache is not None:
        try:
            _cache.cleanup()
        finally:
            # Reset the cache reference to ensure it can be properly garbage collected
            _cache = None

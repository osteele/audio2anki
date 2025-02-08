"""Cache module for storing and retrieving processed files."""

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Protocol, TypedDict

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.expanduser("~/.cache/audio2anki")
METADATA_FILE = "metadata.json"


class CacheEntry(TypedDict):
    """Type definition for cache metadata entries."""

    version: int
    params: dict[str, Any] | None


class Cache(Protocol):
    """Protocol for cache implementations."""

    def retrieve(
        self, key: str, input_path: str | Path, suffix: str, version: int, extra_params: dict[str, Any] | None = None
    ) -> bool:
        """
        Check if a cached result exists.

        Args:
            key: Cache key for the operation
            input_path: Path to input file
            suffix: File suffix for the cached file
            version: Version number for cache invalidation
            extra_params: Additional parameters that affect the output

        Returns:
            True if cache hit, False if miss
        """
        ...

    def store(
        self, key: str, input_path: str | Path, data: bytes, suffix: str, extra_params: dict[str, Any] | None = None
    ) -> str:
        """
        Store data in the cache.

        Args:
            key: Cache key for the operation
            input_path: Path to input file
            data: Data to store
            suffix: File suffix for the cached file
            extra_params: Additional parameters that affect the output

        Returns:
            Path to the cached file
        """
        ...

    def get_path(self, key: str, file_hash: str, suffix: str) -> str:
        """
        Get the path to a cached file.

        Args:
            key: Cache key for the operation
            file_hash: Hash of the input file
            suffix: File suffix for the cached file

        Returns:
            Path to the cached file
        """
        ...

    def clear(self) -> None:
        """Clear all cached files and metadata."""
        ...


class FileCache(Cache):
    """File-based cache implementation."""

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.cache_dir / METADATA_FILE
        self.metadata: dict[str, CacheEntry] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load cache metadata from disk."""
        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            self._save_metadata()

    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f)

    def retrieve(
        self, key: str, input_path: str | Path, suffix: str, version: int, extra_params: dict[str, Any] | None = None
    ) -> bool:
        input_path = Path(input_path)
        file_hash = compute_file_hash(input_path)
        cache_key = f"{key}:{file_hash}"

        if cache_key not in self.metadata:
            return False

        entry = self.metadata[cache_key]
        if entry["version"] != version:
            return False

        if extra_params and entry.get("params") != extra_params:
            return False

        cache_path = self.cache_dir / f"{key}_{file_hash}{suffix}"
        return cache_path.exists()

    def store(
        self, key: str, input_path: str | Path, data: bytes, suffix: str, extra_params: dict[str, Any] | None = None
    ) -> str:
        input_path = Path(input_path)
        file_hash = compute_file_hash(input_path)
        cache_key = f"{key}:{file_hash}"

        # Store metadata
        entry: CacheEntry = {
            "version": 1,  # Default version
            "params": extra_params,
        }
        self.metadata[cache_key] = entry
        self._save_metadata()

        # Store file
        cache_path = self.cache_dir / f"{key}_{file_hash}{suffix}"
        with open(cache_path, "wb") as f:
            f.write(data)

        return str(cache_path)

    def get_path(self, key: str, file_hash: str, suffix: str) -> str:
        """Get the path to a cached file."""
        return str(self.cache_dir / f"{key}_{file_hash}{suffix}")

    def clear(self) -> None:
        """Clear all cached files and metadata."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = {}
        self._save_metadata()
        logger.info(f"Cleared cache directory: {self.cache_dir}")


class DummyCache(Cache):
    """Cache implementation that always misses."""

    def retrieve(
        self, key: str, input_path: str | Path, suffix: str, version: int, extra_params: dict[str, Any] | None = None
    ) -> bool:
        """Always return False (cache miss)."""
        return False

    def store(
        self, key: str, input_path: str | Path, data: bytes, suffix: str, extra_params: dict[str, Any] | None = None
    ) -> str:
        """Store nothing and return a temporary path."""
        import tempfile

        temp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp.write(data)
        temp.close()
        return temp.name

    def get_path(self, key: str, file_hash: str, suffix: str) -> str:
        """Should never be called since retrieve always returns False."""
        raise RuntimeError("get_path called on DummyCache")

    def clear(self) -> None:
        """Do nothing."""
        pass


def compute_file_hash(file_path: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


# Global cache instance
_cache: Cache = FileCache()


def init_cache(bypass: bool = False) -> None:
    """Initialize the cache system."""
    global _cache
    _cache = DummyCache() if bypass else FileCache()


def cache_retrieve(
    key: str, input_path: str | Path, suffix: str, version: int, extra_params: dict[str, Any] | None = None
) -> bool:
    """Check if a cached result exists."""
    return _cache.retrieve(key, input_path, suffix, version, extra_params)


def cache_store(
    key: str, input_path: str | Path, data: bytes, suffix: str, extra_params: dict[str, Any] | None = None
) -> str:
    """Store data in the cache."""
    return _cache.store(key, input_path, data, suffix, extra_params)


def get_cache_path(key: str, file_hash: str, suffix: str) -> str:
    """Get the path to a cached file."""
    return _cache.get_path(key, file_hash, suffix)


def clear_cache() -> None:
    """Clear all cached files and metadata."""
    _cache.clear()

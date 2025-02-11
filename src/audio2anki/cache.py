"""Cache module for storing and retrieving processed files."""

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any, NoReturn, Protocol, TypedDict

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "audio2anki"
METADATA_FILE = "metadata.json"
CURRENT_SCHEMA_VERSION = 1  # Initial schema version


def _abort_on_newer_schema(found_version: int) -> NoReturn:
    """Abort execution when a newer schema version is found."""
    print(
        f"""
Cache schema version {found_version} is newer than the supported version {CURRENT_SCHEMA_VERSION}.
This usually means you've used a newer version of the software with this cache directory.

Options:
1. Clear the cache directory and continue
2. Abort execution

Please choose (1/2): """,
        end="",
        file=sys.stderr,
    )

    choice = input().strip()
    if choice == "1":
        clear_cache()
        sys.exit(0)  # Exit after clearing cache
    sys.exit(1)  # Exit if user chose not to clear cache


class CacheEntry(TypedDict):
    """Type definition for cache metadata entries."""

    version: int
    params: dict[str, Any] | None


class Cache(Protocol):
    """Protocol for cache implementations."""

    def retrieve(
        self, key: str, input_path: str | Path, suffix: str, extra_params: dict[str, Any] | None = None
    ) -> bool:
        """
        Check if a cached result exists.

        Args:
            key: Cache key for the operation
            input_path: Path to input file
            suffix: File suffix for the cached file
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
    """File-based cache implementation using sqlite3 for metadata storage."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "cache.db"
        self.connection = sqlite3.connect(self.db_path)

        # Create tables if they don't exist
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            );
            CREATE TABLE IF NOT EXISTS metadata (
                cache_key TEXT PRIMARY KEY,
                version INTEGER,
                params TEXT
            );
        """)

        # Check/initialize schema version
        cursor = self.connection.cursor()
        cursor.execute("SELECT version FROM schema_version")
        row = cursor.fetchone()

        if row is None:
            # New database, set current version
            cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (CURRENT_SCHEMA_VERSION,))
            self.connection.commit()
        else:
            found_version = row[0]
            if found_version > CURRENT_SCHEMA_VERSION:
                self.connection.close()
                _abort_on_newer_schema(found_version)
            # We don't handle older versions yet since this is the first version

    def retrieve(
        self, key: str, input_path: str | Path, suffix: str, extra_params: dict[str, Any] | None = None
    ) -> bool:
        input_path = Path(input_path)
        file_hash = compute_file_hash(input_path)
        cache_key = f"{key}:{file_hash}"

        cursor = self.connection.cursor()
        cursor.execute("SELECT version, params FROM metadata WHERE cache_key = ?", (cache_key,))
        row = cursor.fetchone()
        if row is None:
            return False
        db_version, db_params = row
        if db_version != 1:  # hardcode version 1 since it's the only version we support
            return False
        stored_params = None if db_params is None else json.loads(db_params)
        if stored_params != extra_params:
            return False

        cache_path = self.cache_dir / f"{key}_{file_hash}{suffix}"
        if not cache_path.exists():
            cursor.execute("DELETE FROM metadata WHERE cache_key = ?", (cache_key,))
            self.connection.commit()
            return False
        return True

    def store(
        self, key: str, input_path: str | Path, data: bytes, suffix: str, extra_params: dict[str, Any] | None = None
    ) -> str:
        input_path = Path(input_path)
        file_hash = compute_file_hash(input_path)
        cache_key = f"{key}:{file_hash}"

        cursor = self.connection.cursor()
        params_json = json.dumps(extra_params) if extra_params is not None else None
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (cache_key, version, params) VALUES (?, ?, ?)", (cache_key, 1, params_json)
        )
        self.connection.commit()

        cache_path = self.cache_dir / f"{key}_{file_hash}{suffix}"
        with open(cache_path, "wb") as f:
            f.write(data)

        return str(cache_path)

    def get_path(self, key: str, file_hash: str, suffix: str) -> str:
        """Get the path to a cached file."""
        return str(self.cache_dir / f"{key}_{file_hash}{suffix}")

    def clear(self) -> None:
        """Clear all cached files and metadata."""
        self.connection.close()
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                cache_key TEXT PRIMARY KEY,
                version INTEGER,
                params TEXT
            )
        """)
        self.connection.commit()
        logger.info(f"Cleared cache directory: {self.cache_dir}")


class DummyCache(Cache):
    """Cache implementation that always misses."""

    def retrieve(
        self, key: str, input_path: str | Path, suffix: str, extra_params: dict[str, Any] | None = None
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


def cache_retrieve(key: str, input_path: str | Path, suffix: str, extra_params: dict[str, Any] | None = None) -> bool:
    """Check if a cached result exists."""
    return _cache.retrieve(key, input_path, suffix, extra_params)


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
    logger.info("Cache cleared due to schema version mismatch")

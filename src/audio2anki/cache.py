"""Cache module for storing and retrieving processed files."""

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn, Protocol, TypedDict

logger = logging.getLogger(__name__)

# Default cache directory location
cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "audio2anki"
# Export the cache directory as a constant for external use
CACHE_DIR = cache_dir
METADATA_FILE = "metadata.json"
CURRENT_SCHEMA_VERSION = 1  # Initial schema version


def get_short_hash(file_hash: str) -> str:
    """Get a shortened version of the hash for display purposes."""
    return file_hash[:8]


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

    def retrieve(self, key: str, suffix: str, extra_params: dict[str, Any] | None = None) -> bool:
        """
        Check if a cached result exists.

        Args:
            key: Cache key for the operation (includes content hash)
            suffix: File suffix for the cached file
            extra_params: Additional parameters that affect the output

        Returns:
            True if cache hit, False if miss
        """
        ...

    def store(self, key: str, data: bytes, suffix: str, extra_params: dict[str, Any] | None = None) -> str:
        """
        Store data in the cache.

        Args:
            key: Cache key for the operation (includes content hash)
            data: Data to store
            suffix: File suffix for the cached file
            extra_params: Additional parameters that affect the output

        Returns:
            Path to the cached file
        """
        ...

    def get_path(self, key: str, suffix: str) -> str:
        """
        Get the path to a cached file.

        Args:
            key: Cache key for the operation (includes content hash)
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

    def __init__(self, cache_dir_path: Path | str = cache_dir):
        self.cache_dir = Path(cache_dir_path)
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

    def retrieve(self, key: str, suffix: str, extra_params: dict[str, Any] | None = None) -> bool:
        """
        Check if a cached result exists based on the cache key.

        Args:
            key: Cache key (includes content hash)
            suffix: File suffix
            extra_params: Additional parameters

        Returns:
            True if cache hit, False if miss
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT version, params FROM metadata WHERE cache_key = ?", (key,))
        row = cursor.fetchone()
        if row is None:
            return False

        db_version, db_params = row
        if db_version != 1:  # hardcode version 1 since it's the only version we support
            return False

        stored_params = None if db_params is None else json.loads(db_params)
        if stored_params != extra_params:
            return False

        cache_path = self.cache_dir / f"{key}{suffix}"
        if not cache_path.exists():
            cursor.execute("DELETE FROM metadata WHERE cache_key = ?", (key,))
            self.connection.commit()
            return False

        return True

    def store(self, key: str, data: bytes, suffix: str, extra_params: dict[str, Any] | None = None) -> str:
        """
        Store data in cache using the provided key.

        The key format is: {sanitized_input_name}_{artifact_name}_{hash}

        Args:
            key: Cache key (includes content hash)
            data: Data to store
            suffix: File suffix
            extra_params: Additional parameters

        Returns:
            Path to the cached file
        """
        cursor = self.connection.cursor()
        params_json = json.dumps(extra_params) if extra_params is not None else None
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (cache_key, version, params) VALUES (?, ?, ?)", (key, 1, params_json)
        )
        self.connection.commit()

        # Write the data to the cache file
        cache_path = self.cache_dir / f"{key}{suffix}"
        with open(cache_path, "wb") as f:
            f.write(data)

        return str(cache_path)

    def get_path(self, key: str, suffix: str) -> str:
        """
        Get the path to a cached file.

        The key format is: {sanitized_input_name}_{artifact_name}_{hash}.

        Args:
            key: Cache key (includes the content hash)
            suffix: File suffix (including the dot)

        Returns:
            Path to the cached file
        """
        return str(self.cache_dir / f"{key}{suffix}")

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


def compute_file_hash(file_path: str | Path) -> str:
    """Compute MD5 hash of a file."""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest()


def get_content_hash(file_paths: Sequence[Path | str]) -> str:
    """
    Compute a hash based on the content of one or more files.

    Args:
        file_paths: Sequence of file paths to compute hash from

    Returns:
        First 16 characters of the hash:
        - For a single file, it's the first 16 chars of the file's md5 hash
        - For multiple files, it's the first 16 chars of the md5 hash of the concatenated full hashes
    """
    if not file_paths:
        raise ValueError("No files provided to hash")

    # Compute all file hashes first
    file_hashes = [compute_file_hash(path) for path in file_paths]

    if len(file_paths) == 1:
        # For a single file, return the first 8 characters of its md5 hash
        return file_hashes[0][:16]

    # For multiple files, concatenate their full hashes and hash again
    combined_hash = hashlib.md5()
    for file_hash in file_hashes:
        combined_hash.update(file_hash.encode())

    # Return first 8 characters of the combined hash
    return combined_hash.hexdigest()[:16]


def get_cache_info() -> dict[str, Any]:
    """Get information about the cache.

    Returns:
        Dictionary containing cache information including:
        - size: Total size in bytes
        - file_count: Number of cached files
        - last_modified: Timestamp of most recent modification
    """
    if not cache_dir.exists():
        return {"size": 0, "file_count": 0, "last_modified": None}

    total_size = 0
    file_count = 0
    latest_mtime = 0

    for path in cache_dir.rglob("*"):
        if path.is_file():
            file_count += 1
            stats = path.stat()
            total_size += stats.st_size
            latest_mtime = max(latest_mtime, stats.st_mtime)

    return {
        "size": total_size,
        "file_count": file_count,
        "last_modified": datetime.fromtimestamp(latest_mtime) if latest_mtime > 0 else None,
    }


def open_cache_directory() -> tuple[bool, str]:
    """Open the cache directory in the system file explorer.

    Returns:
        tuple of (success, message)
    """
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        if sys.platform == "darwin":  # macOS
            subprocess.run(["open", str(cache_dir)], check=True)
        elif sys.platform == "win32":  # Windows
            subprocess.run(["explorer", str(cache_dir)], check=True)
        else:  # Linux and others
            subprocess.run(["xdg-open", str(cache_dir)], check=True)
        return True, f"Opened cache directory: {cache_dir}"
    except subprocess.CalledProcessError:
        return False, "Failed to open cache directory"
    except Exception as e:
        return False, f"Error opening cache directory: {e}"


# Global cache instance
_cache: Cache = FileCache()


def init_cache() -> None:
    """Initialize the cache system."""
    global _cache
    _cache = FileCache()


def set_cache_directory(directory: str | Path) -> None:
    """
    Set the cache directory and reinitialize the cache.

    This ensures that all cache operations use the new directory.

    Args:
        directory: The new cache directory path
    """
    global cache_dir
    cache_dir = Path(directory)
    cache_dir.mkdir(parents=True, exist_ok=True)
    init_cache()  # Reinitialize the cache with the new directory


def cache_retrieve(key: str, suffix: str, extra_params: dict[str, Any] | None = None) -> bool:
    """
    Check if a cached result exists.

    The key format is: {sanitized_input_name}_{artifact_name}_{hash}

    Args:
        key: Cache key (includes content hash)
        suffix: File suffix
        extra_params: Additional parameters

    Returns:
        True if cache hit, False if miss
    """
    return _cache.retrieve(key, suffix, extra_params)


def cache_store(key: str, data: bytes, suffix: str, extra_params: dict[str, Any] | None = None) -> str:
    """
    Store data in the cache.

    The key format is: {sanitized_input_name}_{artifact_name}_{hash}

    Args:
        key: Cache key (includes content hash)
        data: Data to store
        suffix: File suffix
        extra_params: Additional parameters

    Returns:
        Path to the cached file
    """
    return _cache.store(key, data, suffix, extra_params)


def get_cache_path(key: str, suffix: str) -> str:
    """
    Get the path to a cached file.

    Args:
        key: Cache key (includes content hash)
        suffix: File suffix

    Returns:
        Path to the cached file
    """
    return _cache.get_path(key, suffix)


def clear_cache() -> None:
    """Clear all cached files and metadata."""
    _cache.clear()
    logger.info("Cache cleared due to schema version mismatch")

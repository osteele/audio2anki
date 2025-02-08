"""Cache management for audio2anki pipeline stages."""

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.environ.get("HOME", "."), ".cache", "audio2anki")
METADATA_FILE = "metadata.json"


@dataclass
class CacheMetadata:
    """Metadata for a cached file."""

    created_at: float  # Unix timestamp
    stage_name: str
    input_hash: str
    extra_params: dict[str, Any] | None = None

    def is_expired(self, expiry_days: int) -> bool:
        """Check if the cache entry has expired."""
        expiry_time = self.created_at + (expiry_days * 24 * 60 * 60)  # Convert days to seconds
        return time.time() > expiry_time

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a dictionary for serialization."""
        return {
            "created_at": self.created_at,
            "stage_name": self.stage_name,
            "input_hash": self.input_hash,
            "extra_params": self.extra_params or {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheMetadata":
        """Create metadata from a dictionary."""
        return cls(
            created_at=data["created_at"],
            stage_name=data["stage_name"],
            input_hash=data["input_hash"],
            extra_params=data.get("extra_params"),
        )


def compute_file_hash(file_path: str | Path) -> str:
    """Compute a hash of a file's contents."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def init_cache() -> str:
    """Initialize the cache directory."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    metadata_path = os.path.join(CACHE_DIR, METADATA_FILE)
    if not os.path.exists(metadata_path):
        with open(metadata_path, "w") as f:
            json.dump({}, f)
    return CACHE_DIR


def get_cache_path(stage_name: str, input_hash: str, extension: str) -> str:
    """Get the full path for a cached file."""
    filename = f"{input_hash}_{stage_name}{extension}"
    return os.path.join(CACHE_DIR, filename)


def _load_metadata() -> dict[str, CacheMetadata]:
    """Load metadata for all cached files."""
    metadata_path = os.path.join(CACHE_DIR, METADATA_FILE)
    try:
        with open(metadata_path) as f:
            data = json.load(f)
            return {k: CacheMetadata.from_dict(v) for k, v in data.items()}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_metadata(metadata: dict[str, CacheMetadata]) -> None:
    """Save metadata for all cached files."""
    metadata_path = os.path.join(CACHE_DIR, METADATA_FILE)
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_f:
        json.dump({k: v.to_dict() for k, v in metadata.items()}, temp_f, indent=2)
    # Atomic write using rename
    os.replace(temp_f.name, metadata_path)


def cache_store(
    stage_name: str,
    input_file: str | Path,
    data: bytes,
    extension: str,
    extra_params: dict[str, Any] | None = None,
) -> str:
    """Store data in the cache and return the cache path."""
    input_hash = compute_file_hash(input_file)
    cache_path = get_cache_path(stage_name, input_hash, extension)

    # Write data atomically using a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_f:
        temp_f.write(data)
    os.replace(temp_f.name, cache_path)

    # Update metadata
    metadata = _load_metadata()
    metadata[cache_path] = CacheMetadata(
        created_at=time.time(),
        stage_name=stage_name,
        input_hash=input_hash,
        extra_params=extra_params,
    )
    _save_metadata(metadata)

    logger.info(f"Cached {stage_name} output at {cache_path}")
    return cache_path


def cache_retrieve(
    stage_name: str,
    input_file: str | Path,
    extension: str,
    expiry_days: int,
    extra_params: dict[str, Any] | None = None,
) -> bytes | None:
    """Retrieve cached data if it exists and is not expired."""
    input_hash = compute_file_hash(input_file)
    cache_path = get_cache_path(stage_name, input_hash, extension)

    metadata = _load_metadata()
    cache_meta = metadata.get(cache_path)

    if not os.path.exists(cache_path) or not cache_meta:
        return None

    if cache_meta.is_expired(expiry_days):
        logger.info(f"Cache expired for {stage_name}")
        cache_delete(cache_path)
        return None

    if extra_params and cache_meta.extra_params != extra_params:
        logger.info(f"Cache parameters changed for {stage_name}")
        return None

    try:
        with open(cache_path, "rb") as f:
            data = f.read()
        logger.info(f"Using cached {stage_name} output from {cache_path}")
        return data
    except OSError:
        logger.warning(f"Failed to read cache file {cache_path}")
        return None


def cache_delete(cache_path: str) -> None:
    """Delete a cached file and its metadata."""
    try:
        os.unlink(cache_path)
        metadata = _load_metadata()
        if cache_path in metadata:
            del metadata[cache_path]
            _save_metadata(metadata)
        logger.info(f"Deleted cache file {cache_path}")
    except OSError:
        logger.warning(f"Failed to delete cache file {cache_path}")


def expire_old_entries(expiry_days: int) -> None:
    """Remove cache entries older than the specified number of days."""
    metadata = _load_metadata()
    for cache_path, meta in list(metadata.items()):
        if meta.is_expired(expiry_days):
            cache_delete(cache_path)


def cache_size() -> int:
    """Return the total size of the cache in bytes, excluding the metadata file."""
    total_size = 0
    for dirpath, _, filenames in os.walk(CACHE_DIR):
        for f in filenames:
            if f != METADATA_FILE:  # Exclude metadata file from size calculation
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
    return total_size


def clear_cache() -> None:
    """Remove all cache files and metadata."""
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)
    # Create empty metadata file
    with open(os.path.join(CACHE_DIR, METADATA_FILE), "w") as f:
        json.dump({}, f)
    logger.info("Cache cleared")

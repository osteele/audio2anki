"""Persistent artifact cache for pipeline function outputs."""

import hashlib
import json
import logging
import os
import platform
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Use module-level logger
logger = logging.getLogger(__name__)


def try_hard_link(src: Path, dst: Path) -> bool:
    """
    Attempt to create a hard link between src and dst.

    Args:
        src: Source file path
        dst: Destination file path

    Returns:
        True if hard link was created, False if not supported or failed
    """
    # Check if we're on a system that supports hard links
    # Most Unix-like systems (macOS, Linux) support hard links
    if platform.system() in ("Darwin", "Linux"):
        try:
            # Remove destination if it exists
            if dst.exists():
                dst.unlink()

            # Create hard link
            os.link(src, dst)
            logger.debug(f"Created hard link from {src} to {dst}")
            return True
        except OSError as e:
            # Hard link failed (possibly across filesystems)
            logger.debug(f"Hard link failed: {e}. Will fall back to copy.")
            return False
    return False


class ArtifactCache:
    """Persistent cache for pipeline function artifacts with version-based invalidation."""

    def __init__(self, cache_dir: Path):
        """
        Initialize the artifact cache.

        Args:
            cache_dir: The base directory for the cache
        """
        self.cache_dir = cache_dir
        self.db_path = cache_dir / "cache.db"
        self.artifacts_dir = cache_dir / "artifacts"

        # Create cache directories if they don't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)

        # Initialize the database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database for cache metadata."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create artifacts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    artifact_name TEXT NOT NULL,
                    key TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed_at REAL NOT NULL
                )
            """)

            # Create stats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stats (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

            # Create index for faster lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_name_hash_key ON artifacts (artifact_name, input_hash, key)")

            # Check if we need to migrate from an older schema
            cursor.execute("PRAGMA table_info(artifacts)")
            columns = [info[1] for info in cursor.fetchall()]

            # If 'key' column doesn't exist but 'version' does, we need to migrate
            if "key" not in columns and "version" in columns:
                logger.info("Migrating database schema: adding 'key' column")
                try:
                    # Create a new table with the updated schema
                    cursor.execute("""
                        CREATE TABLE artifacts_new (
                            artifact_id TEXT PRIMARY KEY,
                            artifact_name TEXT NOT NULL,
                            key TEXT NOT NULL,
                            input_hash TEXT NOT NULL,
                            file_path TEXT NOT NULL,
                            created_at REAL NOT NULL,
                            last_accessed_at REAL NOT NULL
                        )
                    """)

                    # Copy data from old table to new table, using version as key
                    cursor.execute("""
                        INSERT INTO artifacts_new
                        SELECT artifact_id, artifact_name, version, input_hash, file_path, created_at, last_accessed_at
                        FROM artifacts
                    """)

                    # Drop the old table and rename the new one
                    cursor.execute("DROP TABLE artifacts")
                    cursor.execute("ALTER TABLE artifacts_new RENAME TO artifacts")

                    # Recreate the index
                    cursor.execute(
                        "CREATE INDEX IF NOT EXISTS idx_name_hash_key ON artifacts (artifact_name, input_hash, key)"
                    )

                    conn.commit()
                    logger.info("Database schema migration completed successfully")
                except Exception as e:
                    logger.error(f"Error during database migration: {e}")
                    # If migration fails, we'll recreate the database from scratch
                    cursor.execute("DROP TABLE IF EXISTS artifacts")
                    cursor.execute("DROP TABLE IF EXISTS artifacts_new")
                    cursor.execute("""
                        CREATE TABLE artifacts (
                            artifact_id TEXT PRIMARY KEY,
                            artifact_name TEXT NOT NULL,
                            key TEXT NOT NULL,
                            input_hash TEXT NOT NULL,
                            file_path TEXT NOT NULL,
                            created_at REAL NOT NULL,
                            last_accessed_at REAL NOT NULL
                        )
                    """)
                    conn.commit()
                    logger.info("Created new database schema after migration failure")

            # Also update the index if needed
            if "key" in columns and "version" not in columns:
                cursor.execute("DROP INDEX IF EXISTS idx_name_hash_version")
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_name_hash_key ON artifacts (artifact_name, input_hash, key)"
                )

    def _compute_input_hash(self, inputs: dict[str, Any]) -> str:
        """
        Compute a hash of the input values to identify cache entries.

        Args:
            inputs: Dictionary of input parameters to hash

        Returns:
            String hash of the inputs
        """
        # Convert paths to strings
        serializable_inputs = {}

        logger.debug(f"Computing hash for inputs: {list(inputs.keys())}")

        for k, v in inputs.items():
            if isinstance(v, Path):
                # For Path objects, hash the file contents
                if v.exists() and v.is_file():
                    try:
                        file_hash = self._hash_file(v)
                        logger.debug(f"Hashed file {v} to {file_hash[:8]}...")
                        serializable_inputs[k] = {"type": "file", "hash": file_hash, "path": str(v)}
                    except Exception as e:
                        logger.warning(f"Could not hash file {v}: {e}")
                        serializable_inputs[k] = {"type": "path", "value": str(v)}
                else:
                    logger.debug(f"Path {v} doesn't exist or is not a file, using string representation")
                    serializable_inputs[k] = {"type": "path", "value": str(v)}
            else:
                # For other values, just store them directly
                logger.debug(f"Input {k} is not a Path, using direct value")
                serializable_inputs[k] = {"type": "value", "value": v}

        # Create a stable string representation for hashing
        input_str = json.dumps(serializable_inputs, sort_keys=True)
        hash_result = hashlib.sha256(input_str.encode()).hexdigest()
        logger.debug(f"Final input hash: {hash_result[:8]}...")

        return hash_result

    def _hash_file(self, file_path: Path) -> str:
        """
        Compute a hash of a file's contents.

        Args:
            file_path: Path to the file to hash

        Returns:
            SHA-256 hash of the file contents
        """
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_artifact_path(
        self, artifact_name: str, key: str, inputs: dict[str, Any], extension: str
    ) -> tuple[Path | None, bool]:
        """
        Check if an artifact exists in the cache and return its path.

        Args:
            artifact_name: Name of the artifact
            key: Key of the artifact
            inputs: Dictionary of input parameters
            extension: File extension for the artifact

        Returns:
            Tuple of (Path to the artifact if found, otherwise None, boolean indicating cache hit)
        """
        logger.debug(f"Checking cache for artifact: {artifact_name} (key {key})")

        input_hash = self._compute_input_hash(inputs)
        artifact_id = f"{artifact_name}_{key}_{input_hash}"

        try:
            inputs_json = json.dumps({k: str(v) for k, v in inputs.items()}, sort_keys=True)
        except Exception:
            inputs_json = str(inputs)
        logger.debug(
            f"[LOOKUP] artifact_id={artifact_id}, inputs={inputs_json}, input_hash={input_hash}, "
            f"key={key}, artifact_name={artifact_name}"
        )

        logger.debug(f"Looking for artifact_id: {artifact_id}")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM artifacts WHERE artifact_id = ?", (artifact_id,))
            row = cursor.fetchone()

            if row:
                file_path = Path(row[0])
                logger.debug(f"Found record in database for {artifact_id}, path: {file_path}")

                if file_path.exists():
                    logger.debug(f"File exists at {file_path}, cache HIT")
                    # Update last accessed time
                    current_time = time.time()
                    cursor.execute(
                        "UPDATE artifacts SET last_accessed_at = ? WHERE artifact_id = ?", (current_time, artifact_id)
                    )
                    conn.commit()

                    return file_path, True
                else:
                    logger.debug(f"File doesn't exist at {file_path}, cache MISS despite DB record")
            else:
                logger.debug(f"No record found in database for {artifact_id}")

        # If we get here, either no record found or file doesn't exist
        # Calculate the path where the artifact would be stored
        if not extension.startswith("."):
            extension = f".{extension}"

        file_path = self.artifacts_dir / f"{artifact_id}{extension}"
        logger.debug(f"Returning new artifact path: {file_path} (cache MISS)")
        logger.debug(f"[LOOKUP-RESULT] artifact_id={artifact_id}, cache_hit=False, file_path={file_path}")
        return file_path, False

    def store_artifact(
        self, artifact_name: str, key: str, inputs: dict[str, Any], data_path: Path, extension: str
    ) -> Path:
        """
        Store an artifact in the cache.

        Args:
            artifact_name: Name of the artifact
            key: Key of the artifact
            inputs: Dictionary of input parameters
            data_path: Path to the file to store
            extension: File extension for the artifact

        Returns:
            Path to the stored artifact
        """
        logger.debug(f"Storing artifact {artifact_name} (key {key}) from {data_path}")

        input_hash = self._compute_input_hash(inputs)
        artifact_id = f"{artifact_name}_{key}_{input_hash}"

        try:
            inputs_json = json.dumps({k: str(v) for k, v in inputs.items()}, sort_keys=True)
        except Exception:
            inputs_json = str(inputs)
        logger.debug(
            f"[STORE] artifact_id={artifact_id}, inputs={inputs_json}, input_hash={input_hash}, key={key}, "
            f"artifact_name={artifact_name}"
        )

        logger.debug(f"Generated artifact_id: {artifact_id}")

        if not extension.startswith("."):
            extension = f".{extension}"

        # Create the destination path
        dest_path = self.artifacts_dir / f"{artifact_id}{extension}"
        logger.debug(f"Destination path: {dest_path}")

        # If the file already exists at the destination, just update the metadata
        if dest_path.exists():
            if dest_path.samefile(data_path):
                logger.debug("File already exists at destination and is the same file, updating metadata")
                current_time = time.time()
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        UPDATE artifacts
                        SET last_accessed_at = ?
                        WHERE artifact_id = ?
                        """,
                        (current_time, artifact_id),
                    )
                    conn.commit()
                return dest_path
            else:
                logger.debug("File exists at destination but is a different file, will be replaced")

        # Try to create a hard link
        if try_hard_link(data_path, dest_path):
            logger.debug(f"Stored artifact at {dest_path} using hard link")
        else:
            # Fallback to regular copying
            try:
                logger.debug(f"Copying file from {data_path} to {dest_path}")
                shutil.copy2(data_path, dest_path)
                logger.debug(f"Stored artifact at {dest_path} using copy")
            except Exception as e:
                logger.error(f"Failed to copy artifact to cache: {e}")
                raise

        # Store metadata in the database
        current_time = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO artifacts
                (artifact_id, artifact_name, key, input_hash, file_path, created_at, last_accessed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (artifact_id, artifact_name, key, input_hash, str(dest_path), current_time, current_time),
            )
            conn.commit()
            logger.debug(f"Stored metadata in database for {artifact_id}")

        # Verify the file was properly stored
        if dest_path.exists():
            logger.debug(f"Successfully stored artifact at {dest_path} ({dest_path.stat().st_size} bytes)")
        else:
            logger.error(f"Failed to confirm artifact exists at {dest_path}")
        logger.debug(f"[STORE-RESULT] artifact_id={artifact_id}, file_path={dest_path}")
        return dest_path

    def clear_cache(self) -> tuple[int, int]:
        """
        Clear all artifacts from the cache.

        Returns:
            Tuple of (number of files removed, total bytes freed)
        """
        # Get all file paths from the database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM artifacts")
            rows = cursor.fetchall()

            # Delete all records
            cursor.execute("DELETE FROM artifacts")
            conn.commit()

        # Delete all files
        files_removed = 0
        bytes_freed = 0

        for row in rows:
            file_path = Path(row[0])
            if file_path.exists():
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    files_removed += 1
                    bytes_freed += file_size
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {file_path}: {e}")

        return files_removed, bytes_freed

    def clean_old_artifacts(self, days: int = 14) -> tuple[int, int]:
        """
        Remove artifacts that haven't been accessed for the specified number of days.

        Args:
            days: Number of days since last access before removing an artifact

        Returns:
            Tuple of (number of files removed, total bytes freed)
        """
        cutoff_time = time.time() - (days * 86400)  # 86400 seconds in a day

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get all artifacts older than the cutoff
            cursor.execute("SELECT artifact_id, file_path FROM artifacts WHERE last_accessed_at < ?", (cutoff_time,))
            old_artifacts = cursor.fetchall()

            files_removed = 0
            bytes_freed = 0

            for artifact_id, file_path in old_artifacts:
                file_path_obj = Path(file_path)
                if file_path_obj.exists():
                    try:
                        file_size = file_path_obj.stat().st_size
                        file_path_obj.unlink()
                        files_removed += 1
                        bytes_freed += file_size
                    except Exception as e:
                        logger.warning(f"Failed to delete old cache file {file_path}: {e}")

                # Delete the record even if file deletion failed
                cursor.execute("DELETE FROM artifacts WHERE artifact_id = ?", (artifact_id,))

            conn.commit()

        return files_removed, bytes_freed

    def get_cache_size(self) -> tuple[int, int]:
        """
        Calculate the total size of cached artifacts.

        Returns:
            Tuple of (number of files, total size in bytes)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM artifacts")
            file_count = cursor.fetchone()[0]

        total_size = 0
        for item in self.artifacts_dir.glob("*"):
            if item.is_file():
                total_size += item.stat().st_size

        return file_count, total_size

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get statistics about the cache.

        Returns:
            Dictionary with cache statistics
        """
        file_count, total_size = self.get_cache_size()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get oldest and newest artifact creation dates
            cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM artifacts")
            oldest_time, newest_time = cursor.fetchone()

            # Get artifact count by name
            cursor.execute("SELECT artifact_name, COUNT(*) FROM artifacts GROUP BY artifact_name")
            artifact_counts = dict(cursor.fetchall())

            oldest_date = datetime.fromtimestamp(oldest_time) if oldest_time else None
            newest_date = datetime.fromtimestamp(newest_time) if newest_time else None

        return {
            "file_count": file_count,
            "total_size_bytes": total_size,
            "total_size_human": self._format_size(total_size),
            "oldest_artifact": oldest_date.isoformat() if oldest_date else None,
            "newest_artifact": newest_date.isoformat() if newest_date else None,
            "artifact_counts": artifact_counts,
            "cache_path": str(self.cache_dir),
        }

    def _format_size(self, size_bytes: int) -> str:
        """Convert bytes to a human-readable format."""
        size = float(size_bytes)  # Convert to float for division
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024 or unit == "GB":
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} GB"  # Fallback return for completeness


# Global instance
_artifact_cache: ArtifactCache | None = None


def get_cache_dir() -> Path:
    """Get the standard cache directory for the application."""
    if os.name == "nt":  # Windows
        base_dir = os.environ.get("LOCALAPPDATA", os.path.expanduser("~\\AppData\\Local"))
        cache_dir = Path(base_dir) / "audio2anki" / "cache"
    elif sys.platform == "darwin":  # macOS
        base_dir = os.path.expanduser("~/Library/Caches")
        cache_dir = Path(base_dir) / "audio2anki"
    else:  # Linux, etc.
        base_dir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        cache_dir = Path(base_dir) / "audio2anki"

    return cache_dir


def get_artifact_cache() -> ArtifactCache:
    """
    Get the global artifact cache instance, initializing if needed.

    Returns:
        The artifact cache instance
    """
    global _artifact_cache
    if _artifact_cache is None:
        cache_dir = get_cache_dir()
        _artifact_cache = ArtifactCache(cache_dir)

    return _artifact_cache


def get_cached_artifact(
    artifact_name: str, key: str, inputs: dict[str, Any], extension: str
) -> tuple[Path | None, bool]:
    """
    Get an artifact from the cache if it exists.

    Args:
        artifact_name: Name of the artifact
        key: Key of the artifact
        inputs: Dictionary of input parameters
        extension: File extension for the artifact

    Returns:
        Tuple of (Path to the artifact if found, otherwise None, boolean indicating cache hit)
    """
    return get_artifact_cache().get_artifact_path(artifact_name, key, inputs, extension)


def store_artifact(artifact_name: str, key: str, inputs: dict[str, Any], data_path: Path, extension: str) -> Path:
    """
    Store an artifact in the cache.

    Args:
        artifact_name: Name of the artifact
        key: Key of the artifact
        inputs: Dictionary of input parameters
        data_path: Path to the file to store
        extension: File extension for the artifact

    Returns:
        Path to the stored artifact
    """
    return get_artifact_cache().store_artifact(artifact_name, key, inputs, data_path, extension)


def clear_cache() -> tuple[int, int]:
    """
    Clear all artifacts from the cache.

    Returns:
        Tuple of (number of files removed, total bytes freed)
    """
    return get_artifact_cache().clear_cache()


def clean_old_artifacts(days: int = 14) -> tuple[int, int]:
    """
    Remove artifacts that haven't been accessed for the specified number of days.

    Args:
        days: Number of days since last access before removing an artifact

    Returns:
        Tuple of (number of files removed, total bytes freed)
    """
    return get_artifact_cache().clean_old_artifacts(days)


def get_cache_stats() -> dict[str, Any]:
    """
    Get statistics about the cache.

    Returns:
        Dictionary with cache statistics
    """
    return get_artifact_cache().get_cache_stats()

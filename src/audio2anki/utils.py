"""Utility functions for audio2anki."""

import re
import unicodedata
from pathlib import Path


def format_bytes(size_bytes: int) -> str:
    """Convert bytes to a human-readable format."""
    size = float(size_bytes)  # Convert to float for division
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} GB"  # Fallback return for completeness


def sanitize_filename(filename: str, max_length: int = 32) -> str:
    """
    Sanitize a filename by removing unsafe characters and limiting length.

    Args:
        filename: The filename to sanitize
        max_length: Maximum length for the filename

    Returns:
        A sanitized filename
    """
    # Normalize unicode characters but preserve diacritics
    filename = unicodedata.normalize("NFC", filename)

    # Remove path separators and keep only the basename
    filename = Path(filename).name

    # Replace unsafe characters with underscores
    filename = re.sub(r"[^\w\s.-]", "_", filename)

    # Replace spaces with underscores and collapse multiple underscores
    filename = re.sub(r"[\s_]+", "_", filename)

    # Remove leading and trailing underscores
    filename = filename.strip("_")

    # If filename is empty after sanitization, use a default name
    if not filename:
        filename = "unnamed"

    # Trim to max length while preserving extension if possible
    if len(filename) > max_length:
        name, ext = Path(filename).stem, Path(filename).suffix

        # Calculate available space for the name part
        available_space = max_length - len(ext) - 3  # 3 for ellipsis
        if available_space < 10:  # Ensure we have reasonable space for the name
            available_space = max_length - 3
            ext = ""

        # Truncate the name and add ellipsis
        prefix_length = available_space - 3  # Leave room for "..."
        if prefix_length < 5:  # Ensure we have at least a few characters
            prefix_length = 5

        truncated_name = name[:prefix_length] + "..."
        filename = truncated_name + ext

    return filename


def is_metadata_file(path: Path) -> bool:
    """Check if a file is a system metadata file that should be ignored when determining if a directory is empty."""
    # macOS metadata files
    macos_patterns = {".DS_Store", ".AppleDouble", ".LSOverride"}
    if path.name.startswith("._"):  # macOS resource fork files
        return True

    # Windows metadata files
    windows_patterns = {"Thumbs.db", "ehthumbs.db", "Desktop.ini", "$RECYCLE.BIN"}

    # Linux metadata files
    linux_patterns = {".directory"}
    if path.name.startswith(".Trash-"):
        return True

    # IDE/Editor metadata files
    ide_patterns = {".idea", ".vscode"}
    if path.name.endswith((".swp", "~")) or path.name.startswith(".") and path.name.endswith(".swp"):
        return True

    return path.name in macos_patterns | windows_patterns | linux_patterns | ide_patterns


def is_empty_directory(path: Path) -> bool:
    """Check if a directory is empty, ignoring system metadata files."""
    if not path.is_dir():
        return False

    return all(is_metadata_file(f) for f in path.iterdir())


def is_deck_folder(path: Path) -> bool:
    """Check if a directory is a valid deck folder."""
    import logging

    logger = logging.getLogger("audio2anki")

    logger.debug(f"Checking if {path} is a deck folder")

    if not path.is_dir():
        logger.debug(f"{path} is not a directory")
        return False

    # Required deck files
    required_files = {"deck.txt", "README.md"}
    # At least one CSV file should exist
    has_csv = False
    # Should have a media directory
    has_media_dir = False
    # Track what files we found
    found_files: set[str] = set()

    for item in path.iterdir():
        found_files.add(f"{item.name} ({'dir' if item.is_dir() else 'file'})")

        if is_metadata_file(item):
            continue

        if item.name in required_files:
            required_files.remove(item.name)
        elif item.suffix == ".csv":
            has_csv = True
        elif item.is_dir() and item.name == "media":
            has_media_dir = True
        elif item.name == "import_to_anki.sh" or item.name == "anki_media":
            # These are known additional files that are allowed
            pass
        else:
            # Unknown file found
            logger.debug(f"Found unknown file/dir '{item.name}' in {path}, not a deck folder")
            return False

    result = len(required_files) == 0 and has_csv and has_media_dir
    if not result:
        logger.debug(f"Directory {path} contains: {', '.join(list(found_files))}")
        logger.debug(
            f"Missing requirements for deck folder: required_files={required_files}, has_csv={has_csv}, "
            f"has_media_dir={has_media_dir}"
        )
    else:
        logger.debug(f"Confirmed {path} is a valid deck folder")

    return result

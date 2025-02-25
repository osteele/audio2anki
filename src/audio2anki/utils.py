"""Utility functions for audio2anki."""

import re
import unicodedata
from pathlib import Path


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

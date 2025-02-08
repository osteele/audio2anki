"""Audio transcoding module using pydub."""

import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydub import AudioSegment

from . import cache

logger = logging.getLogger(__name__)


def get_output_path(input_path: str | Path, suffix: str = ".mp3") -> Path:
    """Generate output path for transcoded audio file."""
    input_path = Path(input_path)
    return input_path.with_suffix(suffix)


def transcode_audio(
    input_path: str | Path,
    progress_callback: Callable[[float], None] | None = None,
    target_format: Literal["mp3"] = "mp3",
    target_channels: int = 1,
    target_sample_rate: int = 44100,
    target_bitrate: str = "128k",
) -> Path:
    """
    Transcode an audio file to a standardized format suitable for processing.

    Args:
        input_path: Path to input audio/video file
        progress_callback: Optional callback function to report progress
        target_format: Output audio format (default: mp3)
        target_channels: Number of audio channels (default: 1 for mono)
        target_sample_rate: Sample rate in Hz (default: 44100)
        target_bitrate: Output bitrate (default: 128k)

    Returns:
        Path to the transcoded audio file
    """

    def update_progress(percent: float) -> None:
        """Update progress if callback is provided."""
        if progress_callback:
            progress_callback(percent)

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Check cache first
    extra_params = {
        "target_channels": target_channels,
        "target_sample_rate": target_sample_rate,
        "target_bitrate": target_bitrate,
    }
    update_progress(0)  # Start progress
    cached_result = cache.cache_retrieve("transcode", input_path, f".{target_format}", 7, extra_params)
    if cached_result:
        # Get the path from the cache metadata
        cached_path = cache.get_cache_path("transcode", cache.compute_file_hash(input_path), f".{target_format}")
        logger.info(f"Using cached transcoded file: {cached_path}")
        update_progress(100)  # Cache hit is immediate completion
        return Path(cached_path)

    try:
        # Load the audio file
        logger.info(f"Loading audio file: {input_path}")
        update_progress(10)  # Loading started

        audio = AudioSegment.from_file(str(input_path))
        update_progress(30)  # Loading complete

        # Apply audio transformations
        if audio.channels != target_channels:
            audio = audio.set_channels(target_channels)
            update_progress(40)

        if audio.frame_rate != target_sample_rate:
            audio = audio.set_frame_rate(target_sample_rate)
            update_progress(50)

        update_progress(60)  # Processing complete

        # Export to a temporary file first
        with tempfile.NamedTemporaryFile(suffix=f".{target_format}", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

            # Export the processed audio
            logger.info(f"Exporting processed audio to temporary file: {temp_path}")
            export_params: dict[str, Any] = {
                "format": target_format,
                "parameters": ["-b:a", target_bitrate],  # Pass bitrate as ffmpeg parameter
            }
            if target_format == "mp3":
                export_params["id3v2_version"] = "3"  # Use ID3v2.3 for better compatibility
            audio.export(str(temp_path), **export_params)

            update_progress(80)  # Export complete

            # Store in cache
            with open(temp_path, "rb") as f:
                cached_path = cache.cache_store(
                    "transcode",
                    input_path,
                    f.read(),
                    f".{target_format}",
                    extra_params,
                )

            update_progress(90)  # Cache storage complete

        # Clean up temporary file
        os.unlink(temp_path)

        update_progress(100)  # All done
        return Path(cached_path)

    except Exception as e:
        logger.error(f"Error transcoding audio file: {e}")
        raise

"""Audio transcoding module using pydub."""

import logging
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydub import AudioSegment

logger = logging.getLogger(__name__)


def get_output_path(input_path: str | Path, suffix: str = ".mp3") -> Path:
    """Generate output path for transcoded audio file."""
    input_path = Path(input_path)
    return input_path.with_suffix(suffix)


def transcode_audio(
    input_path: Path,
    output_path: Path,
    progress_callback: Callable[[float], None] | None = None,
    target_format: Literal["mp3"] = "mp3",
    target_channels: int = 1,
    target_sample_rate: int = 44100,
    target_bitrate: str = "128k",
) -> None:
    """Transcode an audio file to a standardized format."""

    def update_progress(percent: float) -> None:
        if progress_callback:
            progress_callback(percent)

    try:
        # Load the audio file
        logger.debug(f"Loading audio file: {input_path}")
        update_progress(10)

        audio = AudioSegment.from_file(str(input_path))
        update_progress(30)

        # Apply audio transformations
        if audio.channels != target_channels:
            audio = audio.set_channels(target_channels)
            update_progress(40)

        if audio.frame_rate != target_sample_rate:
            audio = audio.set_frame_rate(target_sample_rate)
            update_progress(50)

        update_progress(60)

        # Export to a temporary file first
        with tempfile.NamedTemporaryFile(suffix=f".{target_format}", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

            # Export the processed audio
            logger.debug(f"Exporting processed audio to temporary file: {temp_path}")
            export_params: dict[str, Any] = {
                "format": target_format,
                "parameters": ["-b:a", target_bitrate],
            }
            if target_format == "mp3":
                export_params["id3v2_version"] = "3"
            audio.export(str(temp_path), **export_params)

            update_progress(80)

            # Move temporary file to final location
            shutil.move(temp_path, output_path)

            update_progress(100)

    except Exception as e:
        logger.error(f"Error transcoding audio file: {e}")
        raise

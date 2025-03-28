"""Audio transcoding module using pydub."""

import hashlib
import json
import logging
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypedDict

from pydub import AudioSegment

logger = logging.getLogger(__name__)


class TranscodingParams(TypedDict):
    target_format: Literal["mp3"]
    target_channels: int
    target_sample_rate: int
    target_bitrate: str
    library_version: str


# Default transcoding parameters - used by both the transcoding function and version calculation
DEFAULT_TRANSCODING_PARAMS: TranscodingParams = {
    "target_format": "mp3",
    "target_channels": 1,
    "target_sample_rate": 44100,
    "target_bitrate": "128k",
    "library_version": getattr(AudioSegment, "__version__", "1.0.0"),
}


def get_output_path(input_path: str | Path, suffix: str = ".mp3") -> Path:
    """Generate output path for transcoded audio file."""
    input_path = Path(input_path)
    return input_path.with_suffix(suffix)


def get_transcode_version() -> int:
    """
    Generate a version number for the transcoding function based on its current parameters.

    This creates a hash of the default transcoding parameters, ensuring cached artifacts
    are invalidated if the default implementation changes.

    Returns:
        An integer version derived from the hash of parameters
    """
    # Use the default parameter values stored in DEFAULT_TRANSCODING_PARAMS
    params: TranscodingParams = DEFAULT_TRANSCODING_PARAMS.copy()

    # Create a stable string representation for hashing
    param_str = json.dumps(params, sort_keys=True)

    # Hash the parameters and convert to an integer
    hash_obj = hashlib.sha256(param_str.encode())
    # Use the first 4 bytes of the hash as an integer
    version = abs(int.from_bytes(hash_obj.digest()[:4], byteorder="big"))

    return version


def transcode_audio(
    input_path: Path,
    output_path: Path,
    progress_callback: Callable[[float], None] | None = None,
    target_format: Literal["mp3"] = DEFAULT_TRANSCODING_PARAMS["target_format"],
    target_channels: int = DEFAULT_TRANSCODING_PARAMS["target_channels"],
    target_sample_rate: int = DEFAULT_TRANSCODING_PARAMS["target_sample_rate"],
    target_bitrate: str = DEFAULT_TRANSCODING_PARAMS["target_bitrate"],
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

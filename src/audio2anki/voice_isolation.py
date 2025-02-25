"""Voice isolation using Eleven Labs API."""

import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import httpx
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)

API_BASE_URL = "https://api.elevenlabs.io/v1"


class VoiceIsolationError(Exception):
    """Error during voice isolation."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause
        self.error_message = message  # Store message in an attribute that won't conflict with Exception

    def __str__(self) -> str:
        cause_str = f": {self.cause}" if self.cause else ""
        return f"Voice Isolation Error{cause_str}: {self.error_message}"


def _call_elevenlabs_api(input_path: Path, progress_callback: Callable[[float], None]) -> Path:
    """
    Call Eleven Labs API to isolate voice from background noise.

    Args:
        input_path: Path to input audio file
        progress_callback: Optional callback function to report progress

    Returns:
        Path to the raw isolated voice audio file from the API

    Raises:
        VoiceIsolationError: If API call fails
    """

    def update_progress(percent: float) -> None:
        progress_callback(percent * 0.7)  # Scale to 70% of total progress

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise VoiceIsolationError(
            "ELEVENLABS_API_KEY environment variable not set. Get your API key from https://elevenlabs.io"
        )

    try:
        url = f"{API_BASE_URL}/audio-isolation/stream"
        headers = {"xi-api-key": api_key, "accept": "application/json"}

        logger.debug("Uploading audio file to Eleven Labs API")
        update_progress(10)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

            with open(input_path, "rb") as f:
                files = {"audio": (input_path.name, f, "audio/mpeg")}
                with httpx.Client(timeout=60.0) as client:
                    with client.stream("POST", url, headers=headers, files=files) as response:
                        if response.status_code != 200:
                            try:
                                error_data = response.json()
                                error_msg = error_data.get("detail", "API error message")
                            except Exception:
                                error_msg = f"API request failed: {response.status_code}"
                            raise VoiceIsolationError(error_msg) from None

                        logger.debug("Streaming isolated audio from API")
                        update_progress(30)

                        total_chunks = 0
                        for chunk in response.iter_bytes():
                            if not chunk:
                                continue
                            temp_file.write(chunk)
                            total_chunks += 1
                            if total_chunks % 10 == 0:
                                update_progress(30 + (total_chunks % 20))

                        temp_file.flush()
                        os.fsync(temp_file.fileno())

            if total_chunks == 0:
                raise VoiceIsolationError("No audio data received from API") from None

            update_progress(70)
            return Path(temp_path)

    except httpx.TimeoutException as err:
        raise VoiceIsolationError("API request timed out", cause=err) from err
    except httpx.RequestError as err:
        raise VoiceIsolationError(f"API request failed: {err}", cause=err) from err


def _match_audio_properties(source_path: Path, target_path: Path, progress_callback: Callable[[float], None]) -> None:
    """
    Match the duration of the target audio to the source audio by adjusting sample rate.

    Args:
        source_path: Path to source audio file (original)
        target_path: Path to target audio file (to be adjusted)
        progress_callback: Optional callback function to report progress

    Raises:
        VoiceIsolationError: If audio adjustment fails
    """

    def update_progress(percent: float) -> None:
        progress_callback(70 + percent * 0.3)  # Scale remaining 30% of progress

    # Load durations
    source_duration = librosa.get_duration(path=str(source_path))
    # First load without resampling to get the raw samples
    y, original_sr = librosa.load(str(target_path), sr=None)
    target_duration = librosa.get_duration(y=y, sr=original_sr)

    logger.debug(f"Source duration: {source_duration:.2f}s, Target duration: {target_duration:.2f}s")
    update_progress(10)

    # Calculate the required sample rate adjustment
    adjusted_sr = int(original_sr * (target_duration / source_duration))
    logger.debug(f"Adjusting sample rate from {original_sr} to {adjusted_sr} Hz")

    # Save with adjusted sample rate
    sf.write(target_path, y, adjusted_sr)

    update_progress(30)


def isolate_voice(
    input_path: Path,
    output_path: Path,
    progress_callback: Callable[[float], None],
) -> None:
    """
    Isolate voice from background noise using Eleven Labs API and match original audio properties.

    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        progress_callback: Optional callback function to report progress

    Raises:
        VoiceIsolationError: If voice isolation fails
    """
    # First get the isolated voice from API
    isolated_path = _call_elevenlabs_api(input_path, progress_callback)

    # Then match the audio properties
    try:
        _match_audio_properties(isolated_path, output_path, progress_callback)
    finally:
        os.unlink(isolated_path)

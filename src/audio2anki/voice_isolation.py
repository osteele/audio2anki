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


def _isolate_vocals(input_path: str, output_dir: str, progress_callback: Callable[[float], None] | None = None) -> None:
    """
    Isolate vocals from the input audio file using the ElevenLabs API.

    Args:
        input_path: Path to the input audio file
        output_dir: Directory where isolated vocals will be saved
        progress_callback: Callback function to report progress
    """
    if progress_callback is None:
        # Define a no-op function if no callback is provided
        def progress_callback_noop(_: float) -> None:
            return None

        progress_callback = progress_callback_noop

    # Call the API to isolate the vocals
    isolated_path = _call_elevenlabs_api(Path(input_path), progress_callback)

    # Define the output path for the isolated vocals
    output_vocals_path = Path(output_dir) / "vocals.wav"

    # Copy the isolated vocals to the output directory
    import shutil

    shutil.copy(isolated_path, output_vocals_path)


def _match_audio_properties(
    source_path: Path, target_path: Path, progress_callback: Callable[[float], None] | None = None
) -> None:
    """
    Match audio properties of the source file to the target file.

    Args:
        source_path: Path to the source audio file
        target_path: Path to match and save the result
        progress_callback: Callback function to report progress
    """
    # Ensure source file exists
    if not source_path.exists():
        raise FileNotFoundError(f"Source audio file not found: {source_path}")

    # Load the source file (vocals)
    y, sr = librosa.load(str(source_path), sr=None)

    # Save to target path using the source sample rate (cast to int for soundfile)
    sf.write(target_path, y, int(sr))

    if progress_callback:
        progress_callback(100)


def isolate_voice(
    input_path: Path, output_path: Path, progress_callback: Callable[[float], None] | None = None
) -> None:
    """
    Isolate voice from background noise using vocal remover.

    Args:
        input_path: Path to the input audio file
        output_path: Path to save the isolated voice
        progress_callback: Callback function to report progress
    """
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define temporary isolated voice path
        isolated_path = Path(temp_dir) / "vocals.wav"

        # Run vocal isolation
        _isolate_vocals(str(input_path), temp_dir, progress_callback)

        # Ensure the isolated file was created before proceeding
        if not isolated_path.exists():
            raise FileNotFoundError(f"Voice isolation failed to produce output file at {isolated_path}")

        # Match audio properties and copy to final output path
        _match_audio_properties(isolated_path, output_path, progress_callback)

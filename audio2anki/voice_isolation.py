"""Voice isolation using Eleven Labs API."""

import contextlib
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import httpx
import librosa
import soundfile as sf

from audio2anki.usage_tracker import record_api_usage

from .exceptions import Audio2AnkiError

logger = logging.getLogger(__name__)

VOICE_ISOLATION_FORMAT = "mp3"

API_BASE_URL = "https://api.elevenlabs.io/v1"


class VoiceIsolationError(Audio2AnkiError):
    """Error during voice isolation, with optional diagnostic context."""

    def __init__(
        self,
        message: str,
        *,
        input_file: str | None = None,
        transcoded_file: str | None = None,
        transcoded_file_size: int | None = None,
        api_url: str | None = None,
        http_status: int | None = None,
        api_response: str | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.cause = cause
        self.error_message = message
        self.input_file = input_file
        self.transcoded_file = transcoded_file
        self.transcoded_file_size = transcoded_file_size
        self.api_url = api_url
        self.http_status = http_status
        self.api_response = api_response

    def __str__(self) -> str:
        lines = [f"Voice Isolation Error: {self.error_message}"]
        if self.input_file:
            lines.append(f"  File processed: {self.input_file}")
        if self.transcoded_file:
            size_str = f" (size: {self.transcoded_file_size} bytes)" if self.transcoded_file_size is not None else ""
            lines.append(f"  Transcoded file: {self.transcoded_file}{size_str}")
        if self.api_url:
            lines.append(f"  API endpoint: {self.api_url}")
        if self.http_status is not None:
            lines.append(f"  HTTP status: {self.http_status}")
        if self.api_response:
            lines.append(f"  API response: {self.api_response}")
        if self.cause:
            lines.append(f"  Cause: {self.cause}")
        lines.append("\nSuggestions:")
        lines.append("- Check that the input file contains valid audio.")
        lines.append("- Ensure the file is not empty after transcoding.")
        lines.append("- If the problem persists, check API status or try again later.")
        return "\n".join(lines)


def get_voice_isolation_version() -> int:
    """Get the version of the voice isolation function."""
    return 1


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

    temp_path = None
    api_url = f"{API_BASE_URL}/audio-isolation/stream"
    transcoded_file_size = None
    try:
        transcoded_file_size = input_path.stat().st_size if input_path.exists() else None
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

            with open(input_path, "rb") as f:
                files = {"audio": (input_path.name, f, "audio/mpeg")}
                with httpx.Client(timeout=60.0) as client:
                    with client.stream(
                        "POST", api_url, headers={"xi-api-key": api_key, "accept": "application/json"}, files=files
                    ) as response:
                        if response.status_code != 200:
                            api_response = getattr(response, "text", None)
                            raise VoiceIsolationError(
                                "API request failed",
                                input_file=str(input_path),
                                transcoded_file=str(input_path),
                                transcoded_file_size=transcoded_file_size,
                                api_url=api_url,
                                http_status=response.status_code,
                                api_response=api_response,
                            )
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
                        character_cost = response.headers.get("Character-Cost")
                        record_api_usage(
                            model="ElevenLabs",
                            character_cost=int(character_cost or 0),
                        )
        if total_chunks == 0:
            api_response = None
            with contextlib.suppress(Exception):
                api_response = getattr(response, "text", None)
            raise VoiceIsolationError(
                "No audio data received from API",
                input_file=str(input_path),
                transcoded_file=str(input_path),
                transcoded_file_size=transcoded_file_size,
                api_url=api_url,
                http_status=response.status_code if "response" in locals() else None,
                api_response=api_response,
            )
        update_progress(70)
        return Path(temp_path)
    except httpx.TimeoutException as err:
        raise VoiceIsolationError(
            "API request timed out",
            input_file=str(input_path),
            transcoded_file=str(input_path),
            transcoded_file_size=transcoded_file_size,
            api_url=api_url,
            cause=err,
        ) from err
    except httpx.RequestError as err:
        raise VoiceIsolationError(
            f"API request failed: {err}",
            input_file=str(input_path),
            transcoded_file=str(input_path),
            transcoded_file_size=transcoded_file_size,
            api_url=api_url,
            cause=err,
        ) from err


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

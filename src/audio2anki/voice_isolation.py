"""Voice isolation using Eleven Labs API."""

import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import httpx
import librosa
import soundfile as sf

from . import cache

logger = logging.getLogger(__name__)

API_BASE_URL = "https://api.elevenlabs.io/v1"


class VoiceIsolationError(Exception):
    """Error during voice isolation."""


def _call_elevenlabs_api(input_path: Path, progress_callback: Callable[[float], None] | None = None) -> Path:
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
        if progress_callback:
            progress_callback(percent * 0.7)  # Scale to 70% of total progress

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise VoiceIsolationError(
            "ELEVENLABS_API_KEY environment variable not set. Get your API key from https://elevenlabs.io"
        )

    # Check API cache
    file_hash = cache.compute_file_hash(input_path)
    cache_params = {"input_hash": file_hash}

    if cache.cache_retrieve("voice_isolation_raw", input_path, ".mp3", extra_params=cache_params):
        cached_path = cache.get_cache_path("voice_isolation_raw", file_hash, ".mp3")
        logger.debug(f"Using cached raw isolated voice file: {cached_path}")
        update_progress(70)
        return Path(cached_path)

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

            # Store raw API response in cache
            with open(temp_path, "rb") as f:
                cached_path = cache.cache_store(
                    "voice_isolation_raw", input_path, f.read(), ".mp3", extra_params=cache_params
                )

            os.unlink(temp_path)
            update_progress(70)
            return Path(cached_path)

    except httpx.TimeoutException as err:
        raise VoiceIsolationError("API request timed out") from err
    except httpx.RequestError as err:
        raise VoiceIsolationError(f"API request failed: {err}") from err
    except Exception as err:
        raise VoiceIsolationError(f"Voice isolation failed: {err}") from err


def _match_audio_properties(
    source_path: Path, target_path: Path, progress_callback: Callable[[float], None] | None = None
) -> Path:
    """
    Match the duration of the target audio to the source audio by adjusting sample rate.

    Args:
        source_path: Path to source audio file (original)
        target_path: Path to target audio file (to be adjusted)
        progress_callback: Optional callback function to report progress

    Returns:
        Path to the adjusted audio file

    Raises:
        VoiceIsolationError: If audio adjustment fails
    """

    def update_progress(percent: float) -> None:
        if progress_callback:
            progress_callback(70 + percent * 0.3)  # Scale remaining 30% of progress

    # Check adjustment cache
    cache_params = {
        "source_hash": cache.compute_file_hash(source_path),
        "target_hash": cache.compute_file_hash(target_path),
    }

    if cache.cache_retrieve("voice_isolation_adjusted", target_path, ".mp3", extra_params=cache_params):
        cached_path = cache.get_cache_path("voice_isolation_adjusted", cache_params["target_hash"], ".mp3")
        logger.debug(f"Using cached adjusted audio file: {cached_path}")
        update_progress(30)
        return Path(cached_path)

    try:
        # Load durations
        source_duration = librosa.get_duration(path=str(source_path))
        # First load without resampling to get the raw samples
        y, original_sr = librosa.load(str(target_path), sr=None)
        target_duration = librosa.get_duration(y=y, sr=original_sr)

        logger.debug(f"Source duration: {source_duration:.2f}s, Target duration: {target_duration:.2f}s")
        update_progress(10)

        if abs(source_duration - target_duration) > 0.1:
            # Calculate the required sample rate adjustment
            adjusted_sr = int(original_sr * (target_duration / source_duration))
            logger.debug(f"Adjusting sample rate from {original_sr} to {adjusted_sr} Hz")

            # Save with adjusted sample rate
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                sf.write(temp_path, y, adjusted_sr)

                # Cache the adjusted result
                with open(temp_path, "rb") as f:
                    cached_path = cache.cache_store(
                        "voice_isolation_adjusted", target_path, f.read(), ".mp3", extra_params=cache_params
                    )

                os.unlink(temp_path)
        else:
            # If no adjustment needed, cache the original
            with open(target_path, "rb") as f:
                cached_path = cache.cache_store(
                    "voice_isolation_adjusted", target_path, f.read(), ".mp3", extra_params=cache_params
                )

        update_progress(30)
        return Path(cached_path)

    except Exception as e:
        raise VoiceIsolationError(f"Failed to adjust audio properties: {e}") from e


def isolate_voice(
    input_path: str | Path,
    progress_callback: Callable[[float], None] | None = None,
) -> Path:
    """
    Isolate voice from background noise using Eleven Labs API and match original audio properties.

    Args:
        input_path: Path to input audio file
        progress_callback: Optional callback function to report progress

    Returns:
        Path to the processed audio file

    Raises:
        VoiceIsolationError: If voice isolation fails
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # First get the isolated voice from API
    isolated_path = _call_elevenlabs_api(input_path, progress_callback)

    # Then match the audio properties
    return _match_audio_properties(input_path, isolated_path, progress_callback)

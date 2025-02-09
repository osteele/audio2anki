"""Voice isolation using Eleven Labs API."""

import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import httpx

from . import cache

logger = logging.getLogger(__name__)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
API_BASE_URL = "https://api.elevenlabs.io/v1"


class VoiceIsolationError(Exception):
    """Error during voice isolation."""


def isolate_voice(
    input_path: str | Path,
    progress_callback: Callable[[float], None] | None = None,
) -> Path:
    """
    Isolate voice from background noise using Eleven Labs API.

    Args:
        input_path: Path to input audio file
        progress_callback: Optional callback function to report progress

    Returns:
        Path to the isolated voice audio file

    Raises:
        VoiceIsolationError: If voice isolation fails
    """

    def update_progress(percent: float) -> None:
        """Update progress if callback is provided."""
        if progress_callback:
            progress_callback(percent)

    if not ELEVENLABS_API_KEY:
        raise VoiceIsolationError(
            "ELEVENLABS_API_KEY environment variable not set. Get your API key from https://elevenlabs.io"
        )

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Check cache first
    update_progress(0)
    cached_result = cache.cache_retrieve("voice_isolation", input_path, ".mp3", 7)
    if cached_result:
        cached_path = cache.get_cache_path("voice_isolation", cache.compute_file_hash(input_path), ".mp3")
        logger.debug(f"Using cached isolated voice file: {cached_path}")
        update_progress(100)
        return Path(cached_path)

    try:
        # Prepare API request
        url = f"{API_BASE_URL}/audio-isolation/stream"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "accept": "application/json",
        }

        logger.debug("Uploading audio file to Eleven Labs API")
        update_progress(10)

        # Create a temporary file to store the streamed response
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

            # Upload file and stream response
            with open(input_path, "rb") as f:
                files = {"audio": (input_path.name, f, "audio/mpeg")}
                with httpx.Client(timeout=60.0) as client:
                    with client.stream("POST", url, headers=headers, files=files) as response:
                        if response.status_code != 200:
                            try:
                                error_data = response.json()
                                error_msg = error_data.get("detail", f"API request failed: {response.status_code}")
                            except Exception as err:
                                error_msg = f"API request failed: {response.status_code}"
                                raise VoiceIsolationError(error_msg) from err
                            raise VoiceIsolationError(error_msg) from None

                        logger.debug("Streaming isolated audio from API")
                        update_progress(50)

                        # Stream the response to a temporary file
                        total_chunks = 0
                        for chunk in response.iter_bytes():
                            if not chunk:
                                continue
                            temp_file.write(chunk)
                            total_chunks += 1
                            # Update progress based on chunks received
                            if total_chunks % 10 == 0:  # Update every 10 chunks
                                update_progress(50 + (total_chunks % 30))  # Progress between 50-80%

            if total_chunks == 0:
                raise VoiceIsolationError("No audio data received from API") from None

            update_progress(80)

            # Store in cache
            with open(temp_path, "rb") as f:
                cached_path = cache.cache_store("voice_isolation", input_path, f.read(), ".mp3")

            # Clean up temporary file
            os.unlink(temp_path)

            logger.debug(f"Voice isolation complete: {cached_path}")
            update_progress(100)

            return Path(cached_path)

    except httpx.TimeoutException as err:
        raise VoiceIsolationError("API request timed out") from err
    except httpx.RequestError as err:
        raise VoiceIsolationError(f"API request failed: {err}") from err
    except Exception as err:
        raise VoiceIsolationError(f"Voice isolation failed: {err}") from err

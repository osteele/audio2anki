"""Voice isolation using the Eleven Labs API."""

import logging
import os
from pathlib import Path

import httpx

from .pipeline import PipelineContext, produces_artifacts

logger = logging.getLogger(__name__)


@produces_artifacts(isolated_voice={"extension": "mp3"})
def isolate_voice(context: PipelineContext, input_path: Path) -> None:
    """Isolate voice from background noise using Eleven Labs API."""
    # Update input tracking for cache
    context.update_stage_input("isolated_voice", input_path)

    # Get output path
    output_path = context.get_artifact_path("isolated_voice")

    # Check cache first
    if context.retrieve_from_cache("isolated_voice", input_path):
        logger.info("Using cached isolated voice file")
        return

    # Validate input file exists and is readable
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Get API key from environment
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY environment variable not set")

    # Prepare API request
    url = "https://api.elevenlabs.io/v1/speech-to-speech/voice-isolation"
    headers = {
        "Accept": "application/json",
        "xi-api-key": api_key,
    }

    # Read audio file
    with open(input_path, "rb") as f:
        files = {"audio": (input_path.name, f, "audio/mpeg")}

        # Make API request with streaming
        with httpx.Client(timeout=None) as client:
            try:
                with client.stream("POST", url, headers=headers, files=files) as response:
                    response.raise_for_status()

                    # Initialize variables for streaming
                    total_chunks = 0
                    audio_data = bytearray()

                    # Process streaming response
                    for chunk in response.iter_bytes():
                        if chunk:  # Ensure chunk is not empty
                            total_chunks += 1
                            audio_data.extend(chunk)

                            # Update progress (rough estimate)
                            if total_chunks % 10 == 0:  # Update every 10 chunks
                                # Assume we're halfway through after 100 chunks
                                progress = min((total_chunks / 200.0) * 100, 90)
                                context.progress.update_progress(progress)

                    # Validate we got audio data
                    if not audio_data:
                        raise ValueError("No audio data received from API")

                    # Write output file
                    with open(output_path, "wb") as out_file:
                        out_file.write(audio_data)

                    # Store in cache
                    context.store_in_cache("isolated_voice", output_path, input_path)

                    # Mark as complete
                    context.progress.update_progress(100)

            except httpx.HTTPError as e:
                logger.error("HTTP error during voice isolation: %s", str(e))
                if hasattr(e, "response") and e.response is not None:
                    logger.error("Response content: %s", e.response.text)
                raise

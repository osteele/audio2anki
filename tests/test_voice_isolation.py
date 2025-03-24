"""Tests for voice isolation functionality."""

import io
import os
import tempfile
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
import soundfile as sf
from rich.progress import Progress, TaskID

from audio2anki.pipeline import PipelineContext, PipelineProgress, pipeline_function
from audio2anki.types import LanguageCode


class MockResponse:
    """Mock HTTP response for testing."""

    def __init__(self, status_code: int = 200, audio_data: bytes | None = None):
        self.status_code = status_code
        # Create a short audio file for testing if no custom audio data is provided
        self._audio_data = audio_data if audio_data is not None else self._create_test_audio()
        # Add these to make it compatible with httpx.Response
        self.request = MagicMock()
        self.headers = {}
        self.content = self._audio_data

    def _create_test_audio(self) -> bytes:
        """Create a short test audio file in memory."""
        # Create 1 second of audio at 44100Hz
        samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        buffer = io.BytesIO()
        sf.write(buffer, samples, 44100, format="WAV")
        buffer.seek(0)  # Reset buffer position
        return buffer.getvalue()

    def iter_bytes(self):
        """Simulate streaming response."""
        chunk_size = 1024
        data = self._audio_data
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]

    def raise_for_status(self) -> None:
        """Raise an error if status code is not 200."""
        if self.status_code != 200:
            # Create a proper HTTPStatusError with compatible arguments
            error = httpx.HTTPStatusError(
                "HTTP error occurred",
                request=self.request,
                response=MagicMock(spec=httpx.Response),
            )
            # Make the mock response compatible with what's expected
            error.response = MagicMock(spec=httpx.Response)
            error.response.status_code = self.status_code
            error.response.json.return_value = self.json()
            raise error

    def json(self) -> dict[str, Any]:
        """Return mock JSON response for error cases."""
        return {"detail": "API error message"}


@pytest.fixture(params=[200, 400])
def mock_api_response(request: pytest.FixtureRequest) -> MockResponse:
    """Create a mock API response."""
    if request.param == 200:
        return MockResponse(status_code=200)
    return MockResponse(status_code=400)


@pytest.fixture
def mock_http_client(mock_api_response: MockResponse) -> Generator[MagicMock, None, None]:
    """Fixture to mock httpx.Client and its responses."""
    with patch("httpx.Client", autospec=True) as mock_client:
        # Configure the mock client
        mock_instance = mock_client.return_value.__enter__.return_value
        mock_instance.stream.return_value.__enter__.return_value = mock_api_response

        # Mock environment variable
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}):
            yield mock_client


@pytest.fixture
def mock_pipeline_progress() -> PipelineProgress:
    """Create a mock pipeline progress tracker."""
    mock_progress = MagicMock(spec=Progress)
    mock_progress.update = MagicMock()
    mock_console = MagicMock()
    progress = PipelineProgress(
        progress=mock_progress,
        pipeline_task=MagicMock(spec=TaskID),
        console=mock_console,
    )
    progress.start_stage("voice_isolation")
    return progress


@pytest.fixture
def mock_pipeline_context(mock_pipeline_progress: PipelineProgress, tmp_path: Path) -> PipelineContext:
    """Create a mock pipeline context."""
    context = PipelineContext(
        progress=mock_pipeline_progress,
        source_language=LanguageCode("zh"),
    )
    context.set_input_file(tmp_path / "input.mp3")
    return context


# Add a decorator to make the test function compatible with PipelineFunction
@pipeline_function(extension="mp3")
def mock_isolate_voice(input_path: Path, output_path: Path, progress_callback: Callable[[float], None]) -> None:
    """Test implementation that creates the output file."""
    # Simply create the output file
    output_path.touch()


# Add this fixture to create a real test audio file
@pytest.fixture
def real_test_audio_file(tmp_path: Path) -> Path:
    """Create a real test audio file with sine wave data."""
    # Create 1 second of audio at 44100Hz (sine wave at 440Hz)
    samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))

    # Save as MP3 file in the temporary directory
    output_path = tmp_path / "real_test.mp3"
    sf.write(str(output_path), samples, 44100, format="WAV")

    return output_path


def test_isolate_voice_basic(
    test_audio_file: Path,
    mock_http_client: MagicMock,
    mock_pipeline_context: PipelineContext,
    real_test_audio_file: Path,
) -> None:
    """Test basic voice isolation functionality."""
    from audio2anki.voice_isolation import VoiceIsolationError, isolate_voice

    # pyright: ignore[reportPrivateUsage]
    mock_pipeline_context._current_fn = mock_isolate_voice
    mock_pipeline_context.update_stage_input("mock_isolate_voice", real_test_audio_file)

    # Set up the artifacts dictionary
    # pyright: ignore[reportPrivateUsage]
    mock_pipeline_context._artifacts = {"mock_isolate_voice": {"extension": "mp3"}}

    # Create output path in the temporary directory
    output_path = real_test_audio_file.parent / "output.mp3"

    # Define a typed function for the side effect
    def match_audio_side_effect(
        source_path: Path, target_path: Path, progress_callback: Callable[[float], None]
    ) -> None:
        target_path.touch()

    # Mock the cache-related methods
    with (
        patch.object(mock_pipeline_context, "retrieve_from_cache", return_value=None),
        patch.object(mock_pipeline_context, "get_artifact_path", return_value=output_path),
        patch.object(mock_pipeline_context, "store_in_cache"),
        # Mock the audio processing function to create the output file
        patch("audio2anki.voice_isolation._match_audio_properties", side_effect=match_audio_side_effect),
    ):
        # Get the status code from the mock response
        status_code = (
            mock_http_client.return_value.__enter__.return_value.stream.return_value.__enter__.return_value.status_code
        )

        # For status code 400, we expect an error
        if status_code == 400:
            with pytest.raises(VoiceIsolationError, match="API error message|Voice isolation failed"):
                with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}):
                    isolate_voice(
                        real_test_audio_file,
                        output_path,
                        progress_callback=lambda x: None,
                    )
        else:
            # For status code 200, we expect success
            # Create a temporary file to simulate API response with real audio data
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                # Write real audio data to the temp file
                temp_file.write(b"test audio data")
                temp_file.flush()

            # Mock the API call to return our temp file
            with patch("audio2anki.voice_isolation._call_elevenlabs_api", return_value=temp_path):
                with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}):
                    isolate_voice(
                        real_test_audio_file,
                        output_path,
                        progress_callback=lambda x: None,
                    )

                    # Verify the output file exists
                    assert output_path.exists()


def test_isolate_voice_api_timeout(
    test_audio_file: Path, mock_pipeline_context: PipelineContext, real_test_audio_file: Path
) -> None:
    """Test error handling for API timeout."""
    from audio2anki.voice_isolation import VoiceIsolationError, isolate_voice

    mock_pipeline_context._current_fn = mock_isolate_voice
    mock_pipeline_context.update_stage_input("isolated_voice", real_test_audio_file)

    # Set up the artifacts dictionary
    mock_pipeline_context._artifacts = {"isolated_voice": {"extension": "mp3"}}

    # Create output path in the temporary directory
    output_path = real_test_audio_file.parent / "output_timeout.mp3"

    # Mock the cache-related methods
    with (
        patch.object(mock_pipeline_context, "retrieve_from_cache", return_value=None),
        patch.object(mock_pipeline_context, "get_artifact_path", return_value=output_path),
        # Mock the audio processing function to avoid file handling issues
        patch("audio2anki.voice_isolation._match_audio_properties", return_value=None),
    ):
        with patch("httpx.Client", autospec=True) as mock_client:
            mock_instance = mock_client.return_value.__enter__.return_value
            mock_instance.stream.side_effect = httpx.TimeoutException("Timeout")

            with pytest.raises(VoiceIsolationError, match="API request timed out"):
                with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}):
                    isolate_voice(
                        real_test_audio_file,
                        output_path,
                        progress_callback=lambda x: None,
                    )


def test_isolate_voice_request_error(
    test_audio_file: Path, mock_pipeline_context: PipelineContext, real_test_audio_file: Path
) -> None:
    """Test error handling for general request errors."""
    from audio2anki.voice_isolation import VoiceIsolationError, isolate_voice

    mock_pipeline_context._current_fn = mock_isolate_voice
    mock_pipeline_context.update_stage_input("isolated_voice", real_test_audio_file)

    # Set up the artifacts dictionary
    mock_pipeline_context._artifacts = {"isolated_voice": {"extension": "mp3"}}

    # Create output path in the temporary directory
    output_path = real_test_audio_file.parent / "output_request_error.mp3"

    # Mock the cache-related methods
    with (
        patch.object(mock_pipeline_context, "retrieve_from_cache", return_value=None),
        patch.object(mock_pipeline_context, "get_artifact_path", return_value=output_path),
        # Mock the audio processing function to avoid file handling issues
        patch("audio2anki.voice_isolation._match_audio_properties", return_value=None),
    ):
        with patch("httpx.Client", autospec=True) as mock_client:
            mock_instance = mock_client.return_value.__enter__.return_value
            mock_instance.stream.side_effect = httpx.RequestError("Network error")

            with pytest.raises(VoiceIsolationError, match="API request failed: Network error"):
                with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}):
                    isolate_voice(
                        real_test_audio_file,
                        output_path,
                        progress_callback=lambda x: None,
                    )


def test_isolate_voice_empty_response(
    test_audio_file: Path, mock_pipeline_context: PipelineContext, real_test_audio_file: Path
) -> None:
    """Test error handling for empty API response."""
    from audio2anki.voice_isolation import VoiceIsolationError, isolate_voice

    mock_pipeline_context._current_fn = mock_isolate_voice
    mock_pipeline_context.update_stage_input("isolated_voice", real_test_audio_file)

    # Set up the artifacts dictionary
    mock_pipeline_context._artifacts = {"isolated_voice": {"extension": "mp3"}}

    # Create output path in the temporary directory
    output_path = real_test_audio_file.parent / "output_empty.mp3"

    # Mock the cache-related methods
    with (
        patch.object(mock_pipeline_context, "retrieve_from_cache", return_value=None),
        patch.object(mock_pipeline_context, "get_artifact_path", return_value=output_path),
        # Mock the audio processing function to avoid file handling issues
        patch("audio2anki.voice_isolation._match_audio_properties", return_value=None),
    ):
        # Create a mock response that returns no data
        empty_response = MockResponse(status_code=200, audio_data=b"")

        with patch("httpx.Client", autospec=True) as mock_client:
            mock_instance = mock_client.return_value.__enter__.return_value
            mock_instance.stream.return_value.__enter__.return_value = empty_response

            with pytest.raises(VoiceIsolationError, match="No audio data received from API"):
                with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}):
                    isolate_voice(
                        real_test_audio_file,
                        output_path,
                        progress_callback=lambda x: None,
                    )


def test_isolate_voice_no_api_key(
    test_audio_file: Path, tmp_path: Path, mock_pipeline_context: PipelineContext, real_test_audio_file: Path
) -> None:
    """Test error when API key is not set."""
    from audio2anki.voice_isolation import VoiceIsolationError, isolate_voice

    mock_pipeline_context._current_fn = mock_isolate_voice
    mock_pipeline_context.update_stage_input("isolated_voice", real_test_audio_file)

    # Set up the artifacts dictionary
    mock_pipeline_context._artifacts = {"isolated_voice": {"extension": "mp3"}}

    # Create output path in the temporary directory
    output_path = real_test_audio_file.parent / "output_no_key.mp3"

    # Mock the cache-related methods
    with (
        patch.object(mock_pipeline_context, "retrieve_from_cache", return_value=None),
        patch.object(mock_pipeline_context, "get_artifact_path", return_value=output_path),
        # Mock the audio processing function to avoid file handling issues
        patch("audio2anki.voice_isolation._match_audio_properties", return_value=None),
    ):
        # Temporarily remove the API key from environment
        with patch.dict(os.environ, {}, clear=True):  # Clear all env vars for this test
            # Now we expect a ValueError with the specific message about API key
            with pytest.raises(VoiceIsolationError, match="ELEVENLABS_API_KEY.*not set"):
                isolate_voice(
                    real_test_audio_file,
                    output_path,
                    progress_callback=lambda x: None,
                )

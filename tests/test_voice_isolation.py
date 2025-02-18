"""Tests for voice isolation functionality."""

import io
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
import soundfile as sf

from audio2anki.voice_isolation import VoiceIsolationError, isolate_voice


class MockResponse:
    """Mock HTTP response for testing."""

    def __init__(self, status_code: int = 200, audio_data: bytes | None = None):
        self.status_code = status_code
        # Create a short audio file for testing if no custom audio data is provided
        self._audio_data = audio_data if audio_data is not None else self._create_test_audio()

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


def test_isolate_voice_basic(test_audio_file: Path, mock_http_client: MagicMock, tmp_path: Path) -> None:
    """Test basic voice isolation functionality."""
    output_path = tmp_path / "output.mp3"

    # Mock the _match_audio_properties function to copy the temp file to output
    def mock_match_audio(source: Path, target: Path, callback=None) -> Path:
        with open(target, "rb") as src, open(output_path, "wb") as dst:
            dst.write(src.read())
        return output_path

    with patch("audio2anki.voice_isolation._match_audio_properties", side_effect=mock_match_audio):
        if (
            mock_http_client.return_value.__enter__.return_value.stream.return_value.__enter__.return_value.status_code
            == 400
        ):
            with pytest.raises(VoiceIsolationError, match="API error message"):
                isolate_voice(test_audio_file, output_path)
            assert not output_path.exists()
        else:
            isolate_voice(test_audio_file, output_path)
            assert output_path.exists()
            assert output_path.stat().st_size > 0


def test_isolate_voice_with_progress(test_audio_file: Path, mock_http_client: MagicMock, tmp_path: Path) -> None:
    """Test progress callback functionality."""
    output_path = tmp_path / "output.mp3"
    progress_values: list[float] = []

    def progress_callback(percent: float) -> None:
        progress_values.append(percent)

    # Mock the _match_audio_properties function
    def mock_match_audio(source: Path, target: Path, callback=None) -> Path:
        with open(target, "rb") as src, open(output_path, "wb") as dst:
            dst.write(src.read())
        if callback:  # Call the progress callback with final progress
            callback(100)  # This will be scaled by the 0.3 factor in the real function
        return output_path

    with patch("audio2anki.voice_isolation._match_audio_properties", side_effect=mock_match_audio):
        if (
            mock_http_client.return_value.__enter__.return_value.stream.return_value.__enter__.return_value.status_code
            == 400
        ):
            with pytest.raises(VoiceIsolationError):
                isolate_voice(test_audio_file, output_path, progress_callback=progress_callback)
            assert len(progress_values) > 0  # Should have at least initial progress
            assert progress_values[0] <= 10  # Initial progress should be low
        else:
            isolate_voice(test_audio_file, output_path, progress_callback=progress_callback)
            assert progress_values[0] <= 10  # Initial progress should be low
            assert any(45 <= v <= 80 for v in progress_values)  # Processing progress
            assert progress_values[-1] >= 70  # Final progress should be substantial


def test_isolate_voice_api_error(test_audio_file: Path, mock_http_client: MagicMock, tmp_path: Path) -> None:
    """Test error handling for API errors."""
    output_path = tmp_path / "output.mp3"

    # Mock the _match_audio_properties function
    def mock_match_audio(source: Path, target: Path, callback=None) -> Path:
        with open(target, "rb") as src, open(output_path, "wb") as dst:
            dst.write(src.read())
        return output_path

    with patch("audio2anki.voice_isolation._match_audio_properties", side_effect=mock_match_audio):
        if (
            mock_http_client.return_value.__enter__.return_value.stream.return_value.__enter__.return_value.status_code
            == 400
        ):
            with pytest.raises(VoiceIsolationError, match="API error message"):
                isolate_voice(test_audio_file, output_path)
        else:
            isolate_voice(test_audio_file, output_path)
            assert output_path.exists()


def test_isolate_voice_api_timeout(test_audio_file: Path, tmp_path: Path) -> None:
    """Test error handling for API timeout."""
    output_path = tmp_path / "output.mp3"

    with (
        patch("audio2anki.voice_isolation._call_elevenlabs_api", autospec=True) as mock_call_api,
        patch("audio2anki.voice_isolation._match_audio_properties", autospec=True) as mock_match_audio,
    ):
        mock_call_api.side_effect = httpx.TimeoutException("Timeout")
        with pytest.raises(VoiceIsolationError, match="Voice isolation failed: Timeout"):
            isolate_voice(test_audio_file, output_path)
        mock_match_audio.assert_not_called()


def test_isolate_voice_request_error(test_audio_file: Path, tmp_path: Path) -> None:
    """Test error handling for general request errors."""
    output_path = tmp_path / "output.mp3"

    with (
        patch("audio2anki.voice_isolation._call_elevenlabs_api", autospec=True) as mock_call_api,
        patch("audio2anki.voice_isolation._match_audio_properties", autospec=True) as mock_match_audio,
    ):
        mock_call_api.side_effect = httpx.RequestError("Network error")
        with pytest.raises(VoiceIsolationError, match="Voice isolation failed: Network error"):
            isolate_voice(test_audio_file, output_path)
        mock_match_audio.assert_not_called()


def test_isolate_voice_empty_response(test_audio_file: Path, mock_http_client: MagicMock, tmp_path: Path) -> None:
    """Test error handling for empty API response."""
    output_path = tmp_path / "output.mp3"
    temp_isolated_path = tmp_path / "temp_isolated.mp3"
    # Don't create the temp file, simulating an empty response

    with (
        patch(
            "audio2anki.voice_isolation._call_elevenlabs_api", return_value=temp_isolated_path, autospec=True
        ) as mock_call_api,
        patch(
            "audio2anki.voice_isolation._match_audio_properties", return_value=output_path, autospec=True
        ) as mock_match_audio,
    ):
        mock_call_api.side_effect = VoiceIsolationError("No audio data received")

        with pytest.raises(VoiceIsolationError, match="No audio data received"):
            isolate_voice(test_audio_file, output_path)
        mock_match_audio.assert_not_called()

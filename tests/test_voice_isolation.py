"""Tests for voice isolation functionality."""

import os
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from audio2anki.cache import clear_cache
from audio2anki.voice_isolation import VoiceIsolationError, isolate_voice


@pytest.fixture
def test_audio_file(tmp_path: Path) -> Path:
    """Create a test audio file."""
    audio_file = tmp_path / "test_audio.mp3"
    audio_file.write_bytes(b"test audio data")
    return audio_file


class MockResponse:
    """Mock response for API calls."""

    def __init__(self) -> None:
        self.status_code: int = 200
        self.chunks: list[bytes] = [b"isolated", b"voice", b"data"]
        self.chunk_index: int = 0

    def iter_bytes(self) -> Generator[bytes, None, None]:
        yield from self.chunks

    def json(self) -> dict[str, Any]:
        raise RuntimeError("Should not call json() on successful response")


@pytest.fixture
def mock_api_response() -> MockResponse:
    """Mock successful API response with audio data."""
    return MockResponse()


@pytest.fixture(autouse=True)
def setup_and_cleanup() -> Generator[None, None, None]:
    """Set up test environment and clean up after tests."""
    os.environ["ELEVENLABS_API_KEY"] = "test_key"
    yield
    if "ELEVENLABS_API_KEY" in os.environ:
        del os.environ["ELEVENLABS_API_KEY"]


def test_isolate_voice_missing_api_key(test_audio_file: Path):
    """Test error handling when API key is missing."""
    # Delete API key before any operations and clear cache
    del os.environ["ELEVENLABS_API_KEY"]
    clear_cache()

    with pytest.raises(VoiceIsolationError, match="ELEVENLABS_API_KEY.*not set"):
        isolate_voice(test_audio_file)


def test_isolate_voice_basic(test_audio_file: Path, mock_api_response: MockResponse, test_cache_dir: Path) -> None:
    """Test basic voice isolation functionality."""
    with patch("httpx.Client") as mock_client:
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_api_response
        mock_client.return_value.__enter__.return_value.stream.return_value = mock_stream

        output_path = isolate_voice(test_audio_file)
        assert output_path.exists()
        assert output_path.suffix == ".mp3"
        with open(output_path, "rb") as f:
            data = f.read()
            assert b"isolatedvoicedata" == data


def test_isolate_voice_caching(test_audio_file: Path, mock_api_response: MockResponse, test_cache_dir: Path) -> None:
    """Test that voice isolation results are cached."""
    with patch("httpx.Client") as mock_client:
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_api_response
        mock_client.return_value.__enter__.return_value.stream.return_value = mock_stream

        # First call should use API
        path1 = isolate_voice(test_audio_file)
        first_call_count = mock_client.return_value.__enter__().stream.call_count
        assert first_call_count == 1

        # Second call should use cache (i.e. no additional API call)
        path2 = isolate_voice(test_audio_file)
        second_call_count = mock_client.return_value.__enter__().stream.call_count
        # Assert that the call count did not increase after the second call
        assert second_call_count == first_call_count
        # Optionally, assert that the cached path is the same
        assert path1 == path2


def test_isolate_voice_progress_callback(
    test_audio_file: Path, mock_api_response: MockResponse, test_cache_dir: Path
) -> None:
    """Test that progress callback is called with expected values."""
    progress_values: list[float] = []

    def progress_callback(value: float) -> None:
        progress_values.append(value)

    with patch("httpx.Client") as mock_client:
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_api_response
        mock_client.return_value.__enter__.return_value.stream.return_value = mock_stream

        isolate_voice(test_audio_file, progress_callback=progress_callback)

        # Check progress values
        assert 0 in progress_values  # Initial progress
        assert any(50 <= v <= 80 for v in progress_values)  # Processing progress
        assert 100 in progress_values  # Final progress


def test_isolate_voice_file_not_found() -> None:
    """Test error handling for non-existent input file."""
    with pytest.raises(FileNotFoundError):
        isolate_voice("nonexistent.mp3")


def test_isolate_voice_api_error(test_audio_file: Path, test_cache_dir: Path) -> None:
    """Test error handling for API errors."""

    class MockErrorResponse:
        status_code: int = 400

        def json(self) -> dict[str, Any]:
            return {"detail": "API error message"}

        def iter_bytes(self) -> Generator[bytes, None, None]:
            yield from ()

    with patch("httpx.Client") as mock_client:
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = MockErrorResponse()
        mock_client.return_value.__enter__.return_value.stream.return_value = mock_stream
        with pytest.raises(VoiceIsolationError, match="API error message"):
            isolate_voice(test_audio_file)


def test_isolate_voice_api_timeout(test_audio_file: Path, test_cache_dir: Path) -> None:
    """Test error handling for API timeout."""
    with patch("httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.stream.side_effect = httpx.TimeoutException("Timeout")
        with pytest.raises(VoiceIsolationError, match="API request timed out"):
            isolate_voice(test_audio_file)


def test_isolate_voice_request_error(test_audio_file: Path, test_cache_dir: Path) -> None:
    """Test error handling for general request errors."""
    with patch("httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.stream.side_effect = httpx.RequestError("Network error")
        with pytest.raises(VoiceIsolationError, match="API request failed"):
            isolate_voice(test_audio_file)


def test_isolate_voice_empty_response(test_audio_file: Path, test_cache_dir: Path) -> None:
    """Test error handling for empty API response."""

    class MockEmptyResponse:
        status_code: int = 200

        def iter_bytes(self) -> Generator[bytes, None, None]:
            yield from ()

        def json(self) -> dict[str, Any]:
            raise RuntimeError("Should not call json() on successful response")

    with patch("httpx.Client") as mock_client:
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = MockEmptyResponse()
        mock_client.return_value.__enter__.return_value.stream.return_value = mock_stream
        with pytest.raises(VoiceIsolationError, match="No audio data received"):
            isolate_voice(test_audio_file)

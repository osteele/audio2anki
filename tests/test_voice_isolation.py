"""Tests for voice isolation functionality."""

import os
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
import soundfile as sf

from audio2anki.cache import clear_cache
from audio2anki.voice_isolation import VoiceIsolationError, isolate_voice


class MockResponse:
    """Mock HTTP response for testing."""

    def __init__(self, status_code: int = 200):
        self.status_code = status_code
        # Create a short audio file for testing
        self._audio_data = self._create_test_audio()

    def _create_test_audio(self) -> bytes:
        """Create a short test audio file."""
        # Create 1 second of audio at 44100Hz
        samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        with sf.SoundFile(
            "temp.wav",
            mode="w",
            samplerate=44100,
            channels=1,
            format="WAV",
        ) as f:
            f.write(samples)

        with open("temp.wav", "rb") as f:
            data = f.read()
        os.unlink("temp.wav")
        return data

    def iter_bytes(self):
        """Simulate streaming response."""
        chunk_size = 1024
        data = self._audio_data
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]


@pytest.fixture
def test_audio_file(tmp_path: Path) -> Path:
    """Create a test audio file."""
    audio_path = tmp_path / "test_audio.mp3"
    # Create 2 seconds of audio at 44100Hz
    samples = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 88200))
    with sf.SoundFile(
        audio_path,
        mode="w",
        samplerate=44100,
        channels=1,
        format="WAV",
    ) as f:
        f.write(samples)
    return audio_path


@pytest.fixture
def mock_api_response() -> MockResponse:
    """Create a mock API response."""
    return MockResponse()


@pytest.fixture
def test_cache_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a test cache directory."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    with patch("audio2anki.cache.CACHE_DIR", cache_dir):
        yield cache_dir


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
        assert output_path.stat().st_size > 0


def test_isolate_voice_caching(test_audio_file: Path, mock_api_response: MockResponse, test_cache_dir: Path) -> None:
    """Test that voice isolation results are cached."""
    # Create a dummy file to simulate the raw isolated output.
    dummy_path = test_audio_file.parent / "dummy.wav"
    t = np.linspace(0, 1, 44100, endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(dummy_path, samples, 44100, format="WAV")

    call_count = 0

    def fake_call_elevenlabs_api(input_path: str, progress_callback: Callable[[float], None]):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return dummy_path
        else:
            return dummy_path

    with patch("audio2anki.voice_isolation._call_elevenlabs_api", side_effect=fake_call_elevenlabs_api):
        # Clear cache before test
        clear_cache()

        # First call should trigger the API call
        path1 = isolate_voice(test_audio_file)
        assert call_count == 1

        # Second call should retrieve from cache (i.e. no additional API call)
        path2 = isolate_voice(test_audio_file)
        assert call_count == 1
        with open(path1, "rb") as f1, open(path2, "rb") as f2:
            assert f1.read() == f2.read()


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

        # Clear cache before test
        clear_cache()

        isolate_voice(test_audio_file, progress_callback=progress_callback)

        # Check progress values
        assert progress_values[0] <= 10  # Initial progress should be low
        assert any(45 <= v <= 80 for v in progress_values)  # Processing progress
        assert progress_values[-1] >= 70  # Final progress should be substantial

        # (Optionally) verify that progress reaches from a low initial value to a substantial final value.


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

"""Tests for transcription module."""

import os
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from openai import AuthenticationError, OpenAI
from openai.types.audio import Transcription

from audio2anki import cache
from audio2anki.transcribe import TranscriptionSegment, load_transcript, save_transcript, transcribe_audio


@pytest.fixture
def mock_openai() -> Mock:
    """Create a mock OpenAI client."""
    mock = Mock(spec=OpenAI)
    mock.audio = Mock()
    mock.audio.transcriptions = Mock()
    mock.audio.transcriptions.create = Mock()
    return mock


@pytest.fixture
def mock_whisper_response() -> Mock:
    """Return mock Whisper response."""
    mock_response = Mock(spec=Transcription)
    mock_response.text = "Hello world"
    mock_response.segments = [
        Mock(start=0.0, end=2.0, text="Hello"),
        Mock(start=2.0, end=4.0, text="world"),
    ]
    return mock_response


@pytest.fixture
def setup_cache() -> Generator[None, None, None]:
    """Set up and tear down the cache for each test."""
    cache.init_cache()
    yield
    cache.clear_cache()


def test_transcribe_audio(
    tmp_path: Path,
    mock_openai: Mock,
    mock_whisper_response: Mock,
    setup_cache: Generator[None, None, None],
) -> None:
    """Test audio transcription with OpenAI API."""
    # Create dummy audio file
    audio_file = tmp_path / "test.mp3"
    audio_file.touch()

    # Set up mock
    mock_openai.audio.transcriptions.create.return_value = mock_whisper_response

    with (
        patch("audio2anki.transcribe.OpenAI", return_value=mock_openai),
        patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
    ):
        segments = transcribe_audio(
            audio_file,
            transcript_path=None,
            model="whisper-1",
            language="english",
            task_id=None,
            progress=None,
        )

        assert len(segments) == 2
        assert segments[0].text == "Hello"
        assert segments[1].text == "world"
        assert segments[0].start == 0.0
        assert segments[0].end == 2.0
        assert segments[1].start == 2.0
        assert segments[1].end == 4.0

        # Test caching - second call should use cache
        segments2 = transcribe_audio(
            audio_file,
            transcript_path=None,
            model="whisper-1",
            language="english",
            task_id=None,
            progress=None,
        )

        assert segments == segments2
        # Should only call the API once
        assert mock_openai.audio.transcriptions.create.call_count == 1


def test_transcribe_error(tmp_path: Path, mock_openai: Mock, setup_cache: Generator[None, None, None]) -> None:
    """Test transcription error handling."""
    # Create dummy audio file
    audio_file = tmp_path / "test.mp3"
    audio_file.touch()

    # Set up mock error
    mock_openai.audio.transcriptions.create.side_effect = AuthenticationError(
        message="Incorrect API key provided: test-key",
        body={"error": {"message": "Incorrect API key provided: test-key"}},
        response=Mock(status_code=401, reason_phrase="Unauthorized"),
    )

    with (
        patch("audio2anki.transcribe.OpenAI", return_value=mock_openai),
        patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
    ):
        with pytest.raises(RuntimeError, match=r"Transcription failed: Incorrect API key provided: test-key"):
            transcribe_audio(
                audio_file,
                transcript_path=None,
                model="whisper-1",
                language="english",
                task_id=None,
                progress=None,
            )


def test_load_and_save_transcript(tmp_path: Path) -> None:
    """Test loading and saving transcript."""
    transcript_file = tmp_path / "transcript.tsv"
    segments = [
        TranscriptionSegment(start=0.0, end=2.0, text="Hello"),
        TranscriptionSegment(start=2.0, end=4.0, text="world"),
    ]

    # Save transcript
    save_transcript(segments, transcript_file)
    assert transcript_file.exists()

    # Load transcript
    loaded_segments = load_transcript(transcript_file)
    assert len(loaded_segments) == 2
    assert loaded_segments[0].text == "Hello"
    assert loaded_segments[1].text == "world"
    assert loaded_segments[0].start == 0.0
    assert loaded_segments[0].end == 2.0
    assert loaded_segments[1].start == 2.0
    assert loaded_segments[1].end == 4.0


def test_transcribe_with_length_filters(
    tmp_path: Path,
    mock_openai: Mock,
    mock_whisper_response: Mock,
    setup_cache: Generator[None, None, None],
) -> None:
    """Test transcription with length filters."""
    # Create dummy audio file
    audio_file = tmp_path / "test.mp3"
    audio_file.touch()

    # Set up mock response with segments of different lengths
    mock_whisper_response.segments = [
        Mock(start=0.0, end=1.0, text="Short"),  # 1 second
        Mock(start=1.0, end=16.0, text="Too long"),  # 15 seconds
        Mock(start=16.0, end=18.0, text="Good length"),  # 2 seconds
        Mock(start=18.0, end=18.5, text="Too short"),  # 0.5 seconds
    ]
    mock_openai.audio.transcriptions.create.return_value = mock_whisper_response

    with (
        patch("audio2anki.transcribe.OpenAI", return_value=mock_openai),
        patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
    ):
        segments = transcribe_audio(
            audio_file,
            transcript_path=None,
            model="whisper-1",
            language="english",
            task_id=None,
            progress=None,
            min_length=1.5,  # Filter out segments shorter than 1.5 seconds
            max_length=10.0,  # Filter out segments longer than 10 seconds
        )

        # Only segments between 1.5 and 10 seconds should be included
        assert len(segments) == 1
        assert segments[0].text == "Good length"
        assert segments[0].start == 16.0
        assert segments[0].end == 18.0

        # Test caching with filters
        segments2 = transcribe_audio(
            audio_file,
            transcript_path=None,
            model="whisper-1",
            language="english",
            task_id=None,
            progress=None,
            min_length=1.5,
            max_length=10.0,
        )

        assert segments == segments2
        # Should only call the API once
        assert mock_openai.audio.transcriptions.create.call_count == 1


def test_bypass_cache(
    tmp_path: Path,
    mock_openai: Mock,
    mock_whisper_response: Mock,
    setup_cache: Generator[None, None, None],
) -> None:
    """Test bypassing the cache."""
    # Create dummy audio file
    audio_file = tmp_path / "test.mp3"
    audio_file.touch()

    # Set up mock
    mock_openai.audio.transcriptions.create.return_value = mock_whisper_response

    with (
        patch("audio2anki.transcribe.OpenAI", return_value=mock_openai),
        patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
    ):
        # First call with normal cache
        cache.init_cache(bypass=False)
        segments1 = transcribe_audio(
            audio_file,
            transcript_path=None,
            model="whisper-1",
            language="english",
            task_id=None,
            progress=None,
        )

        # Second call with bypass_cache=True
        cache.init_cache(bypass=True)
        segments2 = transcribe_audio(
            audio_file,
            transcript_path=None,
            model="whisper-1",
            language="english",
            task_id=None,
            progress=None,
        )

        assert segments1 == segments2
        # Should call the API twice
        assert mock_openai.audio.transcriptions.create.call_count == 2

"""Tests for transcription module."""

import os
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import httpx
from openai import OpenAI

from audio2anki.models import AudioSegment
from audio2anki.transcribe import load_transcript, transcribe_audio


@pytest.fixture
def mock_openai() -> Mock:
    """Create a mock OpenAI client."""
    mock = Mock(spec=OpenAI)
    mock.audio = Mock()
    mock.audio.transcriptions = Mock()
    return mock


@pytest.fixture
def mock_whisper_response() -> Mock:
    """Return mock Whisper response."""
    mock_response = Mock()
    mock_response.text = "Hello world"
    mock_response.segments = [
        {"start": 0.0, "end": 2.0, "text": "Hello"},
        {"start": 2.0, "end": 4.0, "text": "world"},
    ]
    return mock_response


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
def test_transcribe_audio(tmp_path: Path, mock_openai: Mock, mock_whisper_response: Mock) -> None:
    """Test audio transcription with OpenAI API."""
    # Create dummy audio file
    audio_file = tmp_path / "test.mp3"
    audio_file.touch()

    # Set up mock
    mock_openai.audio.transcriptions.create.return_value = mock_whisper_response

    with patch("openai.OpenAI", return_value=mock_openai):
        segments = transcribe_audio(
            audio_file,
            transcript_path=None,
            model="whisper-1",
            language="english",
        )

        # Check segments
        assert len(segments) == 2
        assert segments[0].text == "Hello"
        assert segments[1].text == "world"

        # Verify API was called correctly
        mock_openai.audio.transcriptions.create.assert_called_once()
        call_args = mock_openai.audio.transcriptions.create.call_args[1]
        assert call_args["model"] == "whisper-1"
        assert call_args["language"] == "english"
        assert call_args["response_format"] == "verbose_json"
        assert call_args["timestamp_granularities"] == ["segment"]


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
def test_transcribe_error(tmp_path: Path, mock_openai: Mock) -> None:
    """Test transcription error handling."""
    # Create dummy audio file
    audio_file = tmp_path / "test.mp3"
    audio_file.touch()

    # Set up mock error
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 401
    mock_response.text = "API error"
    mock_response.request = Mock(spec=httpx.Request)
    mock_response.request.method = "POST"
    mock_response.request.url = "https://api.openai.com/v1/audio/transcriptions"

    mock_openai.audio.transcriptions.create.side_effect = httpx.HTTPStatusError(
        "401 Unauthorized",
        request=mock_response.request,
        response=mock_response,
    )

    with patch("openai.OpenAI", return_value=mock_openai):
        with pytest.raises(RuntimeError, match="Transcription failed: 401 Unauthorized"):
            transcribe_audio(
                audio_file,
                transcript_path=None,
                model="whisper-1",
                language="english",
            )


def test_load_transcript(tmp_path: Path) -> None:
    """Test loading transcript from file."""
    # Create transcript file
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        json.dumps([
            {"start": 0.0, "end": 2.0, "text": "Hello"},
            {"start": 2.0, "end": 4.0, "text": "world"}
        ])
    )

    segments = load_transcript(transcript_path)

    assert len(segments) == 2
    assert segments[0].text == "Hello"
    assert segments[1].text == "world"


def test_load_transcript_not_found(tmp_path: Path) -> None:
    """Test loading transcript from non-existent file."""
    transcript_path = tmp_path / "nonexistent.json"

    with pytest.raises(FileNotFoundError):
        load_transcript(transcript_path)


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
def test_transcribe_with_length_filters(tmp_path: Path, mock_openai: Mock, mock_whisper_response: Mock) -> None:
    """Test transcription with length filters."""
    # Create dummy audio file
    audio_file = tmp_path / "test.mp3"
    audio_file.touch()

    # Set up mock
    mock_openai.audio.transcriptions.create.return_value = mock_whisper_response

    with patch("openai.OpenAI", return_value=mock_openai):
        # Test with min_length filter
        segments = transcribe_audio(
            audio_file,
            transcript_path=None,
            model="whisper-1",
            language="english",
            min_length=2.0,
        )

        # All segments should be >= min_length
        for segment in segments:
            assert segment.end - segment.start >= 2.0

        # Test with max_length filter
        segments = transcribe_audio(
            audio_file,
            transcript_path=None,
            model="whisper-1",
            language="english",
            max_length=3.0,
        )

        # All segments should be <= max_length
        for segment in segments:
            assert segment.end - segment.start <= 3.0

"""Tests for transcription module."""

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest
from openai import OpenAI

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
                task_id=None,
                progress=None,
            )


def test_load_transcript(tmp_path: Path) -> None:
    """Test loading transcript from file."""
    # Create test transcript file
    transcript_file = tmp_path / "transcript.json"
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Hello"},
        {"start": 2.0, "end": 4.0, "text": "world"},
    ]
    with open(transcript_file, "w") as f:
        json.dump({"segments": segments}, f)

    # Load transcript
    loaded_segments = load_transcript(transcript_file)
    assert len(loaded_segments) == 2
    assert loaded_segments[0].text == "Hello"
    assert loaded_segments[1].text == "world"


def test_load_transcript_not_found(tmp_path: Path) -> None:
    """Test loading transcript from non-existent file."""
    transcript_file = tmp_path / "nonexistent.json"
    assert load_transcript(transcript_file) is None


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
def test_transcribe_with_length_filters(tmp_path: Path, mock_openai: Mock, mock_whisper_response: Mock) -> None:
    """Test transcription with length filters."""
    # Create dummy audio file
    audio_file = tmp_path / "test.mp3"
    audio_file.touch()

    # Set up mock response with segments of different lengths
    mock_whisper_response.segments = [
        {"start": 0.0, "end": 1.0, "text": "Short"},  # 1 second
        {"start": 1.0, "end": 16.0, "text": "Too long"},  # 15 seconds
        {"start": 16.0, "end": 18.0, "text": "Good length"},  # 2 seconds
        {"start": 18.0, "end": 18.5, "text": "Too short"},  # 0.5 seconds
    ]
    mock_openai.audio.transcriptions.create.return_value = mock_whisper_response

    with patch("openai.OpenAI", return_value=mock_openai):
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

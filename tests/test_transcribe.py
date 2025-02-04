"""Tests for transcription module."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import httpx
import openai
import pytest
from rich.progress import Progress, TaskID

from audio2anki.models import AudioSegment
from audio2anki.transcribe import load_transcript, transcribe_audio


@pytest.fixture
def mock_openai() -> Mock:
    """Create a mock OpenAI client."""
    mock = Mock()
    mock_audio = MagicMock()
    mock_audio.transcriptions = MagicMock()
    mock.audio = mock_audio
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


@pytest.fixture
def progress() -> Progress:
    """Return progress bar."""
    return Progress()


@pytest.fixture
def task_id(progress: Progress) -> TaskID:
    """Create a task ID for testing."""
    return progress.add_task("test", total=100)


@patch("openai.OpenAI")
@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
def test_transcribe_audio(mock_openai_class: Mock, tmp_path: Path, progress: Progress, task_id: TaskID) -> None:
    """Test audio transcription with OpenAI API."""
    # Create dummy audio file
    audio_file = tmp_path / "test.mp3"
    audio_file.touch()

    # Set up mock response
    mock_response = MagicMock()
    mock_response.text = "Hello world"
    mock_response.segments = [
        MagicMock(start=0, end=2, text="Hello"),
        MagicMock(start=2, end=4, text="world"),
    ]
    mock_client = mock_openai_class.return_value
    mock_client.audio.transcriptions.create.return_value = mock_response

    # Mock OpenAI client creation
    with patch("audio2anki.transcribe.OpenAI", return_value=mock_client):
        segments = transcribe_audio(
            audio_file,
            transcript_path=None,
            model="whisper-1",
            language="english",
            task_id=task_id,
            progress=progress,
        )

        # Check segments
        assert len(segments) == 2
        assert segments[0].text == "Hello"
        assert segments[1].text == "world"

        # Verify API was called correctly
        mock_client.audio.transcriptions.create.assert_called_once()
        call_args = mock_client.audio.transcriptions.create.call_args[1]
        assert call_args["model"] == "whisper-1"
        assert call_args["language"] == "english"
        assert call_args["response_format"] == "verbose_json"


@patch("openai.OpenAI")
@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
def test_transcribe_error(mock_openai_class: Mock, tmp_path: Path, progress: Progress, task_id: TaskID) -> None:
    """Test transcription error handling."""
    # Create dummy audio file
    audio_file = tmp_path / "test.mp3"
    audio_file.touch()

    # Set up mock error
    mock_client = mock_openai_class.return_value
    mock_client.audio.transcriptions.create.side_effect = openai.AuthenticationError(
        "Incorrect API key provided",
        response=Mock(status_code=401),
        body={"error": {"message": "Incorrect API key provided"}}
    )

    # Mock OpenAI client creation
    with patch("audio2anki.transcribe.OpenAI", return_value=mock_client):
        with pytest.raises(openai.AuthenticationError) as exc_info:
            transcribe_audio(
                audio_file,
                transcript_path=None,
                model="whisper-1",
                language="english",
                task_id=task_id,
                progress=progress,
            )

        assert "Incorrect API key provided" in str(exc_info.value)


@patch("openai.OpenAI")
@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
def test_transcribe_with_length_filters(
    mock_openai_class: Mock, tmp_path: Path, progress: Progress, task_id: TaskID
) -> None:
    """Test transcription with length filters."""
    # Create dummy audio file
    audio_file = tmp_path / "test.mp3"
    audio_file.touch()

    # Set up mock response
    mock_response = MagicMock()
    mock_response.segments = [
        MagicMock(start=0, end=1, text="Too short"),  # Should be filtered out
        MagicMock(start=1, end=3, text="Just right"),  # Should be kept
        MagicMock(start=3, end=7, text="Too long"),  # Should be filtered out
    ]
    mock_client = mock_openai_class.return_value
    mock_client.audio.transcriptions.create.return_value = mock_response

    # Mock OpenAI client creation
    with patch("audio2anki.transcribe.OpenAI", return_value=mock_client):
        # Test with length filters
        segments = transcribe_audio(
            audio_file,
            transcript_path=None,
            model="whisper-1",
            task_id=task_id,
            progress=progress,
            min_length=1.5,  # Only keep segments longer than 1.5s
            max_length=3.0,  # Only keep segments shorter than 3.0s
        )

        # Check segments
        assert len(segments) == 1
        assert segments[0].text == "Just right"

        # Verify API was called correctly
        mock_client.audio.transcriptions.create.assert_called_once()
        call_args = mock_client.audio.transcriptions.create.call_args[1]
        assert call_args["model"] == "whisper-1"
        assert call_args["response_format"] == "verbose_json"


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

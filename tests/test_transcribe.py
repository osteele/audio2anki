"""Tests for transcription module."""

import logging
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from openai import AuthenticationError, OpenAI
from openai.types.audio import Transcription
from pydub import AudioSegment

from audio2anki.transcribe import TranscriptionSegment, load_transcript, save_transcript, transcribe_audio
from audio2anki.types import LanguageCode

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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


def test_transcribe_audio(
    tmp_path: Path,
    mock_openai: Mock,
    mock_whisper_response: Mock,
) -> None:
    """Test audio transcription with OpenAI API."""
    # Create a minimal valid audio file (1 second of silence)
    audio_file = tmp_path / "test.mp3"
    silence = AudioSegment.silent(duration=1000)  # 1 second
    silence.export(audio_file, format="mp3")

    # Create output file path
    transcript_path = tmp_path / "transcript.srt"

    # Set up mock
    mock_openai.audio.transcriptions.create.return_value = mock_whisper_response

    with (
        patch("audio2anki.transcribe.OpenAI", return_value=mock_openai),
        patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
    ):
        # Call the transcribe function
        segments = transcribe_audio(
            audio_file,
            transcript_path=transcript_path,
            language=LanguageCode("en"),
            task_id=None,
            progress=None,
        )

        # Verify the segments are correct
        assert len(segments) == 2
        assert segments[0].text == "Hello"
        assert segments[1].text == "world"
        assert segments[0].start == 0.0
        assert segments[0].end == 2.0
        assert segments[1].start == 2.0
        assert segments[1].end == 4.0

        # Verify the transcript file was created
        assert transcript_path.exists()

        # Verify the API was called
        mock_openai.audio.transcriptions.create.assert_called_once()


def test_transcribe_error(tmp_path: Path, mock_openai: Mock) -> None:
    """Test transcription error handling."""
    # Create dummy audio file
    audio_file = tmp_path / "test.mp3"
    audio_file.touch()

    # Create output file path
    transcript_path = tmp_path / "transcript.srt"

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
                transcript_path=transcript_path,
                language=LanguageCode("en"),
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
) -> None:
    """Test transcription with length filters."""
    # Create dummy audio file
    audio_file = tmp_path / "test.mp3"
    audio_file.touch()

    # Create output file path
    transcript_path = tmp_path / "transcript.srt"

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
        # Call transcribe_audio with length filters
        segments = transcribe_audio(
            audio_file,
            transcript_path=transcript_path,
            language=LanguageCode("en"),
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

        # Verify the transcript file was created and contains only the filtered segments
        assert transcript_path.exists()

        # Load the created transcript and verify it contains only the filtered segments
        loaded_segments = load_transcript(transcript_path)
        assert len(loaded_segments) == 1
        assert loaded_segments[0].text == "Good length"


def test_load_srt_transcript(tmp_path: Path) -> None:
    """Test loading SRT transcript."""
    transcript_file = tmp_path / "transcript.srt"

    # Create test files
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write("1\n00:00:00,000 --> 00:00:02,000\nHello\n\n")
        f.write("2\n00:00:02,000 --> 00:00:04,000\nworld\n")

    # Test loading from SRT
    segments = load_transcript(transcript_file)
    assert len(segments) == 2
    assert segments[0].text == "Hello"
    assert segments[1].text == "world"
    assert segments[0].start == 0.0
    assert segments[0].end == 2.0
    assert segments[1].start == 2.0
    assert segments[1].end == 4.0

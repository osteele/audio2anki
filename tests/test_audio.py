"""Tests for the audio module."""

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from pydub import AudioSegment as PydubSegment
from pydub.generators import Sine
from rich.progress import Progress

from audio2anki.audio import (
    MAX_CHUNK_DURATION,
    MAX_CLEAN_FILE_SIZE,
    AudioCleaningError,
    clean_audio,
    split_audio,
    split_large_audio,
    trim_silence,
)
from audio2anki.models import AudioSegment


@pytest.fixture
def segments() -> list[AudioSegment]:
    """Return test segments."""
    return [
        AudioSegment(start=0.0, end=2.0, text="First segment"),
        AudioSegment(start=2.0, end=4.0, text="Second segment"),
    ]


def mock_export(
    output_path: Path, format: str = "mp3", **kwargs: Dict[str, Any]
) -> None:
    """Mock export function that creates empty files."""
    output_path.touch()


def create_test_wav(path: Path, duration_ms: int = 1000) -> None:
    """Create a test WAV file."""
    audio = Sine(440).to_audio_segment(duration=duration_ms)
    audio.export(path, format="wav")


@patch("pydub.AudioSegment.from_file")
@patch("audio2anki.audio.PydubSegment.from_file")
def test_split_audio(
    mock_audio_from_file: Mock,
    mock_split_from_file: Mock,
    segments: list[AudioSegment],
    progress: Progress,
    tmp_path: Path,
) -> None:
    """Test audio splitting functionality."""
    # Set up mock
    mock_audio = MagicMock()
    mock_split_from_file.return_value = mock_audio
    mock_audio.__getitem__.return_value = mock_audio  # Return same mock for any slice
    mock_audio.export.side_effect = mock_export

    # Create test files
    input_file = tmp_path / "test.mp3"
    input_file.write_bytes(b"test audio data")  # Write some test data
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "media").mkdir()

    # Mock clean_audio to return the input file
    with patch("audio2anki.audio.clean_audio", return_value=input_file):
        # Initialize progress bar
        task_id = progress.add_task("Splitting audio", total=len(segments))

        # Split audio
        result = split_audio(input_file, segments, output_dir, task_id, progress)

        # Check results
        assert len(result) == len(segments)
        assert all(s.audio_file is not None for s in result)
        assert all(
            s.audio_file is not None
            and s.audio_file.startswith("audio2anki_")
            and s.audio_file.endswith(".mp3")
            for s in result
        )
        assert all(
            s.audio_file is not None and (output_dir / "media" / s.audio_file).exists()
            for s in result
        )

        # Verify mock calls
        assert mock_split_from_file.call_count == 1  # Only for split_audio
        # Convert milliseconds to samples for slice arguments
        # Each segment gets sliced twice: once for extraction, once for trim_silence
        expected_slices = [
            call(slice(0, 2000)),  # First segment extraction
            call(slice(0, 0)),  # First segment trim
            call(slice(2000, 4000)),  # Second segment extraction
            call(slice(0, 0)),  # Second segment trim
        ]
        mock_audio.__getitem__.assert_has_calls(expected_slices)
        assert mock_audio.export.call_count == len(segments)


@patch("pydub.AudioSegment.from_file")
@patch("audio2anki.audio.PydubSegment.from_file")
def test_split_audio_empty_segments(
    mock_audio_from_file: Mock,
    mock_split_from_file: Mock,
    progress: Progress,
    tmp_path: Path,
) -> None:
    """Test handling of empty segment list."""
    # Set up mock
    mock_audio = MagicMock()
    mock_split_from_file.return_value = mock_audio
    mock_audio.__getitem__.return_value = mock_audio
    mock_audio.export.side_effect = mock_export

    # Create test files
    input_file = tmp_path / "test.mp3"
    input_file.write_bytes(b"test audio data")  # Write some test data
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "media").mkdir()

    # Mock clean_audio to return the input file
    with patch("audio2anki.audio.clean_audio", return_value=input_file):
        # Initialize progress bar
        task_id = progress.add_task("Splitting audio", total=0)

        # Split audio with empty segments list
        result = split_audio(input_file, [], output_dir, task_id, progress)

        # Check results
        assert len(result) == 0
        assert (
            mock_split_from_file.call_count == 0
        )  # Should not load audio for empty segments
        assert mock_audio.__getitem__.call_count == 0
        assert mock_audio.export.call_count == 0


@patch("pydub.AudioSegment.from_file")
@patch("audio2anki.audio.PydubSegment.from_file")
def test_split_audio_existing_media_dir(
    mock_audio_from_file: Mock,
    mock_split_from_file: Mock,
    segments: list[AudioSegment],
    progress: Progress,
    tmp_path: Path,
) -> None:
    """Test handling of existing media directory."""
    # Set up mock
    mock_audio = MagicMock()
    mock_split_from_file.return_value = mock_audio
    mock_audio.__getitem__.return_value = mock_audio  # Return same mock for any slice
    mock_audio.export.side_effect = mock_export

    # Create test files
    input_file = tmp_path / "test.mp3"
    input_file.write_bytes(b"test audio data")  # Write some test data
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "media").mkdir()

    # Mock clean_audio to return the input file
    with patch("audio2anki.audio.clean_audio", return_value=input_file):
        # Initialize progress bar
        task_id = progress.add_task("Splitting audio", total=len(segments))

        # Split audio
        result = split_audio(input_file, segments, output_dir, task_id, progress)

        # Check results
        assert len(result) == len(segments)
        assert all(s.audio_file is not None for s in result)
        assert all(
            s.audio_file is not None
            and s.audio_file.startswith("audio2anki_")
            and s.audio_file.endswith(".mp3")
            for s in result
        )
        assert all(
            s.audio_file is not None and (output_dir / "media" / s.audio_file).exists()
            for s in result
        )

        # Verify mock calls
        assert mock_split_from_file.call_count == 1
        # Convert milliseconds to samples for slice arguments
        # Each segment gets sliced twice: once for extraction, once for trim_silence
        expected_slices = [
            call(slice(0, 2000)),  # First segment extraction
            call(slice(0, 0)),  # First segment trim
            call(slice(2000, 4000)),  # Second segment extraction
            call(slice(0, 0)),  # Second segment trim
        ]
        mock_audio.__getitem__.assert_has_calls(expected_slices)
        assert mock_audio.export.call_count == len(segments)


@patch("audio2anki.audio.detect_nonsilent")
def test_trim_silence(mock_detect_nonsilent: Mock) -> None:
    """Test silence trimming functionality."""
    # Create a mock audio segment
    audio = MagicMock(spec=PydubSegment)
    audio.__len__.return_value = 10000  # 10 seconds

    # Test case 1: Normal case with silence at both ends
    mock_detect_nonsilent.return_value = [
        (1000, 8000)
    ]  # Non-silent section from 1s to 8s
    trimmed = trim_silence(audio)
    audio.__getitem__.assert_called_with(slice(1000, 8000))

    # Test case 2: No silence to trim
    mock_detect_nonsilent.return_value = [(0, 10000)]
    trimmed = trim_silence(audio)
    audio.__getitem__.assert_called_with(slice(0, 10000))

    # Test case 3: All silence
    mock_detect_nonsilent.return_value = []
    trimmed = trim_silence(audio)
    assert trimmed == audio  # Should return original audio


@patch("audio2anki.audio.Client")
def test_clean_audio_small_file(
    mock_client_class: Mock,
    mock_hf_client: Mock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    progress: Progress,
) -> None:
    """Test cleaning a small audio file."""
    # Set up environment and mocks
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    mock_client_class.return_value = mock_hf_client

    # Create test file under size limit
    input_file = tmp_path / "small.wav"
    create_test_wav(input_file)

    # Mock the job
    mock_job = MagicMock()
    mock_job.done.return_value = True
    cleaned_file = tmp_path / "cleaned.wav"
    cleaned_file.touch()
    mock_job.result.return_value = [str(cleaned_file)]
    mock_hf_client.submit.return_value = mock_job

    # Clean audio with force mode
    task_id = progress.add_task("Cleaning audio", total=1)
    result = clean_audio(input_file, progress, task_id, clean_mode="force")
    assert result is not None
    assert result != input_file  # Should return a different file (the cleaned one)
    assert mock_client_class.called_once_with(
        "anyantudre/resemble-enhance-demo", hf_token="fake-token"
    )
    mock_hf_client.submit.assert_called_once()


@patch("audio2anki.audio.Client")
def test_clean_audio_large_file(
    mock_client_class: Mock,
    mock_hf_client: Mock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    progress: Progress,
) -> None:
    """Test cleaning a large audio file."""
    # Set up environment and mocks
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    mock_client_class.return_value = mock_hf_client

    # Create test file over size limit
    input_file = tmp_path / "large.wav"
    input_file.write_bytes(b"x" * (MAX_CLEAN_FILE_SIZE + 1))

    # Mock audio loading and manipulation
    mock_audio = MagicMock()
    mock_audio.__len__.return_value = (
        MAX_CHUNK_DURATION * 3
    )  # Three chunks total duration
    # Each chunk should be 60s or less
    mock_chunk = MagicMock()
    mock_chunk.__len__.return_value = MAX_CHUNK_DURATION
    mock_audio.__getitem__.return_value = mock_chunk
    mock_chunk.export.side_effect = lambda path, format: Path(path).touch()

    with patch("audio2anki.audio.PydubSegment") as mock_pydub:
        mock_pydub.from_file.return_value = mock_audio
        mock_pydub.empty.return_value = mock_audio

        # Mock the job
        mock_job = MagicMock()
        mock_job.done.return_value = True
        cleaned_file = tmp_path / "cleaned.wav"
        cleaned_file.touch()
        mock_job.result.return_value = [str(cleaned_file)]
        mock_hf_client.submit.return_value = mock_job

        # Clean audio with force mode
        task_id = progress.add_task("Cleaning audio", total=3)
        result = clean_audio(input_file, progress, task_id, clean_mode="force")
        assert result is not None
        assert mock_client_class.called_once_with(
            "anyantudre/resemble-enhance-demo", hf_token="fake-token"
        )
        assert mock_hf_client.submit.call_count == 3


def test_clean_audio_no_token(tmp_path: Path, progress: Progress) -> None:
    """Test cleaning without HF_TOKEN."""
    input_file = tmp_path / "test.wav"
    create_test_wav(input_file)

    # Should return None in default mode
    with patch.dict("os.environ", {}, clear=True):  # Clear all env vars
        task_id = progress.add_task("Cleaning audio", total=1)
        result = clean_audio(input_file, progress, task_id, clean_mode="auto")
        assert result is None

        # Should raise error in force mode
        with pytest.raises(
            AudioCleaningError,
            match="HF_TOKEN environment variable is required for audio cleaning",
        ):
            clean_audio(input_file, progress, task_id, clean_mode="force")


def test_clean_audio_skip_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    progress: Progress,
) -> None:
    """Test skip mode always returns None."""
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    input_file = tmp_path / "test.wav"
    create_test_wav(input_file)

    task_id = progress.add_task("Cleaning audio", total=1)
    result = clean_audio(input_file, progress, task_id, clean_mode="skip")
    assert result is None


def test_split_large_audio() -> None:
    """Test splitting audio into chunks."""
    # Create mock audio of 12 minutes (in milliseconds)
    duration_ms = int(MAX_CHUNK_DURATION * 2.4)  # Should create 3 chunks
    mock_audio = MagicMock()
    mock_audio.__len__.return_value = duration_ms
    mock_audio.__getitem__.return_value = mock_audio

    chunks = list(split_large_audio(mock_audio))
    assert len(chunks) == 3

    # Check chunk sizes (in milliseconds)
    assert mock_audio.__getitem__.call_count == 3
    mock_audio.__getitem__.assert_any_call(slice(0, MAX_CHUNK_DURATION))
    mock_audio.__getitem__.assert_any_call(
        slice(MAX_CHUNK_DURATION, MAX_CHUNK_DURATION * 2)
    )
    mock_audio.__getitem__.assert_any_call(slice(MAX_CHUNK_DURATION * 2, duration_ms))

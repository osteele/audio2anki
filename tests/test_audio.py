"""Tests for the audio module."""

from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from rich.progress import Progress

from audio2anki.audio import split_audio
from audio2anki.models import AudioSegment


@pytest.fixture
def segments() -> list[AudioSegment]:
    """Return test segments."""
    return [
        AudioSegment(start=0.0, end=2.0, text="First segment"),
        AudioSegment(start=2.0, end=4.0, text="Second segment"),
    ]


def mock_export(output_path: Path, format: str, parameters: list[str]) -> None:
    """Mock export function that creates empty files."""
    output_path.touch()


@patch("pydub.AudioSegment.from_file")
def test_split_audio(
    mock_from_file: Mock,
    segments: list[AudioSegment],
    progress: Progress,
    tmp_path: Path,
) -> None:
    """Test audio splitting functionality."""
    # Set up mock
    mock_audio = MagicMock()
    mock_from_file.return_value = mock_audio
    mock_audio.__getitem__.return_value = mock_audio
    mock_audio.export.side_effect = mock_export

    # Create test files
    input_file = tmp_path / "test.mp3"
    input_file.write_bytes(b"test audio data")  # Write some test data
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Initialize progress bar
    task_id = progress.add_task("Splitting audio", total=len(segments))

    # Split audio
    result = split_audio(input_file, segments, output_dir, task_id, progress)

    # Check results
    assert len(result) == len(segments)
    assert all(s.audio_file is not None for s in result)
    assert all(s.audio_file.startswith("audio_") and s.audio_file.endswith(".mp3") for s in result)
    assert all((output_dir / "media" / s.audio_file).exists() for s in result)

    # Verify mock calls
    assert mock_from_file.call_count == 1
    assert mock_audio.__getitem__.call_count == len(segments)
    assert mock_audio.export.call_count == len(segments)


@patch("pydub.AudioSegment.from_file")
def test_split_audio_empty_segments(
    mock_from_file: Mock,
    progress: Progress,
    tmp_path: Path,
) -> None:
    """Test handling of empty segment list."""
    # Set up mock
    mock_audio = MagicMock()
    mock_from_file.return_value = mock_audio
    mock_audio.__getitem__.return_value = mock_audio
    mock_audio.export.side_effect = mock_export

    # Create test files
    input_file = tmp_path / "test.mp3"
    input_file.write_bytes(b"test audio data")  # Write some test data
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Initialize progress bar
    task_id = progress.add_task("Splitting audio", total=0)

    # Split audio
    result = split_audio(input_file, [], output_dir, task_id, progress)

    # Check results
    assert len(result) == 0
    assert mock_from_file.call_count == 1
    assert mock_audio.__getitem__.call_count == 0
    assert mock_audio.export.call_count == 0


@patch("pydub.AudioSegment.from_file")
def test_split_audio_existing_media_dir(
    mock_from_file: Mock,
    segments: list[AudioSegment],
    progress: Progress,
    tmp_path: Path,
) -> None:
    """Test handling of existing media directory."""
    # Set up mock
    mock_audio = MagicMock()
    mock_from_file.return_value = mock_audio
    mock_audio.__getitem__.return_value = mock_audio
    mock_audio.export.side_effect = mock_export

    # Create test files
    input_file = tmp_path / "test.mp3"
    input_file.write_bytes(b"test audio data")  # Write some test data
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "media").mkdir()

    # Initialize progress bar
    task_id = progress.add_task("Splitting audio", total=len(segments))

    # Split audio
    result = split_audio(input_file, segments, output_dir, task_id, progress)

    # Check results
    assert len(result) == len(segments)
    assert all(s.audio_file is not None for s in result)
    assert all(s.audio_file.startswith("audio_") and s.audio_file.endswith(".mp3") for s in result)
    assert all((output_dir / "media" / s.audio_file).exists() for s in result)

    # Verify mock calls
    assert mock_from_file.call_count == 1
    assert mock_audio.__getitem__.call_count == len(segments)
    assert mock_audio.export.call_count == len(segments)

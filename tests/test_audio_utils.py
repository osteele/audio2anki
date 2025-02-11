"""Tests for audio utilities."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from pydub import AudioSegment as PydubSegment
from rich.progress import Progress, TaskID

from audio2anki.audio_utils import compute_file_hash, split_audio, trim_silence
from audio2anki.models import AudioSegment


@pytest.fixture
def test_audio_file(tmp_path: Path) -> Path:
    """Create a test audio file."""
    audio = PydubSegment.silent(duration=1000)  # 1 second of silence
    file_path = tmp_path / "test_audio.wav"
    audio.export(str(file_path), format="wav")
    return file_path


def test_split_audio(test_audio_file: Path, tmp_path: Path) -> None:
    """Test audio splitting functionality."""
    # Create test segments
    segments = [
        AudioSegment(start=0.0, end=0.5, text="First"),
        AudioSegment(start=0.5, end=1.0, text="Second"),
    ]

    # Create mock progress tracking
    progress = Mock(spec=Progress)
    task_id = Mock(spec=TaskID)

    # Create output directory
    output_dir = tmp_path / "media"

    # Split audio
    result_segments = split_audio(test_audio_file, segments, output_dir, task_id, progress)

    # Check that output files were created
    assert output_dir.exists()
    assert len(list(output_dir.glob("*.mp3"))) == 2

    # Check that segments were updated with audio file names
    assert all(seg.audio_file is not None for seg in result_segments)
    for segment in result_segments:
        assert segment.audio_file is not None
        assert segment.audio_file.startswith("audio2anki_")
        assert segment.audio_file.endswith(".mp3")
        assert (output_dir / segment.audio_file).exists()


def test_trim_silence() -> None:
    """Test silence trimming functionality."""
    # Create a test audio segment with silence at both ends
    silence = PydubSegment.silent(duration=500)  # 500ms of silence
    # Create non-silent audio by modifying the samples
    non_silent = PydubSegment.silent(duration=500)
    non_silent = non_silent._spawn(b'\x01' * (len(non_silent.raw_data)))  # Non-zero samples
    # Combine silence and non-silent parts
    audio = silence + non_silent + silence  # 1.5 seconds total
    trimmed = trim_silence(audio)
    assert len(trimmed) < len(audio)  # Should have trimmed the silence


def test_compute_file_hash(test_audio_file: Path) -> None:
    """Test file hash computation."""
    hash1 = compute_file_hash(test_audio_file)
    hash2 = compute_file_hash(test_audio_file)

    # Same file should produce same hash
    assert hash1 == hash2
    # Hash should be 8 characters long
    assert len(hash1) == 8
    # Hash should be hexadecimal
    assert all(c in "0123456789abcdef" for c in hash1)

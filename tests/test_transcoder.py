"""Tests for the audio transcoder module."""

import os
from pathlib import Path

import pytest
from pydub import AudioSegment

from audio2anki.transcoder import transcode_audio


@pytest.fixture
def test_audio_file(tmp_path: Path) -> Path:
    """Create a test audio file."""
    audio = AudioSegment.silent(duration=1000)  # 1 second of silence
    file_path = tmp_path / "test_audio.wav"
    audio.export(str(file_path), format="wav")
    return file_path


def test_transcode_audio_creates_mp3(test_audio_file: Path, tmp_path: Path) -> None:
    """Test that transcode_audio creates an MP3 file."""
    output_path = transcode_audio(test_audio_file)
    assert output_path.exists()
    assert output_path.suffix == ".mp3"


def test_transcode_audio_with_progress(test_audio_file: Path) -> None:
    """Test that transcode_audio calls progress callback."""
    progress_values = []

    def progress_callback(value: float) -> None:
        progress_values.append(value)

    transcode_audio(test_audio_file, progress_callback=progress_callback)
    assert len(progress_values) > 0
    assert 100 in progress_values  # Final progress value should be 100


def test_transcode_audio_with_custom_params(test_audio_file: Path) -> None:
    """Test that transcode_audio respects custom parameters."""
    output_path = transcode_audio(
        test_audio_file,
        target_channels=1,
        target_sample_rate=22050,
    )
    assert output_path.suffix == ".mp3"
    audio = AudioSegment.from_file(str(output_path))
    assert audio.channels == 1
    assert audio.frame_rate == 22050


def test_transcode_audio_nonexistent_file() -> None:
    """Test that transcode_audio raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        transcode_audio("nonexistent.mp3")


def test_transcode_audio_caches_result(test_audio_file: Path) -> None:
    """Test that transcode_audio caches results."""
    # First transcoding
    output_path = transcode_audio(test_audio_file)
    mtime = os.path.getmtime(output_path)

    # Second transcoding should use cached file
    output_path2 = transcode_audio(test_audio_file)
    assert output_path == output_path2
    # Allow small difference in mtime due to filesystem precision
    assert abs(os.path.getmtime(output_path2) - mtime) < 0.1

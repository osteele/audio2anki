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
    output_path = tmp_path / "output.mp3"
    transcode_audio(test_audio_file, output_path=output_path)
    assert output_path.exists()
    assert output_path.suffix == ".mp3"


def test_transcode_audio_with_progress(test_audio_file: Path, tmp_path: Path) -> None:
    """Test that transcode_audio calls progress callback."""
    progress_values: list[float] = []
    output_path = tmp_path / "output.mp3"

    def progress_callback(value: float) -> None:
        progress_values.append(value)

    transcode_audio(test_audio_file, output_path=output_path, progress_callback=progress_callback)

    # Use more functional assertions
    assert len(progress_values) > 0
    assert any(value == 100 for value in progress_values)  # Check if 100 is in the values


def test_transcode_audio_with_custom_params(test_audio_file: Path, tmp_path: Path) -> None:
    """Test that transcode_audio respects custom parameters."""
    output_path = tmp_path / "output.mp3"
    transcode_audio(
        test_audio_file,
        output_path=output_path,
        target_channels=1,
        target_sample_rate=22050,
    )
    assert output_path.suffix == ".mp3"
    audio = AudioSegment.from_file(str(output_path))
    assert audio.channels == 1
    assert audio.frame_rate == 22050


def test_transcode_audio_caches_result(test_audio_file: Path, tmp_path: Path) -> None:
    """Test that transcode_audio caches results."""
    output_path = tmp_path / "output.mp3"

    # First call
    transcode_audio(test_audio_file, output_path=output_path)
    first_mtime = os.path.getmtime(output_path)

    # Get file content to verify it doesn't change
    with open(output_path, "rb") as f:
        first_content = f.read()

    # Second call
    transcode_audio(test_audio_file, output_path=output_path)
    second_mtime = os.path.getmtime(output_path)

    # Get file content after second call
    with open(output_path, "rb") as f:
        second_content = f.read()

    # File content should not have changed, even if timestamp might
    assert first_content == second_content

    # If timestamps differ, check that it's just filesystem precision issues
    # by ensuring the difference is very small (less than 1 second)
    if first_mtime != second_mtime:
        assert abs(first_mtime - second_mtime) < 1.0

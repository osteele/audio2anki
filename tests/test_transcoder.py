"""Tests for the audio transcoder module."""

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
    assert len(progress_values) > 0
    assert 100 in progress_values  # Final progress value should be 100


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

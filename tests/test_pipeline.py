"""Tests for the pipeline module."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from rich.progress import Progress

from audio2anki.pipeline import PipelineContext, PipelineProgress, generate_deck


@pytest.fixture
def mock_progress() -> Progress:
    """Create a mock progress bar."""
    mock = Mock(spec=Progress)
    mock.update = Mock()  # Explicitly create update method
    return mock


@pytest.fixture
def mock_pipeline_progress(mock_progress: Progress) -> PipelineProgress:
    """Create a mock pipeline progress tracker."""
    progress = Mock(spec=PipelineProgress)
    progress.progress = mock_progress
    progress.current_stage = "generate_deck"
    progress.stage_tasks = {"generate_deck": Mock()}
    progress.update_stage = Mock()
    return progress


def test_generate_deck(tmp_path: Path, mock_pipeline_progress: PipelineProgress) -> None:
    """Test deck generation stage."""
    # Create test files
    transcript_file = tmp_path / "transcript.srt"
    with open(transcript_file, "w") as f:
        f.write("1\n00:00:00,000 --> 00:00:02,000\n你好\n\n")
        f.write("2\n00:00:02,000 --> 00:00:04,000\n谢谢\n")

    translation_file = tmp_path / "translation.srt"
    with open(translation_file, "w") as f:
        f.write("1\n00:00:00,000 --> 00:00:02,000\nHello\n\n")
        f.write("2\n00:00:02,000 --> 00:00:04,000\nThank you\n")

    pronunciation_file = tmp_path / "pronunciation.srt"
    with open(pronunciation_file, "w") as f:
        f.write("1\n00:00:00,000 --> 00:00:02,000\nNǐ hǎo\n\n")
        f.write("2\n00:00:02,000 --> 00:00:04,000\nXièxie\n")

    # Create test audio file
    from pydub import AudioSegment as PydubSegment
    audio = PydubSegment.silent(duration=4000)  # 4 seconds of silence
    audio_file = tmp_path / "test_audio.mp3"
    audio.export(str(audio_file), format="mp3")

    # Create context
    context = PipelineContext(
        primary=translation_file,
        isolated_audio=audio_file,
        transcription_srt=transcript_file,
        pronunciation_srt=pronunciation_file,
        source_language="chinese",
        target_language="english",
    )

    # Change to tmp_path as working directory
    import os

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # Run deck generation
        context = generate_deck(context, mock_pipeline_progress)

        # Check output
        deck_dir = Path.cwd() / "deck"
        assert deck_dir.exists()
        assert deck_dir.is_dir()
        assert (deck_dir / "deck.txt").exists()
        assert (deck_dir / "README.md").exists()
        assert (deck_dir / "media").exists()

        # Check deck.txt content
        with open(deck_dir / "deck.txt") as f:
            content = f.read().splitlines()
            assert len(content) == 3  # Header + 2 segments
            assert content[0] == "Hanzi\tPinyin\tEnglish\tAudio"
            # Split each line into fields and check each field separately
            fields1 = content[1].split("\t")
            fields2 = content[2].split("\t")
            # Check text
            assert fields1[0] == "你好"
            assert fields2[0] == "谢谢"
            # Check pronunciation
            assert fields1[1] == "Nǐ hǎo"
            assert fields2[1] == "Xièxie"
            # Check translation
            assert fields1[2] == "Hello"
            assert fields2[2] == "Thank you"
            # Check audio (just verify it exists)
            assert fields1[3].startswith("[sound:")
            assert fields2[3].startswith("[sound:")
    finally:
        os.chdir(old_cwd)

"""Tests for the Anki deck generation module."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from rich.progress import Progress, TaskID

from audio2anki.anki import create_anki_deck, process_deck
from audio2anki.models import AudioSegment, TranscriptionSegment
from audio2anki.pipeline import PipelineProgress


@pytest.fixture
def segments() -> list[AudioSegment]:
    """Return test segments."""
    return [
        TranscriptionSegment(
            start=0.0,
            end=2.0,
            text="你好",
            pronunciation="Nǐ hǎo",
            translation="Hello",
            audio_file="audio_0001.mp3",
        ),
        TranscriptionSegment(
            start=2.0,
            end=4.0,
            text="谢谢",
            pronunciation="Xièxie",
            translation="Thank you",
            audio_file="audio_0002.mp3",
        ),
    ]


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
    progress.stage_tasks = {"generate_deck": Mock(spec=TaskID)}
    return progress


def test_create_anki_deck(segments: list[AudioSegment], tmp_path: Path) -> None:
    """Test Anki deck creation."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    deck_dir = create_anki_deck(
        segments,
        output_dir,
        source_language="chinese",
        target_language="english",
    )

    # Check deck directory structure
    assert deck_dir.exists()
    assert deck_dir.is_dir()
    assert (deck_dir / "media").exists()
    assert (deck_dir / "media").is_dir()
    assert (deck_dir / "deck.txt").exists()
    assert (deck_dir / "README.md").exists()

    # Check deck.txt content
    with open(deck_dir / "deck.txt") as f:
        content = f.read().splitlines()
        assert len(content) == 3  # Header + 2 segments
        assert content[0] == "Hanzi\tPinyin\tEnglish\tAudio"
        assert content[1] == "你好\tNǐ hǎo\tHello\t[sound:audio_0001.mp3]"
        assert content[2] == "谢谢\tXièxie\tThank you\t[sound:audio_0002.mp3]"

    # Check README.md exists and has content
    with open(deck_dir / "README.md") as f:
        readme = f.read()
        assert "Anki Deck Import Instructions" in readme
        assert "media folder" in readme


def test_create_anki_deck_missing_fields(tmp_path: Path) -> None:
    """Test deck creation with missing optional fields."""
    segments = [
        TranscriptionSegment(
            start=0.0,
            end=2.0,
            text="Test",
        )
    ]

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    deck_dir = create_anki_deck(segments, output_dir)

    # Check deck.txt content
    with open(deck_dir / "deck.txt") as f:
        content = f.read().splitlines()
        assert len(content) == 2  # Header + 1 segment
        assert content[0] == "Text\tPronunciation\tTranslation\tAudio"
        assert content[1] == "Test\t\t\t"  # Empty optional fields


def test_create_anki_deck_with_progress(
    segments: list[AudioSegment], tmp_path: Path, mock_progress: Progress
) -> None:
    """Test deck creation with progress tracking."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    task_id = Mock(spec=TaskID)

    deck_dir = create_anki_deck(
        segments,
        output_dir,
        task_id,
        mock_progress,
        source_language="chinese",
        target_language="english",
    )

    # Just verify the deck was created successfully
    assert deck_dir.exists()
    assert (deck_dir / "deck.txt").exists()
    assert (deck_dir / "media").exists()


def test_process_deck(
    segments: list[AudioSegment],
    tmp_path: Path,
    mock_pipeline_progress: PipelineProgress,
) -> None:
    """Test deck processing in pipeline."""
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

    # Create deck directory structure
    deck_dir = tmp_path / "deck"
    deck_dir.mkdir()
    media_dir = deck_dir / "media"
    media_dir.mkdir()

    # Change to tmp_path as working directory
    import os

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        deck_dir = process_deck(
            translation_file,
            mock_pipeline_progress,
            transcription_file=transcript_file,
            pronunciation_file=pronunciation_file,
            source_language="chinese",
            target_language="english",
        )

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
            assert content[1].startswith("你好\tNǐ hǎo\tHello\t")  # Audio filename will be dynamic
            assert content[2].startswith("谢谢\tXièxie\tThank you\t")
    finally:
        os.chdir(old_cwd)

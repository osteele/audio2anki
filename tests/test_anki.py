"""Tests for the Anki deck generation module."""

import csv
from pathlib import Path

import pytest

from audio2anki.anki import create_anki_deck
from audio2anki.models import AudioSegment, TranscriptionSegment


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


def test_create_anki_deck(segments: list[AudioSegment], tmp_path: Path) -> None:
    """Test Anki deck creation."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    deck_file = create_anki_deck(segments, output_dir)

    assert deck_file.exists()
    with open(deck_file) as f:
        content = f.read().splitlines()
        assert len(content) == 3  # Header + 2 segments
        assert content[0] == "Text\tPronunciation\tTranslation\tAudio"
        assert content[1] == "你好\tNǐ hǎo\tHello\t[sound:audio_0001.mp3]"
        assert content[2] == "谢谢\tXièxie\tThank you\t[sound:audio_0002.mp3]"


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

    deck_file = create_anki_deck(segments, output_dir)

    assert deck_file.exists()
    with open(deck_file) as f:
        content = f.read().splitlines()
        assert len(content) == 2  # Header + 1 segment
        assert content[0] == "Text\tPronunciation\tTranslation\tAudio"
        assert content[1] == "Test\t\t\t"  # Empty optional fields


def test_create_anki_deck_empty_segments(tmp_path: Path) -> None:
    """Test deck creation with no segments."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    deck_file = create_anki_deck([], output_dir)

    # Read and verify content
    with open(deck_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)

    # Should only have header
    assert len(rows) == 1
    assert rows[0] == ["Text", "Pronunciation", "Translation", "Audio"]

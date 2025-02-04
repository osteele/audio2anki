"""Tests for the Anki module."""

import csv
from pathlib import Path

import pytest

from audio2anki.anki import create_anki_deck
from audio2anki.models import AudioSegment


@pytest.fixture
def segments() -> list[AudioSegment]:
    """Return test segments."""
    return [
        AudioSegment(
            start=0.0,
            end=2.0,
            text="你好",
            translation="Hello",
            pronunciation="Nǐ hǎo",
            audio_file="audio_0001.mp3",
        ),
        AudioSegment(
            start=2.0,
            end=4.0,
            text="谢谢",
            translation="Thank you",
            pronunciation="Xièxie",
            audio_file="audio_0002.mp3",
        ),
    ]


def test_create_anki_deck(segments: list[AudioSegment], tmp_path: Path) -> None:
    """Test Anki deck creation."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    deck_file = create_anki_deck(segments, output_dir)
    assert deck_file.exists()

    # Read and verify content
    with open(deck_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)

    # Check header
    assert rows[0] == ["#Text", "Pronunciation", "Translation", "Audio"]

    # Check content
    assert rows[1] == ["你好", "Nǐ hǎo", "Hello", "[sound:audio_0001.mp3]"]
    assert rows[2] == ["谢谢", "Xièxie", "Thank you", "[sound:audio_0002.mp3]"]


def test_create_anki_deck_missing_fields(tmp_path: Path) -> None:
    """Test deck creation with missing optional fields."""
    segments = [
        AudioSegment(
            start=0.0,
            end=2.0,
            text="Test",
        )
    ]

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    deck_file = create_anki_deck(segments, output_dir)

    # Read and verify content
    with open(deck_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)

    # Check that missing fields are empty
    assert rows[1] == ["Test", "", "", ""]


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
    assert rows[0] == ["#Text", "Pronunciation", "Translation", "Audio"]

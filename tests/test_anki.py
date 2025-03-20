"""Tests for the Anki deck generation module."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from rich.progress import Progress, TaskID

from audio2anki.anki import create_anki_deck, generate_anki_deck
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


def test_create_anki_deck_with_progress(segments: list[AudioSegment], tmp_path: Path, mock_progress: Progress) -> None:
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


def test_generate_anki_deck(
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

    # Create test audio file
    from pydub import AudioSegment as PydubSegment

    audio = PydubSegment.silent(duration=4000)  # 4 seconds of silence
    audio_file = tmp_path / "test_audio.mp3"
    audio.export(str(audio_file), format="mp3")

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
        deck_dir = generate_anki_deck(
            translation_file,
            mock_pipeline_progress,
            transcription_file=transcript_file,
            pronunciation_file=pronunciation_file,
            source_language="chinese",
            target_language="english",
            input_audio_file=audio_file,
        )

        # Check output
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


def test_generate_anki_deck_with_output_folder(
    tmp_path: Path,
    mock_pipeline_progress: PipelineProgress,
) -> None:
    """Test deck generation with output folder specified."""
    # Use predefined paths instead of calculating them, since determination logic is now in main.py

    # Create test files
    transcript_file = tmp_path / "transcript.srt"
    with open(transcript_file, "w") as f:
        f.write("1\n00:00:00,000 --> 00:00:02,000\n你好\n\n")

    translation_file = tmp_path / "translation.srt"
    with open(translation_file, "w") as f:
        f.write("1\n00:00:00,000 --> 00:00:02,000\nHello\n\n")

    # Create test audio file
    from pydub import AudioSegment as PydubSegment

    audio = PydubSegment.silent(duration=2000)  # 2 seconds of silence
    audio_file = tmp_path / "chinese_lesson.mp3"
    audio.export(str(audio_file), format="mp3")

    # Change to tmp_path as working directory
    import os

    old_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        # Test with custom output folder
        custom_folder = "my_custom_folder"

        # Just use a direct path for the test
        tmp_path / custom_folder

        # Now test the full generation function
        deck_dir = generate_anki_deck(
            translation_file,
            mock_pipeline_progress,
            transcription_file=transcript_file,
            source_language="chinese",
            target_language="english",
            input_audio_file=audio_file,
            output_folder=custom_folder,
        )

        # Check that the deck was created with the correct name structure
        # Different OS paths might be handled differently, focus on the directory name
        assert deck_dir.name == custom_folder
        assert deck_dir.exists()
        assert (deck_dir / "deck.txt").exists()

        # Test with derived folder name
        # Since we're not calculating paths in anki.py anymore, we need to explicitly
        # set the output path to what would be derived normally
        derived_path = f"decks/{audio_file.stem}"

        derived_deck_dir = generate_anki_deck(
            translation_file,
            mock_pipeline_progress,
            transcription_file=transcript_file,
            source_language="chinese",
            target_language="english",
            input_audio_file=audio_file,
            output_folder=derived_path,
        )

        # Check that the directory structure is correct
        assert "decks" in str(derived_deck_dir)
        assert "chinese_lesson" in str(derived_deck_dir)
        assert derived_deck_dir.exists()
        assert (derived_deck_dir / "deck.txt").exists()

        # Test with existing non-deck folder - in real usage, main.py would detect this case
        # and create a nested folder. Since we're testing anki.py directly, we simulate
        # this by passing the correct nested path
        existing_folder = tmp_path / "existing_folder"
        existing_folder.mkdir()
        (existing_folder / "some_file.txt").write_text("test")

        # This is the path that would be calculated by determine_output_path in main.py
        nested_path = f"existing_folder/{audio_file.stem}"

        existing_folder_deck_dir = generate_anki_deck(
            translation_file,
            mock_pipeline_progress,
            transcription_file=transcript_file,
            source_language="chinese",
            target_language="english",
            input_audio_file=audio_file,
            output_folder=nested_path,
        )

        # Check that the directory structure is correct - we're using the nested path directly now
        assert "existing_folder" in str(existing_folder_deck_dir)
        assert audio_file.stem in str(existing_folder_deck_dir)
        assert existing_folder_deck_dir.exists()
        assert (existing_folder_deck_dir / "deck.txt").exists()

        # Test with existing deck folder (should replace)
        # In real usage, main.py would recognize this as an existing deck folder
        deck_folder_name = "deck_folder"
        deck_folder = tmp_path / deck_folder_name
        deck_folder.mkdir()
        (deck_folder / "deck.txt").write_text("old content")
        (deck_folder / "media").mkdir()

        replaced_deck_dir = generate_anki_deck(
            translation_file,
            mock_pipeline_progress,
            transcription_file=transcript_file,
            source_language="chinese",
            target_language="english",
            input_audio_file=audio_file,
            output_folder="deck_folder",
        )

        # Check that we're using the existing deck folder
        assert deck_folder.name in str(replaced_deck_dir)
        assert replaced_deck_dir.exists()

        # Verify content was replaced
        with open(replaced_deck_dir / "deck.txt") as f:
            content = f.read()
            assert "你好" in content  # New content
    finally:
        os.chdir(old_cwd)

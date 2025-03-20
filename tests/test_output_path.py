"""Tests for output path determination logic."""

import os
from pathlib import Path

import pytest

from audio2anki.main import determine_output_path


@pytest.fixture
def setup_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Set up test directories with various structures."""
    # Create an existing deck directory
    deck_dir = tmp_path / "existing_deck"
    deck_dir.mkdir()
    (deck_dir / "deck.txt").write_text("test deck content")
    (deck_dir / "media").mkdir()

    # Create a non-deck directory
    non_deck_dir = tmp_path / "non_deck_dir"
    non_deck_dir.mkdir()
    (non_deck_dir / "some_file.txt").write_text("This is not a deck")

    return deck_dir, non_deck_dir


def test_no_output_folder_use_default_input_file(tmp_path: Path) -> None:
    """Test when output_folder is not specified, using default input file."""
    # Using a mock input file
    input_file = Path("/path/to/default.mp3")

    result = determine_output_path(base_path=tmp_path, output_folder=None, input_file=input_file)

    # Should return path derived from input file
    expected_path = tmp_path / "decks/default"
    assert result == expected_path


def test_no_output_folder_with_input_file(tmp_path: Path) -> None:
    """Test when input_file is specified but output_folder is not."""
    input_file = Path("/path/to/lesson.mp3")

    result = determine_output_path(base_path=tmp_path, output_folder=None, input_file=input_file)

    # Should return decks/lesson path (but not create it)
    expected_path = tmp_path / "decks/lesson"
    assert result == expected_path

    # Directory should not be created
    assert not result.exists()


def test_input_file_as_string(tmp_path: Path) -> None:
    """Test with input_file as a string path."""
    # Test with string path
    input_file_str = "/path/to/another_lesson.mp3"

    result = determine_output_path(base_path=tmp_path, output_folder=None, input_file=Path(input_file_str))

    # Should return decks/another_lesson path (but not create it)
    expected_path = tmp_path / "decks/another_lesson"
    assert result == expected_path

    # Directory should not be created
    assert not result.exists()


def test_output_folder_nonexistent(tmp_path: Path) -> None:
    """Test when output_folder is specified but doesn't exist."""
    # Using a mock input file
    mock_input = Path("/path/to/test.mp3")
    result = determine_output_path(base_path=tmp_path, output_folder="new_folder", input_file=mock_input)

    # Should return the specified path (but not create it)
    expected_path = tmp_path / "new_folder"
    assert result == expected_path

    # Directory should not be created
    assert not result.exists()


def test_output_folder_is_deck_dir(setup_dirs: tuple[Path, Path], tmp_path: Path) -> None:
    """Test when output_folder is a deck directory."""
    deck_dir, _ = setup_dirs

    # Get the name of the deck_dir relative to tmp_path
    rel_path = deck_dir.relative_to(tmp_path)

    # Using a mock input file
    mock_input = Path("/path/to/test.mp3")
    result = determine_output_path(base_path=tmp_path, output_folder=str(rel_path), input_file=mock_input)

    # Should return the existing deck directory
    assert result == deck_dir


def test_output_folder_is_non_deck_dir(setup_dirs: tuple[Path, Path], tmp_path: Path) -> None:
    """Test when output_folder is not a deck directory."""
    _, non_deck_dir = setup_dirs

    # Get the name of the non_deck_dir relative to tmp_path
    rel_path = non_deck_dir.relative_to(tmp_path)

    # Test with input file
    input_file = Path("/path/to/lesson.mp3")
    result = determine_output_path(base_path=tmp_path, output_folder=str(rel_path), input_file=input_file)

    # Should return a nested path with name derived from input file (but not create it)
    expected_path = non_deck_dir / "lesson"
    assert result == expected_path

    # Directory should not be created
    assert not result.exists()

    # Test with a different input file
    # Using a different mock input file
    another_input = Path("/path/to/another.mp3")
    result_with_different_input = determine_output_path(
        base_path=tmp_path, output_folder=str(rel_path), input_file=another_input
    )

    # Should return a nested path with name derived from input file
    expected_path_another = non_deck_dir / "another"
    assert result_with_different_input == expected_path_another

    # Directory should not be created
    assert not result_with_different_input.exists()


def test_absolute_paths(tmp_path: Path) -> None:
    """Test with absolute paths for output_folder."""
    # Create a directory outside tmp_path
    other_dir = tmp_path.parent / "other_test_dir"
    os.makedirs(other_dir, exist_ok=True)
    try:
        # Use an absolute path as output_folder
        # Using a mock input file
        mock_input = Path("/path/to/test.mp3")
        result = determine_output_path(base_path=tmp_path, output_folder=str(other_dir), input_file=mock_input)

        # Should return a path in the specified directory with name derived from input file
        expected_path = other_dir / "test"
        assert result == expected_path

        # Create and test a deck dir
        deck_dir = tmp_path.parent / "deck_test_dir"
        os.makedirs(deck_dir, exist_ok=True)
        (deck_dir / "deck.txt").write_text("test")
        (deck_dir / "media").mkdir(exist_ok=True)

        # Using a mock input file
        mock_input = Path("/path/to/test.mp3")
        result_deck = determine_output_path(base_path=tmp_path, output_folder=str(deck_dir), input_file=mock_input)

        # For existing deck dirs, we should return the exact path
        assert result_deck == deck_dir
    finally:
        # Clean up dirs created outside tmp_path
        import shutil

        if other_dir.exists():
            shutil.rmtree(other_dir)
        if (tmp_path.parent / "deck_test_dir").exists():
            shutil.rmtree(tmp_path.parent / "deck_test_dir")


def test_deep_nested_paths(tmp_path: Path) -> None:
    """Test with deeply nested output paths."""
    # Test with nested output folder that doesn't exist
    # Using a mock input file
    mock_input = Path("/path/to/test.mp3")
    result = determine_output_path(base_path=tmp_path, output_folder="a/b/c/d", input_file=mock_input)

    expected_path = tmp_path / "a/b/c/d"
    assert result == expected_path

    # Directory should not be created
    assert not result.exists()

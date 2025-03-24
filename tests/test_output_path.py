"""Tests for output path determination logic."""

from pathlib import Path

import click
import pytest

from audio2anki.main import determine_output_path, is_deck_folder, is_empty_directory


@pytest.fixture
def setup_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Set up test directories with various structures."""
    # Create an existing deck directory
    deck_dir = tmp_path / "existing_deck"
    deck_dir.mkdir()
    (deck_dir / "deck.txt").write_text("test deck content")
    (deck_dir / "README.md").write_text("test readme content")
    (deck_dir / "test.csv").write_text("test csv content")
    media_dir = deck_dir / "media"
    media_dir.mkdir()

    # Create an empty directory
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    # Create a non-deck directory with some unrelated content
    non_deck_dir = tmp_path / "non_deck_dir"
    non_deck_dir.mkdir()
    (non_deck_dir / "some_file.txt").write_text("This is not a deck")

    return deck_dir, empty_dir, non_deck_dir


def test_is_empty_directory(tmp_path: Path) -> None:
    """Test is_empty_directory function."""
    # Create an empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert is_empty_directory(empty_dir)

    # Create a directory with a metadata file
    meta_dir = tmp_path / "with_meta"
    meta_dir.mkdir()
    (meta_dir / ".DS_Store").write_text("metadata")
    assert is_empty_directory(meta_dir)

    # Create a directory with a real file
    non_empty_dir = tmp_path / "non_empty"
    non_empty_dir.mkdir()
    (non_empty_dir / "real_file.txt").write_text("content")
    assert not is_empty_directory(non_empty_dir)


def test_is_deck_folder(tmp_path: Path) -> None:
    """Test is_deck_folder function."""
    # Create a valid deck folder
    deck_dir = tmp_path / "deck"
    deck_dir.mkdir()
    (deck_dir / "deck.txt").write_text("deck content")
    (deck_dir / "README.md").write_text("readme content")
    (deck_dir / "cards.csv").write_text("csv content")
    (deck_dir / "media").mkdir()
    assert is_deck_folder(deck_dir)

    # Create an invalid deck folder (missing CSV)
    invalid_deck = tmp_path / "invalid_deck"
    invalid_deck.mkdir()
    (invalid_deck / "deck.txt").write_text("content")
    (invalid_deck / "README.md").write_text("content")
    (invalid_deck / "media").mkdir()
    assert not is_deck_folder(invalid_deck)

    # Create a non-deck folder
    non_deck = tmp_path / "non_deck"
    non_deck.mkdir()
    (non_deck / "some_file.txt").write_text("content")
    assert not is_deck_folder(non_deck)


def test_no_output_folder_specified(tmp_path: Path) -> None:
    """Test when output_folder is not specified."""
    # Using a mock input file
    input_file = Path("/path/to/lesson1.mp3")

    result = determine_output_path(base_path=tmp_path, output_folder=None, input_file=input_file)

    # Should use decks/lesson1 path
    expected_path = tmp_path / "decks/lesson1"
    assert result == expected_path


def test_no_output_folder_target_exists_and_valid(tmp_path: Path) -> None:
    """Test when target path exists and is a valid deck folder or empty directory."""
    # Create a deck directory
    deck_dir = tmp_path / "decks/lesson1"
    deck_dir.mkdir(parents=True)
    (deck_dir / "deck.txt").write_text("test deck content")
    (deck_dir / "README.md").write_text("readme content")
    (deck_dir / "cards.csv").write_text("csv content")
    (deck_dir / "media").mkdir()

    input_file = Path("/path/to/lesson1.mp3")
    result = determine_output_path(base_path=tmp_path, output_folder=None, input_file=input_file)
    assert result == deck_dir


def test_no_output_folder_target_exists_not_valid(tmp_path: Path) -> None:
    """Test when target path exists but is not empty nor a deck folder."""
    # Create a non-deck directory
    non_deck_dir = tmp_path / "decks/lesson1"
    non_deck_dir.mkdir(parents=True)
    (non_deck_dir / "some_file.txt").write_text("This is not a deck")

    input_file = Path("/path/to/lesson1.mp3")
    with pytest.raises(click.ClickException) as excinfo:
        determine_output_path(base_path=tmp_path, output_folder=None, input_file=input_file)

    assert "neither empty nor a deck folder" in str(excinfo.value)


def test_output_folder_absolute_path(tmp_path: Path) -> None:
    """Test with absolute path for output_folder."""
    # Using a mock input file
    input_file = Path("/path/to/lesson1.mp3")
    # Create an absolute path outside tmp_path
    abs_path = tmp_path.parent / "absolute_dir"

    result = determine_output_path(base_path=tmp_path, output_folder=str(abs_path), input_file=input_file)
    assert result == abs_path


def test_output_folder_relative_path(tmp_path: Path) -> None:
    """Test with relative path for output_folder."""
    # Using a mock input file
    input_file = Path("/path/to/lesson1.mp3")
    rel_path = "relative_dir"

    result = determine_output_path(base_path=tmp_path, output_folder=rel_path, input_file=input_file)
    assert result == tmp_path / rel_path


def test_output_folder_exists_and_empty(setup_dirs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """Test when output_folder exists and is empty."""
    _, empty_dir, _ = setup_dirs
    rel_path = empty_dir.relative_to(tmp_path)

    input_file = Path("/path/to/lesson1.mp3")
    result = determine_output_path(base_path=tmp_path, output_folder=str(rel_path), input_file=input_file)

    # Should use the empty directory directly
    assert result == empty_dir


def test_output_folder_exists_and_is_deck(setup_dirs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """Test when output_folder exists and is a deck folder."""
    deck_dir, _, _ = setup_dirs
    rel_path = deck_dir.relative_to(tmp_path)

    input_file = Path("/path/to/lesson1.mp3")
    result = determine_output_path(base_path=tmp_path, output_folder=str(rel_path), input_file=input_file)

    # Should use the deck directory directly
    assert result == deck_dir


def test_output_folder_exists_not_empty_not_deck(setup_dirs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """Test when output_folder exists but is neither empty nor a deck folder."""
    _, _, non_deck_dir = setup_dirs
    rel_path = non_deck_dir.relative_to(tmp_path)

    input_file = Path("/path/to/lesson1.mp3")
    result = determine_output_path(base_path=tmp_path, output_folder=str(rel_path), input_file=input_file)

    # Should create a nested path with the input filename
    expected_path = non_deck_dir / "lesson1"
    assert result == expected_path


def test_nested_output_folder_exists_but_invalid(setup_dirs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """Test when nested output path exists but is invalid."""
    _, _, non_deck_dir = setup_dirs
    rel_path = non_deck_dir.relative_to(tmp_path)

    # Create an invalid directory at the nested location
    invalid_nested = non_deck_dir / "lesson1"
    invalid_nested.mkdir()
    (invalid_nested / "some_file.txt").write_text("content")

    input_file = Path("/path/to/lesson1.mp3")
    with pytest.raises(click.ClickException) as excinfo:
        determine_output_path(base_path=tmp_path, output_folder=str(rel_path), input_file=input_file)

    assert "neither empty nor a deck folder" in str(excinfo.value)


def test_nested_output_folder_exists_and_valid(setup_dirs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """Test when nested output path exists and is valid."""
    _, _, non_deck_dir = setup_dirs
    rel_path = non_deck_dir.relative_to(tmp_path)

    # Create a valid deck directory at the nested location
    valid_nested = non_deck_dir / "lesson1"
    valid_nested.mkdir()
    (valid_nested / "deck.txt").write_text("deck content")
    (valid_nested / "README.md").write_text("readme content")
    (valid_nested / "cards.csv").write_text("csv content")
    (valid_nested / "media").mkdir()

    input_file = Path("/path/to/lesson1.mp3")
    result = determine_output_path(base_path=tmp_path, output_folder=str(rel_path), input_file=input_file)

    assert result == valid_nested


def test_deep_nested_paths(tmp_path: Path) -> None:
    """Test with deeply nested output paths."""
    # Test with nested output folder that doesn't exist
    input_file = Path("/path/to/lesson1.mp3")
    result = determine_output_path(base_path=tmp_path, output_folder="a/b/c/d", input_file=input_file)

    expected_path = tmp_path / "a/b/c/d"
    assert result == expected_path

    # Directory should not be created
    assert not result.exists()

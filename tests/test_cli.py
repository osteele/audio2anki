"""Tests for the CLI module."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from audio2anki.cli import main, process_audio
from audio2anki.models import AudioSegment


@pytest.fixture
def runner() -> CliRunner:
    """Click test runner."""
    return CliRunner()


@pytest.fixture
def mock_segments() -> list[AudioSegment]:
    """Sample processed segments."""
    return [
        AudioSegment(
            start=0.0,
            end=2.0,
            text="Test",
            translation="Test",
            pronunciation="Test",
            audio_file="audio_0001.mp3",
        )
    ]


@patch("audio2anki.cli.transcribe_audio")
@patch("audio2anki.cli.translate_segments")
@patch("audio2anki.cli.split_audio")
@patch("audio2anki.cli.create_anki_deck")
def test_process_audio(
    mock_create_deck: Mock,
    mock_split: Mock,
    mock_translate: Mock,
    mock_transcribe: Mock,
    mock_segments: list[AudioSegment],
    tmp_path: Path,
) -> None:
    """Test audio processing pipeline."""
    # Set up mocks
    mock_transcribe.return_value = mock_segments
    mock_translate.return_value = mock_segments
    mock_split.return_value = mock_segments
    mock_create_deck.return_value = tmp_path / "deck.txt"

    # Create test files
    input_file = tmp_path / "test.mp3"
    input_file.touch()
    output_dir = tmp_path / "output"

    # Process audio
    process_audio(
        input_file,
        None,
        output_dir,
        "small",
        None,
        1.0,
        15.0,
        -40,
        debug=True,
        progress=None,
    )

    # Verify pipeline
    mock_transcribe.assert_called_once()
    mock_translate.assert_called_once()
    mock_split.assert_called_once()

    # Check debug file creation
    debug_file = output_dir / "debug.txt"
    assert debug_file.exists()


def test_cli_basic(runner: CliRunner, tmp_path: Path) -> None:
    """Test basic CLI functionality."""
    input_file = tmp_path / "test.mp3"
    input_file.touch()

    with patch("audio2anki.cli.process_audio") as mock_process:
        mock_process.return_value = []
        with patch("audio2anki.cli.create_anki_deck") as mock_create_deck:
            mock_create_deck.return_value = tmp_path / "deck.txt"

            result = runner.invoke(main, [str(input_file)])
            assert result.exit_code == 0


def test_cli_with_options(runner: CliRunner, tmp_path: Path) -> None:
    """Test CLI with various options."""
    input_file = tmp_path / "test.mp3"
    input_file.touch()
    transcript = tmp_path / "transcript.txt"
    transcript.touch()

    with patch("audio2anki.cli.process_audio") as mock_process:
        mock_process.return_value = []
        with patch("audio2anki.cli.create_anki_deck") as mock_create_deck:
            mock_create_deck.return_value = tmp_path / "deck.txt"

            result = runner.invoke(
                main,
                [
                    str(input_file),
                    "--transcript",
                    str(transcript),
                    "--model",
                    "large",
                    "--language",
                    "zh",
                    "--debug",
                ],
            )
            assert result.exit_code == 0

            # Verify process_audio was called with correct arguments
            call_args = mock_process.call_args[0]
            assert call_args[0] == input_file
            assert call_args[1] == transcript
            assert call_args[3] == "large"  # model
            assert call_args[4] == "zh"  # language
            assert call_args[8]  # debug


def test_cli_error_handling(runner: CliRunner, tmp_path: Path) -> None:
    """Test CLI error handling."""
    input_file = tmp_path / "test.mp3"
    input_file.touch()

    with patch("audio2anki.cli.process_audio") as mock_process:
        mock_process.side_effect = Exception("Test error")

        result = runner.invoke(main, [str(input_file)])
        assert result.exit_code != 0
        assert "Error: Test error" in result.output


def test_cli_missing_input(runner: CliRunner) -> None:
    """Test CLI with missing input file."""
    result = runner.invoke(main, ["nonexistent.mp3"])
    assert result.exit_code != 0
    assert "Error" in result.output

"""Tests for translation module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from rich.progress import Progress

from audio2anki.transcribe import TranscriptionSegment, load_transcript, save_transcript
from audio2anki.translate import TranslationItem, TranslationProvider, TranslationResponse, translate_segments
from audio2anki.types import LanguageCode


@pytest.mark.parametrize(
    "input_text,expected_translation",
    [
        ("你好", "Hello"),
        ("谢谢", "Thank you"),
    ],
)
def test_translate_with_openai(input_text: str, expected_translation: str) -> None:
    """Test basic translation with OpenAI."""
    segment = TranscriptionSegment(start=0.0, end=1.0, text=input_text, translation=None)

    # Create temporary files for input and output
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "input.json"
        output_file = Path(temp_dir) / "output.json"

        # Save segment to input file
        save_transcript([segment], input_file)

        # Create a mock response
        mock_response = TranslationResponse(
            items=[
                TranslationItem(
                    start_time=0.0,
                    end_time=1.0,
                    text=input_text,
                    translation=expected_translation,
                    pronunciation=None,
                )
            ]
        )

        # Patch the translate_with_openai function to return our mock response
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("audio2anki.translate.translate_with_openai", return_value=mock_response),
        ):
            with Progress() as progress:
                task_id = progress.add_task("Translating", total=1)

                # Translate segment
                translate_segments(input_file, output_file, LanguageCode("en"), task_id, progress)

                # Load the result and verify
                result = load_transcript(output_file)
                assert len(result) == 1
                assert result[0].translation == expected_translation


def test_translate_with_deepl() -> None:
    """Test translation using DeepL."""
    segment = TranscriptionSegment(start=0.0, end=1.0, text="Bonjour", translation=None)

    # Create temporary files for input and output
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "input.json"
        output_file = Path(temp_dir) / "output.json"

        # Save segment to input file
        save_transcript([segment], input_file)

        # Set up mock DeepL response
        mock_deepl_response = Mock()
        mock_deepl_response.text = "Hello"

        with (
            patch.dict(os.environ, {"DEEPL_API_TOKEN": "test-key", "OPENAI_API_KEY": "test-key"}),
            patch("deepl.Translator") as mock_deepl,
            # Also patch translate_with_openai in case of fallback
            patch("audio2anki.translate.translate_with_openai", return_value=TranslationResponse(items=[])),
        ):
            # Setup mock translator
            mock_translator = Mock()
            mock_translator.translate_text.return_value = mock_deepl_response
            mock_deepl.return_value = mock_translator

            with Progress() as progress:
                task_id = progress.add_task("Translating", total=1)

                # Translate segment using DeepL
                translate_segments(
                    input_file,
                    output_file,
                    LanguageCode("en"),
                    task_id,
                    progress,
                    translation_provider=TranslationProvider.DEEPL,
                )

                # Load the result and verify
                result = load_transcript(output_file)
                assert len(result) == 1
                assert result[0].translation is not None

                # Verify DeepL was used
                assert mock_translator.translate_text.call_count >= 1


def test_fallback_to_openai_when_deepl_fails() -> None:
    """Test fallback to OpenAI when DeepL initialization fails."""
    segment = TranscriptionSegment(start=0.0, end=1.0, text="Hola", translation=None)

    # Create temporary files for input and output
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "input.json"
        output_file = Path(temp_dir) / "output.json"

        # Save segment to input file
        save_transcript([segment], input_file)

        # Create a mock OpenAI response for the fallback
        mock_response = TranslationResponse(
            items=[
                TranslationItem(
                    start_time=0.0,
                    end_time=1.0,
                    text="Hola",
                    translation="Hello",
                    pronunciation=None,
                )
            ]
        )

        with (
            patch.dict(os.environ, {"DEEPL_API_TOKEN": "test-key", "OPENAI_API_KEY": "test-key"}),
            patch("deepl.Translator") as mock_deepl,
            patch("audio2anki.translate.translate_with_openai", return_value=mock_response),
        ):
            # Make DeepL fail to trigger fallback
            mock_deepl.side_effect = Exception("DeepL error")

            with Progress() as progress:
                task_id = progress.add_task("Translating", total=1)

                # Translate segment - should succeed using mocked OpenAI
                translate_segments(
                    input_file,
                    output_file,
                    LanguageCode("en"),
                    task_id,
                    progress,
                    translation_provider=TranslationProvider.DEEPL,  # Should fall back to OpenAI
                )

                # Load the result and verify
                result = load_transcript(output_file)
                assert len(result) == 1
                assert result[0].translation == "Hello"


def test_no_api_keys_raises_error() -> None:
    """Test that missing API keys raise appropriate errors."""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "input.json"
        output_file = Path(temp_dir) / "output.json"

        # Create an empty input file
        save_transcript([], input_file)

        with patch.dict(os.environ, {}, clear=True), Progress() as progress:
            task_id = progress.add_task("test", total=1)
            with pytest.raises(ValueError) as exc:
                translate_segments(input_file, output_file, LanguageCode("en"), task_id, progress)
            assert "OPENAI_API_KEY environment variable is required" in str(exc.value)

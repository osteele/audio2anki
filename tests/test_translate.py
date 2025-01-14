"""Tests for translation module."""

import os
from unittest.mock import Mock, patch, MagicMock

import pytest
import httpx
from openai import OpenAI
from rich.progress import Progress, TaskID

from audio2anki.models import AudioSegment
from audio2anki.translate import translate_segments


@pytest.fixture
def segments() -> list[AudioSegment]:
    """Return test segments."""
    return [
        AudioSegment(start=0.0, end=2.0, text="你好"),
        AudioSegment(start=2.0, end=4.0, text="谢谢"),
    ]


@pytest.fixture
def progress() -> Progress:
    """Return progress bar."""
    return Progress()


@pytest.fixture
def task_id(progress: Progress) -> TaskID:
    """Return task ID."""
    return progress.add_task("Translating...", total=2)


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
def test_translate_segments(
    segments: list[AudioSegment],
    progress: Progress,
    task_id: TaskID,
) -> None:
    """Test translation of segments."""
    # Set up mock response
    mock_message = Mock()
    mock_message.content = "Hello"
    mock_choice = Mock()
    mock_choice.message = mock_message
    mock_response = Mock()
    mock_response.choices = [mock_choice]

    with patch.object(OpenAI, "__init__", return_value=None) as mock_init:
        with patch.object(OpenAI, "chat") as mock_chat:
            mock_chat.completions.create.return_value = mock_response

            # Translate segments
            translate_segments(segments, "english", task_id, progress)

            # Check translations
            assert segments[0].translation == "Hello"
            assert segments[1].translation == "Hello"

            # Verify API was called correctly
            assert mock_chat.completions.create.call_count == 2
            for call in mock_chat.completions.create.call_args_list:
                args = call[1]
                assert args["model"] == "gpt-3.5-turbo"
                assert len(args["messages"]) == 2
                assert args["messages"][0]["role"] == "system"
                assert args["messages"][1]["role"] == "user"


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
def test_translate_error_handling(
    segments: list[AudioSegment],
    progress: Progress,
    task_id: TaskID,
) -> None:
    """Test error handling in translation."""
    # Set up mock error
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 401
    mock_response.text = "API error"
    mock_response.request = Mock(spec=httpx.Request)
    mock_response.request.method = "POST"
    mock_response.request.url = "https://api.openai.com/v1/chat/completions"

    with patch.object(OpenAI, "__init__", return_value=None) as mock_init:
        with patch.object(OpenAI, "chat") as mock_chat:
            mock_chat.completions.create.side_effect = httpx.HTTPStatusError(
                "401 Unauthorized",
                request=mock_response.request,
                response=mock_response,
            )

            # Test error handling
            with pytest.raises(RuntimeError, match="Translation failed: 401 Unauthorized"):
                translate_segments(segments, "english", task_id, progress)


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
def test_translate_empty_response(
    segments: list[AudioSegment],
    progress: Progress,
    task_id: TaskID,
) -> None:
    """Test handling of empty response."""
    # Set up mock with empty response
    mock_message = Mock()
    mock_message.content = ""
    mock_choice = Mock()
    mock_choice.message = mock_message
    mock_response = Mock()
    mock_response.choices = [mock_choice]

    with patch.object(OpenAI, "__init__", return_value=None) as mock_init:
        with patch.object(OpenAI, "chat") as mock_chat:
            mock_chat.completions.create.return_value = mock_response

            # Test error handling
            with pytest.raises(ValueError, match="Empty response from OpenAI"):
                translate_segments(segments, "english", task_id, progress)


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    class MockChoice:
        def __init__(self, text: str):
            self.message = MagicMock()
            self.message.content = text

    class MockResponse:
        def __init__(self, text: str):
            self.choices = [MockChoice(text)]

    return MockResponse("Translated text")


@pytest.fixture
def mock_deepl_response():
    """Mock DeepL API response."""
    class MockResult:
        def __init__(self, text: str):
            self.text = text

    return MockResult("Translated text")


@pytest.fixture
def segments_deepl():
    """Sample audio segments."""
    return [
        AudioSegment(start=0, end=1, text="Hello"),
        AudioSegment(start=1, end=2, text="World"),
    ]


def test_translate_segments_with_openai(segments_deepl, mock_openai_response):
    """Test translation using OpenAI."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai.return_value = mock_client

            with Progress() as progress:
                task_id = progress.add_task("test", total=len(segments_deepl))
                translated = translate_segments(segments_deepl, "english", task_id, progress)

            assert len(translated) == 2
            assert all(s.translation == "Translated text" for s in translated)
            assert mock_client.chat.completions.create.call_count == 2


def test_translate_segments_with_deepl(segments_deepl, mock_deepl_response):
    """Test translation using DeepL."""
    with patch.dict(os.environ, {"DEEPL_API_TOKEN": "test-key"}):
        with patch("deepl.Translator") as mock_deepl:
            mock_translator = MagicMock()
            mock_translator.translate_text.return_value = mock_deepl_response
            mock_deepl.return_value = mock_translator

            with Progress() as progress:
                task_id = progress.add_task("test", total=len(segments_deepl))
                translated = translate_segments(segments_deepl, "english", task_id, progress)

            assert len(translated) == 2
            assert all(s.translation == "Translated text" for s in translated)
            assert mock_translator.translate_text.call_count == 2


def test_translate_segments_fallback_to_openai(segments_deepl, mock_openai_response):
    """Test fallback to OpenAI when DeepL fails."""
    with patch.dict(os.environ, {
        "DEEPL_API_TOKEN": "test-key",
        "OPENAI_API_KEY": "test-key"
    }):
        with patch("deepl.Translator") as mock_deepl:
            mock_deepl.side_effect = Exception("DeepL error")
            
            with patch("openai.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.chat.completions.create.return_value = mock_openai_response
                mock_openai.return_value = mock_client

                with Progress() as progress:
                    task_id = progress.add_task("test", total=len(segments_deepl))
                    translated = translate_segments(segments_deepl, "english", task_id, progress)

                assert len(translated) == 2
                assert all(s.translation == "Translated text" for s in translated)
                assert mock_client.chat.completions.create.call_count == 2


def test_translate_segments_no_api_keys():
    """Test error when no API keys are available."""
    with patch.dict(os.environ, {}, clear=True):
        with Progress() as progress:
            task_id = progress.add_task("test", total=1)
            with pytest.raises(ValueError) as exc:
                translate_segments([], "english", task_id, progress)
            assert "Neither DEEPL_API_TOKEN nor OPENAI_API_KEY" in str(exc.value)

"""Tests for translation module."""

import os
from unittest.mock import Mock, patch

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

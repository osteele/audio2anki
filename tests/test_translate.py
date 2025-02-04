"""Tests for the translate module."""

import os
from unittest.mock import MagicMock, Mock, patch

import openai
import pytest
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
def segments_deepl() -> list[AudioSegment]:
    """Return test segments for DeepL tests."""
    return [
        AudioSegment(start=0, end=1, text="Hello"),
        AudioSegment(start=1, end=2, text="World"),
    ]


class MockChoice:
    """Mock OpenAI chat completion choice."""

    def __init__(self, content: str):
        self.message = MagicMock()
        self.message.content = content


@pytest.fixture
def task_id(progress: Progress) -> TaskID:
    """Create a task ID for testing."""
    return progress.add_task("test", total=100)


@patch("openai.OpenAI")
@patch.dict(os.environ, {"OPENAI_API_KEY": "valid-key"})
def test_translate_segments(
    mock_openai_class: Mock,
    segments: list[AudioSegment],
    progress: Progress,
    task_id: TaskID,
) -> None:
    """Test translation with OpenAI."""
    # Set up mock
    mock_client = mock_openai_class.return_value
    mock_chat = MagicMock()
    mock_chat.completions = MagicMock()

    # Mock responses for translation and pinyin for first segment
    mock_response1_trans = MagicMock()
    mock_response1_trans.choices = [MagicMock(message=MagicMock(content="Hello"))]
    mock_response1_pinyin = MagicMock()
    mock_response1_pinyin.choices = [MagicMock(message=MagicMock(content="ni hao"))]

    # Mock responses for translation and pinyin for second segment
    mock_response2_trans = MagicMock()
    mock_response2_trans.choices = [MagicMock(message=MagicMock(content="Thank you"))]
    mock_response2_pinyin = MagicMock()
    mock_response2_pinyin.choices = [MagicMock(message=MagicMock(content="xie xie"))]

    # Set up the sequence of responses
    mock_chat.completions.create.side_effect = [
        mock_response1_trans,  # First segment translation
        mock_response1_pinyin,  # First segment pinyin
        mock_response2_trans,  # Second segment translation
        mock_response2_pinyin,  # Second segment pinyin
    ]
    mock_client.chat = mock_chat

    # Mock OpenAI client creation
    with patch("audio2anki.translate.OpenAI", return_value=mock_client):
        translated = translate_segments(segments, task_id, progress, "english")
        assert len(translated) == len(segments)
        assert translated[0].translation == "Hello"
        assert translated[0].pronunciation == "ni hao"
        assert translated[1].translation == "Thank you"
        assert translated[1].pronunciation == "xie xie"


@patch("openai.OpenAI")
@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
def test_translate_error_handling(
    mock_openai_class: Mock,
    segments: list[AudioSegment],
    progress: Progress,
    task_id: TaskID,
) -> None:
    """Test error handling during translation."""
    # Set up mock
    mock_client = mock_openai_class.return_value
    mock_chat = MagicMock()
    mock_chat.completions = MagicMock()
    mock_chat.completions.create.side_effect = openai.AuthenticationError(
        "Incorrect API key provided",
        response=Mock(status_code=401),
        body={"error": {"message": "Incorrect API key provided"}},
    )
    mock_client.chat = mock_chat

    # Mock OpenAI client creation
    with patch("audio2anki.translate.OpenAI", return_value=mock_client):
        with pytest.raises(RuntimeError, match="Error translating segment: Incorrect API key provided"):
            translate_segments(segments, task_id, progress, "english")


@patch("openai.OpenAI")
@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
def test_translate_empty_response(
    mock_openai_class: Mock,
    segments: list[AudioSegment],
    progress: Progress,
    task_id: TaskID,
) -> None:
    """Test handling of empty translation response."""
    # Set up mock
    mock_client = mock_openai_class.return_value
    mock_chat = MagicMock()
    mock_chat.completions = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=""))]  # Empty response
    mock_chat.completions.create.return_value = mock_response
    mock_client.chat = mock_chat

    # Mock OpenAI client creation
    with patch("audio2anki.translate.OpenAI", return_value=mock_client):
        with pytest.raises(RuntimeError, match="Error translating segment: Empty response from OpenAI"):
            translate_segments(segments, task_id, progress, "english")


@patch("openai.OpenAI")
@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
def test_translate_segments_with_openai(
    mock_openai_class: Mock,
    segments_deepl: list[AudioSegment],
    task_id: TaskID,
    progress: Progress,
) -> None:
    """Test translation with OpenAI."""
    # Set up mock
    mock_client = mock_openai_class.return_value
    mock_chat = MagicMock()
    mock_chat.completions = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Translated text"))]
    mock_chat.completions.create.return_value = mock_response
    mock_client.chat = mock_chat

    # Mock OpenAI client creation
    with patch("audio2anki.translate.OpenAI", return_value=mock_client):
        translated = translate_segments(segments_deepl, task_id, progress, "english")

        # Check results
        assert len(translated) == len(segments_deepl)
        assert all(s.translation == "Translated text" for s in translated)
        mock_client.chat.completions.create.assert_called()


@patch("openai.OpenAI")
@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
def test_translate_segments_fallback_to_openai(
    mock_openai_class: Mock,
    segments_deepl: list[AudioSegment],
    task_id: TaskID,
    progress: Progress,
) -> None:
    """Test fallback to OpenAI when DeepL fails."""
    with patch.dict(os.environ, {
        "DEEPL_API_TOKEN": "test-key",
        "OPENAI_API_KEY": "test-key"
    }):
        with patch("deepl.Translator") as mock_deepl:
            mock_deepl.side_effect = Exception("DeepL error")

            # Set up OpenAI mock
            mock_client = mock_openai_class.return_value
            mock_chat = MagicMock()
            mock_chat.completions = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Translated text"))]
            mock_chat.completions.create.return_value = mock_response
            mock_client.chat = mock_chat

            # Mock OpenAI client creation
            with patch("audio2anki.translate.OpenAI", return_value=mock_client):
                translated = translate_segments(segments_deepl, task_id, progress, "english")

                # Check results
                assert len(translated) == len(segments_deepl)
                assert all(s.translation == "Translated text" for s in translated)
                mock_client.chat.completions.create.assert_called()


@patch("openai.OpenAI")
@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
def test_translate_error_handling_openai(
    mock_openai_class: Mock,
    segments_deepl: list[AudioSegment],
    task_id: TaskID,
    progress: Progress,
) -> None:
    """Test error handling during translation."""
    # Set up mock
    mock_client = mock_openai_class.return_value
    mock_chat = MagicMock()
    mock_chat.completions = MagicMock()
    mock_chat.completions.create.side_effect = Exception("Translation error")
    mock_client.chat = mock_chat

    # Mock OpenAI client creation
    with patch("audio2anki.translate.OpenAI", return_value=mock_client):
        with pytest.raises(RuntimeError, match="Error translating segment: Translation error"):
            translate_segments(segments_deepl, task_id, progress, "english")


@patch("openai.OpenAI")
@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
def test_translate_empty_response_openai(
    mock_openai_class: Mock,
    segments_deepl: list[AudioSegment],
    task_id: TaskID,
    progress: Progress,
) -> None:
    """Test handling of empty translation response."""
    # Set up mock
    mock_client = mock_openai_class.return_value
    mock_chat = MagicMock()
    mock_chat.completions = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=None))]  # Empty response
    mock_chat.completions.create.return_value = mock_response
    mock_client.chat = mock_chat

    # Mock OpenAI client creation
    with patch("audio2anki.translate.OpenAI", return_value=mock_client):
        with pytest.raises(RuntimeError, match="Error translating segment: Empty response from OpenAI"):
            translate_segments(segments_deepl, task_id, progress, "english")

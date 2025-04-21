from unittest.mock import patch

from audio2anki.select_sentences import filter_segments, is_one_word
from audio2anki.transcribe import TranscriptionSegment
from audio2anki.types import LanguageCode


# Mock class for DetectionResult
class MockDetectionResult:
    def __init__(self, language: str, confidence: float = 0.9, is_ambiguous: bool = False):
        self.language = language
        self.confidence = confidence
        self.is_ambiguous = is_ambiguous

    def __bool__(self):
        return bool(self.language)


class DummySeg(TranscriptionSegment):
    def __init__(self, text: str):
        super().__init__(start=0.0, end=1.0, text=text)


def make_segments(texts: list[str]) -> list[DummySeg]:
    return [DummySeg(t) for t in texts]


def test_reject_one_word():
    segs = make_segments(["Hello", "Hello world", "Test"])
    with patch("audio2anki.select_sentences.detect_language", return_value=MockDetectionResult("en")):
        result = filter_segments(segs, source_language=LanguageCode("en"))
    assert all(" " in s.text for s in result)


def test_reject_ending_comma():
    segs = make_segments(["Hello,", "Hello world，", "Hello world.", "Hello world"])
    with patch("audio2anki.select_sentences.detect_language", return_value=MockDetectionResult("en")):
        result = filter_segments(segs, source_language=LanguageCode("en"))
    assert all(not s.text.endswith(",") and not s.text.endswith("，") for s in result)


def test_remove_duplicates():
    segs = make_segments(["Hello world", "Hello world", "Hi there", "Hi there"])
    with patch("audio2anki.select_sentences.detect_language", return_value=MockDetectionResult("en")):
        result = filter_segments(segs, source_language=LanguageCode("en"))
    texts = [s.text for s in result]
    assert texts == ["Hello world", "Hi there"]


def test_language_filter():
    segs = make_segments(["Hello world", "Bonjour le monde"])

    def fake_detect(text: str):
        return MockDetectionResult("en" if "Hello" in text else "fr")

    with patch("audio2anki.select_sentences.detect_language", side_effect=fake_detect):
        result = filter_segments(segs, source_language=LanguageCode("en"))
    assert any("Hello" in s.text for s in result)
    assert all("Bonjour" not in s.text for s in result)


def test_majority_language_detection():
    segs = make_segments(["Hello world", "How are you?", "Bonjour le monde"])

    def fake_detect(text: str):
        return MockDetectionResult("en" if "Hello" in text or "How" in text else "fr")

    with patch("audio2anki.select_sentences.detect_language", side_effect=fake_detect):
        result = filter_segments(segs, source_language=None)
    texts = [s.text for s in result]
    assert "Bonjour le monde" not in texts
    assert "Hello world" in texts and "How are you?" in texts


def test_is_one_word():
    assert is_one_word("Hello")
    assert not is_one_word("Hello world")
    assert not is_one_word("")


def test_is_one_word_punctuation():
    assert is_one_word("Hello.")
    assert is_one_word("Hello?")
    assert is_one_word("Hello!")


def test_is_one_word_cjk():
    # Single characters should be detected as one word
    assert is_one_word("中")  # Single Hanzi
    assert is_one_word("日")  # Single Kanji
    assert is_one_word("あ")  # Single Hiragana

    # Two characters should be detected as one word
    assert is_one_word("中国")  # Two Hanzi
    assert is_one_word("日本")  # Two Kanji
    assert is_one_word("あい")  # Two Hiragana

    # Three or more characters should not be detected as one word
    assert not is_one_word("中国人")  # Three Hanzi
    assert not is_one_word("日本語")  # Three Kanji
    assert not is_one_word("あいう")  # Three Hiragana

    # Mixed CJK sentences should not be detected as one word
    assert not is_one_word("我喜欢学习")  # Hanzi sentence
    assert not is_one_word("私は日本人です")  # Kanji + Hiragana
    assert not is_one_word("これは本です")  # Hiragana + Kanji

    # Spaces in CJK should not be detected as one word
    assert not is_one_word("你 好")  # Hanzi with space
    assert not is_one_word("こん にちは")  # Hiragana with space


def test_is_one_word_specific_chinese_phrases():
    # These specific phrases should not be detected as one word
    assert not is_one_word("说法吗?")  # Chinese phrase with question mark
    assert not is_one_word("发生了一些事情。")  # Chinese sentence with period

    # These should also not be detected as one word
    assert not is_one_word("你可以重复语言")
    assert not is_one_word("这并不是")

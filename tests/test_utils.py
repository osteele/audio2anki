"""Tests for utility functions."""

from audio2anki.utils import sanitize_filename


def test_sanitize_filename() -> None:
    """Test filename sanitization."""
    # Test that diacritics are preserved in Vietnamese
    assert sanitize_filename("Bài học tiếng Việt số 1.mp3") == "Bài_học_tiếng_Việt_số_1.mp3"

    # Test Korean characters
    assert sanitize_filename("한국어 강의 레슨5.mp3") == "한국어_강의_레슨5.mp3"

    # Test Chinese characters
    assert sanitize_filename("中文课程　、第三课.mp3") == "中文课程_第三课.mp3"

    # Test Japanese characters
    assert sanitize_filename("日本語講座_レッスン2.mp3") == "日本語講座_レッスン2.mp3"

    # Test unsafe characters are replaced
    assert sanitize_filename("file/with:unsafe*chars?") == "with_unsafe_chars"

    # Test multiple spaces and underscores are collapsed
    assert sanitize_filename("multiple   spaces___here") == "multiple_spaces_here"

    # Test leading and trailing underscores are removed
    assert sanitize_filename("_leading_and_trailing_") == "leading_and_trailing"

    # Test length limiting preserves both ends
    long_name = "this_is_a_very_long_filename_that_exceeds_the_limit_by_far.mp3"
    sanitized = sanitize_filename(long_name)
    assert len(sanitized) <= 32
    assert "..." in sanitized
    assert sanitized.endswith(".mp3")
    assert sanitized.startswith("this_is")

    # Test empty or all-special-chars filename
    assert sanitize_filename("") == "unnamed"
    assert sanitize_filename("???///:::") == "unnamed"

"""Shared type definitions for audio2anki."""


class LanguageCode(str):
    """A two-or three-letter ISO 639-1 language code."""

    def __new__(cls, code: str) -> "LanguageCode":
        """Create a new LanguageCode instance with validation."""
        if not (2 <= len(code) <= 3) or not code.isalpha() or not code.islower():
            raise ValueError(f"Invalid language code: {code}. Must be a two-letter ISO 639-1 code.")
        return super().__new__(cls, code)

"""Data models for audio2anki."""

from dataclasses import dataclass


@dataclass
class AudioSegment:
    """A segment of audio with its transcription and translation."""
    start: float
    end: float
    text: str
    translation: str | None = None
    pronunciation: str | None = None
    audio_file: str | None = None

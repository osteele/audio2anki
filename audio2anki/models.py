"""Data models for audio2anki."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .transcribe import TranscriptionSegment

from dataclasses import dataclass
from pathlib import Path

from .transcribe import TranscriptionSegment

AudioSegment = TranscriptionSegment


@dataclass
class PipelineResult:
    deck_dir: Path
    segments: list[AudioSegment]

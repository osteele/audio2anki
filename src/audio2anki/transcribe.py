"""Transcription module using OpenAI API."""

import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from rich.progress import Progress, TaskID

from .models import AudioSegment


def load_transcript(file: Path) -> list[AudioSegment]:
    """Load transcript from file."""
    segments = []
    with open(file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                start, end, text = float(parts[0]), float(parts[1]), parts[2]
                segments.append(AudioSegment(start=start, end=end, text=text))
    return segments


def transcribe_audio(
    audio_file: Path,
    transcript_path: Path | None,
    model: str,
    language: str | None = None,
    min_length: float | None = None,
    max_length: float | None = None,
) -> list[AudioSegment]:
    """Transcribe audio using OpenAI Whisper API.
    
    Args:
        audio_file: Path to audio file
        transcript_path: Path to transcript file (optional)
        model: Whisper model to use (e.g. "whisper-1")
        language: Language code (e.g. "en", "zh", "ja")
        min_length: Minimum segment length in seconds
        max_length: Maximum segment length in seconds
    """
    if transcript_path and transcript_path.exists():
        segments = load_transcript(transcript_path)
    else:
        # Check for API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Initialize OpenAI client
        client = OpenAI()

        # Open audio file
        with open(audio_file, "rb") as f:
            try:
                # Transcribe using OpenAI API
                response = client.audio.transcriptions.create(
                    file=f,
                    model=model,
                    language=language,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
            except Exception as e:
                raise RuntimeError(f"Transcription failed: {str(e)}")

        # Convert to segments
        segments = []
        for segment in response.segments:
            start = float(segment.start)
            end = float(segment.end)
            text = segment.text.strip()
            segments.append(AudioSegment(start=start, end=end, text=text))

    # Filter by length
    filtered_segments = []
    for segment in segments:
        duration = segment.end - segment.start
        if min_length is not None and duration <= min_length:
            continue
        if max_length is not None and duration > max_length:
            continue
        filtered_segments.append(segment)

    return filtered_segments

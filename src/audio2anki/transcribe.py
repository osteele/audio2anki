"""Transcription module using OpenAI API."""

import os
from pathlib import Path

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
    task_id: TaskID | None = None,
    progress: Progress | None = None,
    language: str | None = None,
    min_length: float | None = None,
    max_length: float | None = None,
) -> list[AudioSegment]:
    """Transcribe audio using OpenAI Whisper API.

    Args:
        audio_file: Path to audio file
        transcript_path: Path to transcript file (optional)
        model: Whisper model to use (e.g. "whisper-1")
        task_id: Progress bar task ID (optional)
        progress: Progress bar instance (optional)
        language: Language code (e.g. "en", "zh", "ja")
        min_length: Minimum segment length in seconds
        max_length: Maximum segment length in seconds
    """
    if transcript_path and transcript_path.exists():
        segments = load_transcript(transcript_path)
        if progress and task_id:
            progress.update(task_id, completed=100)
        return segments

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Update progress
    if progress and task_id:
        progress.update(task_id, description="Transcribing audio with Whisper...")

    # Transcribe audio
    with open(audio_file, "rb") as f:
        response = (
            client.audio.transcriptions.create(
                file=f,
                model=model,
                response_format="verbose_json",
                language=language,
            )
            if language
            else client.audio.transcriptions.create(
                file=f,
                model=model,
                response_format="verbose_json",
            )
        )

    # Process segments
    segments = []
    if not hasattr(response, "segments"):
        raise ValueError("Invalid response from OpenAI: missing segments")

    for segment in response.segments or []:
        start = float(segment.start)
        end = float(segment.end)

        # Apply length constraints if specified
        if min_length and (end - start) < min_length:
            continue
        if max_length and (end - start) > max_length:
            continue

        segments.append(
            AudioSegment(
                start=start,
                end=end,
                text=segment.text.strip(),
            )
        )

    # Save transcript if path provided
    if transcript_path:
        with open(transcript_path, "w") as f:
            for segment in segments:
                f.write(f"{segment.start}\t{segment.end}\t{segment.text}\n")

    if progress and task_id:
        progress.update(task_id, completed=100)
    return segments

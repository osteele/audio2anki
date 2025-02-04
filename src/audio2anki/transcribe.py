"""Transcription module using OpenAI API."""

import os
from pathlib import Path
from typing import cast

import openai
from openai import NotGiven, OpenAI
from openai.types.audio import TranscriptionVerbose
from rich.progress import Progress, TaskID

from .models import AudioSegment


def load_transcript(file: Path) -> list[AudioSegment]:
    """Load transcript from file."""
    import json

    segments = []
    try:
        with open(file) as f:
            data = json.load(f)
            for segment in data:
                segments.append(
                    AudioSegment(
                        start=float(segment["start"]),
                        end=float(segment["end"]),
                        text=segment["text"],
                    )
                )
    except json.JSONDecodeError:
        # Try tab-separated format as fallback
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
    task_id: TaskID,
    progress: Progress,
    language: str | None = None,
    min_length: float = 0.0,
    max_length: float = float("inf"),
) -> list[AudioSegment]:
    """Transcribe audio file using OpenAI's Whisper model."""
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Create OpenAI client
    client = OpenAI()

    try:
        # Transcribe audio
        with audio_file.open("rb") as file:
            response = client.audio.transcriptions.create(
                file=file,
                model="whisper-1",
                response_format="verbose_json",
                language=language if language and language != "auto" else NotGiven(),
            )
        response = cast(TranscriptionVerbose, response)

        # Process segments
        segments = []
        if response.segments:
            for segment in response.segments:
                # Apply length filters
                duration = segment.end - segment.start
                if duration < min_length or duration > max_length:
                    continue

                segments.append(
                    AudioSegment(
                        start=segment.start,
                        end=segment.end,
                        text=segment.text,
                        translation=None,
                        pronunciation=None,
                        audio_file=None,
                    )
                )

        # Write transcript if requested
        if transcript_path:
            with open(transcript_path, "w") as f:
                for segment in segments:
                    f.write(f"{segment.text}\n")

        progress.update(task_id, completed=100)
        return segments

    except openai.AuthenticationError:
        raise  # Re-raise authentication errors directly
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}") from e

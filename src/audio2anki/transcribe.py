"""Transcription module using OpenAI API."""

import os
from dataclasses import dataclass
from pathlib import Path

import httpx
from openai import OpenAI
from rich.progress import Progress, TaskID


@dataclass
class TranscriptionSegment:
    """A segment of transcribed audio with optional translation."""

    start: float
    end: float
    text: str
    translation: str | None = None
    pronunciation: str | None = None
    audio_file: str | None = None


def format_timestamp(seconds: float) -> str:
    """Convert seconds into SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_srt(segments: list[TranscriptionSegment]) -> str:
    """Format transcription segments into SRT content."""
    srt_lines: list[str] = []
    for idx, segment in enumerate(segments, start=1):
        start_ts = format_timestamp(segment.start)
        end_ts = format_timestamp(segment.end)
        srt_lines.append(f"{idx}")
        srt_lines.append(f"{start_ts} --> {end_ts}")
        srt_lines.append(segment.text)
        srt_lines.append("")
    return "\n".join(srt_lines)


def parse_srt(file: Path) -> list[TranscriptionSegment]:
    """Parse an SRT file into transcription segments."""
    segments: list[TranscriptionSegment] = []
    with open(file, encoding="utf-8") as f:
        content = f.read()
    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = block.splitlines()
        if len(lines) >= 3:
            # First line: index (ignored)
            # Second line: timestamps
            # Remaining lines: text (join them if multiline)
            time_line = lines[1]
            try:
                start_str, end_str = time_line.split(" --> ")

                # Convert timestamp format HH:MM:SS,mmm to seconds
                def ts_to_seconds(ts: str) -> float:
                    h, m, s_ms = ts.split(":")
                    s, ms = s_ms.split(",")
                    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

                start = ts_to_seconds(start_str.strip())
                end = ts_to_seconds(end_str.strip())
            except Exception:
                continue
            text = " ".join(lines[2:]).strip()
            segments.append(TranscriptionSegment(start=start, end=end, text=text))
    return segments


def load_transcript(file: Path) -> list[TranscriptionSegment]:
    """Load transcript from file. Supports TSV or SRT based on file extension."""
    if file.suffix.lower() == ".srt":
        return parse_srt(file)
    segments: list[TranscriptionSegment] = []
    with open(file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                start, end, text = float(parts[0]), float(parts[1]), parts[2]
                segments.append(TranscriptionSegment(start=start, end=end, text=text))
    return segments


def save_transcript(segments: list[TranscriptionSegment], file: Path) -> None:
    """Save transcript to file. If the file extension is .srt, format as SRT, otherwise TSV."""
    if file.suffix.lower() == ".srt":
        content = format_srt(segments)
    else:
        content = "\n".join(f"{s.start}\t{s.end}\t{s.text}" for s in segments)
    with open(file, "w", encoding="utf-8") as f:
        f.write(content)


def transcribe_audio(
    audio_file: Path,
    transcript_path: Path,
    model: str,
    task_id: TaskID | None = None,
    progress: Progress | None = None,
    language: str | None = None,
    min_length: float | None = None,
    max_length: float | None = None,
) -> list[TranscriptionSegment]:
    """Transcribe audio using OpenAI Whisper API.

    Args:
        audio_file: Path to audio file
        transcript_path: Path where transcript will be saved
        model: Whisper model to use (e.g. "whisper-1")
        task_id: Progress bar task ID (optional)
        progress: Progress bar instance (optional)
        language: Language code (e.g. "en", "zh", "ja")
        min_length: Minimum segment length in seconds
        max_length: Maximum segment length in seconds

    Returns:
        List of transcription segments

    Raises:
        ValueError: If OPENAI_API_KEY is not set or if the response is invalid
        RuntimeError: If transcription fails
    """
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Update progress
    if progress and task_id:
        progress.update(task_id, description="Transcribing audio with Whisper...")

    try:
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
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Transcription failed: {e.response.status_code} {e.response.reason_phrase}") from e
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}") from e

    # Process segments
    segments: list[TranscriptionSegment] = []
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

        segments.append(TranscriptionSegment(start=start, end=end, text=segment.text.strip()))

    # Save transcript to the output path
    save_transcript(segments, transcript_path)

    if progress and task_id:
        progress.update(task_id, completed=100)
    return segments

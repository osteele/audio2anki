"""Transcription module using OpenAI Whisper."""

import os
from pathlib import Path
from typing import Any

import whisper
from rich.progress import Progress

from .cli import AudioSegment


def load_transcript(file: Path) -> list[AudioSegment]:
    """Load segments from a transcript file."""
    segments = []
    current_segment = None
    
    with open(file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = None
            elif current_segment is None:
                current_segment = AudioSegment(start=0.0, end=0.0, text=line)
            else:
                current_segment.text += " " + line
    
    if current_segment:
        segments.append(current_segment)
    
    return segments


def transcribe_audio(
    audio_path: Path,
    transcript_path: Path | None,
    *,
    model: str = "small",
    language: str | None = None,
    min_length: float = 1.0,
    max_length: float = 15.0,
) -> list[AudioSegment]:
    """Transcribe audio using Whisper."""
    if transcript_path:
        return load_transcript(transcript_path)
    
    # Load Whisper model
    whisper_model = whisper.load_model(model)
    
    # Transcribe
    result = whisper_model.transcribe(
        str(audio_path),
        language=language,
        task="transcribe",
        verbose=False,
    )
    
    # Convert segments
    segments: list[AudioSegment] = []
    current_segment = None
    
    for seg in result["segments"]:
        start: float = seg["start"]
        end: float = seg["end"]
        text: str = seg["text"].strip()
        
        # Skip empty segments
        if not text:
            continue
        
        # Merge short segments
        if current_segment and (start - current_segment.end) < 0.3:
            duration = end - current_segment.start
            if duration <= max_length:
                current_segment.end = end
                current_segment.text += " " + text
                continue
        
        # Add completed segment
        if current_segment:
            segments.append(current_segment)
        
        # Start new segment
        current_segment = AudioSegment(
            start=start,
            end=end,
            text=text,
        )
    
    # Add final segment
    if current_segment:
        segments.append(current_segment)
    
    return segments

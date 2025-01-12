"""Anki deck generation module."""

import csv
from pathlib import Path

from .cli import AudioSegment


def create_anki_deck(segments: list[AudioSegment], output_dir: Path) -> Path:
    """Create Anki-compatible CSV file."""
    output_file = output_dir / "deck.txt"
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        
        # Write header
        writer.writerow(["Text", "Pronunciation", "Translation", "Audio"])
        
        # Write segments
        for segment in segments:
            writer.writerow([
                segment.text,
                segment.pronunciation or "",
                segment.translation or "",
                f"[sound:{segment.audio_file}]" if segment.audio_file else "",
            ])
    
    return output_file

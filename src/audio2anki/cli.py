#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

from .transcribe import transcribe_audio
from .translate import translate_segments
from .audio import split_audio
from .anki import create_anki_deck
from .models import AudioSegment

console = Console()


def process_audio(
    input_file: Path,
    transcript_file: Path | None,
    output_dir: Path,
    model: str,
    language: str | None,
    min_length: float,
    max_length: float,
    debug: bool,
) -> list[AudioSegment]:
    """Process audio file and return segments with translations."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        # Step 1: Transcribe
        task_id = progress.add_task("Transcribing audio...", total=100)
        segments = transcribe_audio(
            input_file,
            transcript_file,
            model=model,
            language=language,
            min_length=min_length,
            max_length=max_length,
        )
        progress.update(task_id, completed=100)
        
        # Step 2: Translate and get pronunciation
        task_id = progress.add_task("Translating segments...", total=len(segments))
        translated_segments = translate_segments(segments, "english", task_id, progress)
        
        # Step 3: Split audio
        task_id = progress.add_task("Splitting audio...", total=len(segments))
        audio_segments = split_audio(input_file, segments, output_dir, task_id, progress)
        
        if debug:
            debug_file = output_dir / "debug.txt"
            with open(debug_file, "w") as f:
                for seg in audio_segments:
                    f.write(f"Time: {seg.start:.2f}-{seg.end:.2f}\n")
                    f.write(f"Text: {seg.text}\n")
                    f.write(f"Translation: {seg.translation}\n")
                    f.write(f"Pronunciation: {seg.pronunciation}\n")
                    f.write(f"Audio: {seg.audio_file}\n\n")
    
    return audio_segments


@click.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--transcript",
    type=click.Path(exists=True, path_type=Path),
    help="Optional transcript file",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="./output",
    help="Output directory",
)
@click.option(
    "--model",
    type=click.Choice(["whisper-1"]),
    default="whisper-1",
    help="Whisper model to use",
)
@click.option(
    "--language",
    help="Source language (auto-detect if not specified)",
)
@click.option(
    "--min-length",
    type=float,
    default=1.0,
    help="Minimum segment length in seconds",
)
@click.option(
    "--max-length",
    type=float,
    default=15.0,
    help="Maximum segment length in seconds",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Generate debug information",
)
def main(
    input_file: Path,
    transcript: Optional[Path],
    output: Path,
    model: str,
    language: Optional[str],
    min_length: float,
    max_length: float,
    debug: bool,
) -> None:
    """Convert audio files to Anki flashcards with translations.
    
    Process an audio or video file, optionally using an existing transcript,
    and create an Anki-compatible deck with translations and audio segments.
    """
    try:
        segments = process_audio(
            input_file,
            transcript,
            output,
            model,
            language,
            min_length,
            max_length,
            debug,
        )
        
        # Create Anki deck
        deck_file = create_anki_deck(segments, output)
        console.print(f"\n✨ Created Anki deck: {deck_file}")
        console.print("Import this file into Anki along with the media files in the output directory.")
        
    except Exception as e:
        console.print(f"\n❌ Error: {str(e)}", style="red")
        raise click.Abort()


if __name__ == "__main__":
    main()

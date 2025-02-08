#!/usr/bin/env python3

import os
import platform
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn

from .anki import create_anki_deck
from .audio_utils import split_audio
from .models import AudioSegment
from .transcribe import transcribe_audio
from .translate import translate_segments

console = Console()


def get_anki_media_path() -> str:
    """Get the platform-specific Anki media collection path."""
    system = platform.system()
    username = os.environ.get("USER") or os.environ.get("USERNAME")

    if system == "Darwin":  # macOS
        return f"/Users/{username}/Library/Application Support/Anki2/User 1/collection.media"
    elif system == "Windows":
        return f"C:/Users/{username}/AppData/Roaming/Anki2/User 1/collection.media"
    else:  # Linux
        return f"/home/{username}/.local/share/Anki2/User 1/collection.media"


def print_import_instructions(console: Console, output_dir: Path) -> None:
    """Print instructions for importing into Anki."""
    media_path = get_anki_media_path()

    instructions = (
        "[bold]Import Instructions[/bold]\n\n"
        "1. [bold]Import the Deck[/bold]:\n"
        f"   - Open Anki\n"
        f"   - Click File > Import\n"
        f"   - Select: {output_dir}/deck.txt\n"
        "   - In the import dialog:\n"
        '     • Set Type to "Basic"\n'
        "     • Check field mapping:\n"
        "       ‣ Field 1: Front (Original text)\n"
        "       ‣ Field 2: Pronunciation\n"
        "       ‣ Field 3: Back (Translation)\n"
        "       ‣ Field 4: Audio\n"
        '     • Set "Field separator" to "Tab"\n'
        '     • Check "Allow HTML in fields"\n\n'
        "2. [bold]Import the Audio[/bold]:\n"
        f"   - Copy all files from: {output_dir}/media\n"
        f"   - Paste them into: {media_path}\n\n"
        "3. [bold]Verify the Import[/bold]:\n"
        "   - Check that cards show:\n"
        "     • Front: Original text\n"
        "     • Back: Pronunciation, translation, and audio\n"
        "   - Test the audio playback"
    )

    console.print()
    console.print(Panel(instructions, title="Next Steps", expand=False))


def process_audio(
    input_file: Path,
    transcript_file: Path | None,
    output_dir: Path,
    model: str,
    language: str | None,
    min_length: float,
    max_length: float,
    silence_thresh: int,
    debug: bool = False,
    progress: Progress | None = None,
) -> list[AudioSegment]:
    """Process audio file and return segments with translations."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create progress bar if not provided
    if progress is None:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )

    # Step 1: Transcribe
    task_id = progress.add_task("Starting transcription...", total=100)
    segments = transcribe_audio(
        input_file,
        transcript_file,
        model,
        task_id,
        progress,
        language=language,
        min_length=min_length,
        max_length=max_length,
    )

    # Step 2: Translate and get pronunciation
    task_id = progress.add_task("Starting translation...", total=len(segments))

    # Default to Chinese if not specified and text contains Chinese characters
    if not language and any("\u4e00" <= c <= "\u9fff" for c in segments[0].text):
        source_language = "chinese"
    else:
        source_language = language

    segments = translate_segments(
        segments,
        "english",
        task_id,
        progress,
        source_language=source_language,
    )

    # Step 3: Split audio
    task_id = progress.add_task("Starting audio split...", total=len(segments))
    segments = split_audio(
        input_file,
        segments,
        output_dir,
        task_id,
        progress,
        silence_thresh=silence_thresh,
    )

    if debug:
        debug_file = output_dir / "debug.txt"
        with open(debug_file, "w") as f:
            for seg in segments:
                f.write(f"Time: {seg.start:.2f}-{seg.end:.2f}\n")
                f.write(f"Text: {seg.text}\n")
                f.write(f"Translation: {seg.translation}\n")
                f.write(f"Pronunciation: {seg.pronunciation}\n")
                f.write(f"Audio: {seg.audio_file}\n\n")

    return segments


@click.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--transcript",
    type=click.Path(path_type=Path),
    help="Path to transcript file (optional)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("output"),
    help="Output directory (default: output)",
)
@click.option(
    "--model",
    type=str,
    default="whisper-1",
    help="Whisper model to use (default: whisper-1)",
)
@click.option(
    "--language",
    type=str,
    help="Source language (default: auto-detect)",
)
@click.option(
    "--min-length",
    type=float,
    default=1.0,
    help="Minimum segment length in seconds (default: 1.0)",
)
@click.option(
    "--max-length",
    type=float,
    default=15.0,
    help="Maximum segment length in seconds (default: 15.0)",
)
@click.option(
    "--silence-thresh",
    type=int,
    default=-40,
    help="Silence threshold in dB (default: -40)",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable debug output",
)
@click.option(
    "--quiet/--no-quiet",
    default=False,
    help="Suppress import instructions",
)
def main(
    input_file: Path,
    transcript: Path | None,
    output: Path,
    model: str,
    language: str | None,
    min_length: float,
    max_length: float,
    silence_thresh: int,
    debug: bool,
    quiet: bool,
) -> None:
    """Convert audio files to Anki flashcards with translations.

    Process an audio or video file, optionally using an existing transcript,
    and create an Anki-compatible deck with translations and audio segments.
    """
    console = Console()
    segments = []
    deck_file = None

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            segments = process_audio(
                input_file,
                transcript,
                output,
                model,
                language,
                min_length,
                max_length,
                silence_thresh,
                debug,
                progress,
            )

            # Create Anki deck
            deck_file = create_anki_deck(segments, output)

        # Show success message and instructions after progress completes
        console.print(f"\n✨ Created Anki deck with {len(segments)} cards")
        console.print(f"📝 Deck file: {deck_file}")
        console.print(f"🔊 Audio files: {output}/media/")

        # Show import instructions unless quiet mode is enabled
        if not quiet:
            print_import_instructions(console, output)

    except Exception as e:
        console.print(f"\n❌ Error: {str(e)}")
        raise click.Abort() from None


if __name__ == "__main__":
    main()

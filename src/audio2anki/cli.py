#!/usr/bin/env python3

import logging
import os
import platform
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn, TaskID
import sys
import subprocess

from .anki import create_anki_deck
from .audio import AudioCleaningError, clean_audio, split_audio
from .models import AudioSegment
from .transcribe import transcribe_audio
from .translate import translate_segments, translate_text

console = Console()
logger = logging.getLogger(__name__)


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


def check_system_dependencies() -> None:
    """Check if required system dependencies are installed."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except FileNotFoundError:
        console.print("[red]❌ Error: ffmpeg is not installed[/red]")
        console.print("\nPlease install ffmpeg:")
        console.print("\nOn macOS:")
        console.print("  brew install ffmpeg")
        console.print("\nOn Ubuntu/Debian:")
        console.print("  sudo apt-get install ffmpeg")
        console.print("\nOn Windows:")
        console.print("1. Install Chocolatey from https://chocolatey.org/")
        console.print("2. Run: choco install ffmpeg")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Error: ffmpeg check failed: {e.stderr.decode()}[/red]")
        sys.exit(1)


def process_audio(
    input_file: Path,
    transcript_file: Path | None,
    output_dir: Path,
    model: str,
    target_language: str,
    min_length: float,
    max_length: float,
    silence_thresh: int,
    debug: bool,
    progress: Progress,
    clean_mode: str | None = None,
) -> list[AudioSegment]:
    """Process audio file and create Anki deck."""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Transcribe audio
    task_id = progress.add_task("Transcribing audio...", total=100)
    segments = transcribe_audio(
        input_file,
        transcript_file,
        model=model,
        task_id=task_id,
        progress=progress,
    )
    progress.update(task_id, completed=100, visible=False)

    # Translate segments
    if target_language:
        task_id = progress.add_task("Translating...", total=len(segments))
        segments = translate_segments(segments, task_id, progress, target_language)
        progress.update(task_id, completed=len(segments), visible=False)

    # Split audio into segments
    task_id = progress.add_task("Splitting audio...", total=len(segments))
    segments = split_audio(
        input_file,
        segments,
        output_dir,
        min_length=min_length,
        max_length=max_length,
        silence_thresh=silence_thresh,
        task_id=task_id,
        progress=progress,
        clean_mode=clean_mode,
    )
    progress.update(task_id, completed=len(segments), visible=False)

    # Create Anki deck
    deck_file = create_anki_deck(segments, output_dir)

    # Write debug info
    if debug:
        debug_file = output_dir / "debug.txt"
        with open(debug_file, "w") as f:
            for segment in segments:
                f.write(f"Start: {segment.start:.2f}s\n")
                f.write(f"End: {segment.end:.2f}s\n")
                f.write(f"Text: {segment.text}\n")
                if segment.translation:
                    f.write(f"Translation: {segment.translation}\n")
                if segment.pronunciation:
                    f.write(f"Pronunciation: {segment.pronunciation}\n")
                if segment.audio_file:
                    f.write(f"Audio: {segment.audio_file}\n")
                f.write("\n")

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
    "-l",
    help="Source language (e.g. 'en', 'zh'). If not specified, will be auto-detected.",
    default="",
    type=str,
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
@click.option(
    "--clean/--no-clean",
    is_flag=True,
    default=None,
    help="Force audio cleaning (requires HF_TOKEN) / Never clean audio",
)
@click.option(
    "--target-language",
    type=str,
    default="",
    help="Target language (e.g. 'en', 'zh')",
)
@click.option(
    "--use-deepl/--no-use-deepl",
    default=False,
    help="Use DeepL for translation",
)
def main(
    input_file: Path,
    transcript: Path | None,
    output: Path,
    model: str,
    language: str,
    min_length: float,
    max_length: float,
    silence_thresh: int,
    debug: bool,
    quiet: bool,
    clean: bool | None,
    target_language: str,
    use_deepl: bool,
) -> None:
    """Convert audio files to Anki flashcards with translations."""
    # Configure logging
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        # Suppress INFO messages from speechbrain
        logging.getLogger('speechbrain').setLevel(logging.WARNING)

    # Check system dependencies
    check_system_dependencies()

    console = Console()
    segments = []
    deck_file = None

    clean_mode = "force" if clean else ("skip" if clean is False else None)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            # Process audio
            segments = process_audio(
                input_file,
                transcript,
                output,
                model,
                target_language,
                min_length,
                max_length,
                silence_thresh,
                debug,
                progress,
                clean_mode=clean_mode,
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

    except AudioCleaningError as e:
        console.print(f"\n❌ Audio cleaning error: {str(e)}")
        console.print("Make sure HF_TOKEN is set in your environment if using --clean")
        raise click.Abort() from e
    except Exception as e:
        console.print(f"\n❌ Error: {str(e)}")
        raise click.Abort() from e


if __name__ == "__main__":
    main()

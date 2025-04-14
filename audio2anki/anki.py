"""Anki deck generation module."""

import csv
import os
import platform
import shutil
from pathlib import Path
from typing import Any

from rich.progress import Progress, TaskID

from .audio_utils import split_audio
from .config import load_config
from .models import AudioSegment
from .pipeline import PipelineProgress

# Template for the README.md file
README_TEMPLATE = r"""# Anki Deck Import Instructions

This export package contains:
- `deck.csv`: A CSV file with Chinese text, pinyin, English translation, and audio references
- `deck.txt`: A tab-separated file with the same content for manual import
- Audio files in the `media` folder (named with content hashes to avoid conflicts)
- `import_to_anki.sh`: A shell script to help with importing into Anki

## Import Options

### Option 1: Using the Shell Script (Recommended)
1. Unzip this file to a location on your computer
2. Open Terminal
3. Navigate to the unzipped directory
4. Run the script: `./import_to_anki.sh`

The script will:
- Install `uv` if it's not already present (via https://docs.astral.sh/uv/getting-started/installation/)
- Launch Anki if it's not already running (macOS only)
- Import the cards directly into your selected deck using the `add2anki` tool

### Option 2: Manual Installation with uv
1. Install `uv` from https://docs.astral.sh/uv/getting-started/installation/
2. Run: `uv tool add2anki deck.csv --tags audio2anki`

### Option 3: Manual Import
1. Open Anki
2. Click "File" > "Import"
3. Select the `deck.txt` file in this directory
4. In the import dialog:
    - Set "Type" to "Basic"
    - Set "Deck" to your desired deck name
    - Set "Fields separated by" to "Tab"
5. Import the audio files:
    - Copy all files from the `media` folder
    - Paste them into your Anki media folder: `{media_path}`

## Finding Your Anki Media Folder

The Anki media folder is typically located at:

- **macOS**: `~/Library/Application Support/Anki2/[profile]/collection.media`
- **Windows**: `C:\\Users\\[username]\\AppData\\Roaming\\Anki2\\[profile]\\collection.media`
- **Linux**: `~/.local/share/Anki2/[profile]/collection.media`

Where `[profile]` is usually `User 1` if you haven't created additional profiles.

You can also find it from Anki by going to:
1. Tools > Add-ons > Open Add-ons Folder
2. Go up one directory level
3. Navigate to your profile folder, then to `collection.media`

Note: The media files are named with a hash of the source audio to avoid conflicts.
{alias_note}
"""


def get_anki_media_dir() -> Path:
    """Get the Anki media directory for the current platform."""
    system = platform.system()
    home = Path.home()

    if system == "Darwin":  # macOS
        return home / "Library/Application Support/Anki2/User 1/collection.media"
    elif system == "Windows":
        return Path(os.getenv("APPDATA", "")) / "Anki2/User 1/collection.media"
    else:  # Linux and others
        return home / ".local/share/Anki2/User 1/collection.media"


def is_deck_folder(path: Path) -> bool:
    """Check if a path is an existing Anki deck folder."""
    deck_csv = path / "deck.csv"
    deck_txt = path / "deck.txt"
    media_dir = path / "media"
    return path.exists() and deck_csv.exists() and deck_txt.exists() and media_dir.exists()


def create_anki_deck(
    segments: list[AudioSegment],
    output_dir: Path,
    task_id: TaskID | None = None,
    progress: Progress | None = None,
    input_audio_file: Path | None = None,
    source_language: str | None = None,
    target_language: str | None = None,
) -> Path:
    """Create Anki-compatible deck directory with media files.

    Args:
        segments: List of audio segments to include in the deck
        output_dir: Directory to create the deck in
        task_id: Progress bar task ID (optional)
        progress: Progress bar instance (optional)
        input_audio_file: Path to the original audio file (optional)
        source_language: Source language (e.g. "zh", "ja")
        target_language: Target language (e.g. "en", "fr")

    Returns:
        Path to the created deck directory
    """
    # Use the provided output_dir directly (already processed for output folder options)
    # Just ensure directory exists and create media folder
    deck_dir = output_dir
    deck_dir.mkdir(parents=True, exist_ok=True)
    media_dir = deck_dir / "media"
    media_dir.mkdir(exist_ok=True)

    # Split audio into segments if input file is provided
    if input_audio_file and progress and task_id:
        config = load_config()
        segments = split_audio(
            input_audio_file,
            segments,
            media_dir,
            task_id,
            progress,
            silence_thresh=config.silence_thresh,
            padding_ms=config.audio_padding_ms,
        )

    # Initialize columns based on language
    target_language_name = "English" if target_language == "en" else (target_language or "Translation").capitalize()
    if source_language == "zh":
        columns = ["Hanzi", "Pinyin", target_language_name, "Audio"]
    elif source_language == "ja":
        columns = ["Japanese", "Pronunciation", target_language_name, "Audio"]
    else:
        columns = ["Text", "Pronunciation", target_language_name, "Audio"]

    def create_deck_file(file_path: Path, delimiter: str | None = None, add_anki_header: bool = False):
        """Create a deck file with the given segments and columns.

        Args:
            file_path: Path to the output file
            delimiter: CSV delimiter to use (None for default, '\t' for TSV)
            add_anki_header: Whether to add Anki-specific header lines
        """
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            # Add Anki-specific headers if needed
            if add_anki_header:
                f.write("#separator:tab\n")
                f.write(f"#columns:{','.join(columns)}\n")

            # Create writer with appropriate delimiter
            writer = csv.writer(f, delimiter=delimiter or ",")
            writer.writerow(columns)

            # Track progress if available
            if progress and task_id and file_path.name == "deck.txt":  # Only track for first file
                total = len(segments)
                progress.update(task_id, total=total)

            # Write all segments
            for segment in segments:
                # Create a dictionary with all possible fields
                fields = {
                    "Hanzi": segment.text,
                    "Japanese": segment.text,
                    "Text": segment.text,
                    "Pinyin": segment.pronunciation or "",
                    "Pronunciation": segment.pronunciation or "",
                    target_language_name: segment.translation or "",
                    "English": segment.translation or "",
                    "Audio": f"[sound:{segment.audio_file}]" if segment.audio_file else "",
                }

                # Use a list comprehension to create the row based on columns
                row = [fields[column] for column in columns]
                writer.writerow(row)

                # Update progress for the first file only
                if progress and task_id and file_path.name == "deck.txt":
                    progress.update(task_id, advance=1)

    # Create deck.txt file (tab-delimited with Anki headers)
    create_deck_file(deck_dir / "deck.txt", delimiter="\t", add_anki_header=True)

    # Create deck.csv file (comma-delimited without Anki headers)
    create_deck_file(deck_dir / "deck.csv")

    # Copy the import_to_anki.sh script to the deck directory
    script_path = Path(__file__).parent / "resources" / "import_to_anki.sh"
    target_script_path = deck_dir / "import_to_anki.sh"
    try:
        shutil.copy2(script_path, target_script_path)
        # Make the script executable
        os.chmod(target_script_path, 0o755)
    except Exception as e:
        print(f"Warning: Could not copy import_to_anki.sh script: {e}")

    # Create README.md from template
    media_path = get_anki_media_dir()
    alias_term = (
        "alias" if platform.system() == "Darwin" else "shortcut" if platform.system() == "Windows" else "symbolic link"
    )
    article = "an" if alias_term[0].lower() in "aeiou" else "a"
    alias_note = f"{article.capitalize()} {alias_term} to your Anki media folder is provided for convenience."

    readme_content = README_TEMPLATE.format(media_path=media_path, alias_note=alias_note)

    readme_file = deck_dir / "README.md"
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write(readme_content)

    # Create a symbolic link to Anki media folder
    anki_media_dir = get_anki_media_dir()
    media_link = deck_dir / "anki_media"
    try:
        if media_link.exists():
            media_link.unlink()
        media_link.symlink_to(anki_media_dir, target_is_directory=True)
    except Exception as e:
        print(f"Warning: Could not create symbolic link to Anki media folder: {e}")

    # Copy media files to Anki media directory
    if anki_media_dir.exists():
        for file in media_dir.glob("*.mp3"):
            try:
                shutil.copy2(file, anki_media_dir)
            except Exception as e:
                print(f"Warning: Could not copy {file.name} to Anki media folder: {e}")

    return deck_dir


def generate_anki_deck(
    segments_file: str | Path,
    progress: PipelineProgress,
    **kwargs: Any,
) -> Path:
    """Process deck generation stage in the pipeline.

    Args:
        segments_file: Path to the segments JSON file containing transcriptions, translations, and pronunciations
        progress: Pipeline progress tracker
        **kwargs: Additional arguments passed from the pipeline

    Returns:
        Path to the generated deck directory
    """
    from .transcribe import load_transcript, load_transcript_json

    # Type assertion to handle the progress object correctly
    pipeline_progress = progress
    if not pipeline_progress:
        raise TypeError("Expected PipelineProgress object")

    segments_path = Path(segments_file)
    if not segments_path.exists():
        raise FileNotFoundError(f"Segments file not found: {segments_path}")

    # Get the output path from kwargs
    deck_dir = kwargs.get("output_folder")
    if deck_dir is None:
        # Fallback to cwd if no output folder was specified
        deck_dir = Path.cwd()
    elif isinstance(deck_dir, str):
        # Convert string to Path
        deck_dir = Path(deck_dir)

    # Determine if this is a deck folder that needs to be replaced
    is_deck_folder = False
    if deck_dir.exists():
        deck_csv = deck_dir / "deck.csv"
        deck_txt = deck_dir / "deck.txt"
        media_dir = deck_dir / "media"
        is_deck_folder = (deck_csv.exists() or deck_txt.exists()) and media_dir.exists()

        # If it's a deck folder, clean up any existing files
        if is_deck_folder:
            import os
            import shutil

            for item in deck_dir.glob("*"):
                if item.is_file() or item.is_symlink():
                    os.unlink(item)
                elif item.is_dir() and item.name != "anki_media":  # Keep the anki_media symlink
                    shutil.rmtree(item)

    # Create directory if it doesn't exist
    deck_dir.mkdir(parents=True, exist_ok=True)

    # Check if the file is a JSON file
    is_json = segments_path.suffix.lower() == ".json"

    # Load segments - either from JSON or from legacy SRT files
    if is_json:
        # For JSON files, use the JSON loader which has all data
        enriched_segments = load_transcript_json(segments_path)
    else:
        # For backward compatibility, handle SRT files
        enriched_segments = load_transcript(segments_path)

        # Store translations before they get overwritten
        for seg in enriched_segments:
            seg.translation = seg.text

        # Get transcription and pronunciation files from context
        transcription_file = kwargs.get("transcription_file")
        pronunciation_file = kwargs.get("pronunciation_file")

        # Load transcription and pronunciation if available
        if transcription_file:
            transcription_segments = load_transcript(Path(transcription_file))
            # Update text from transcription
            for t_seg, tr_seg in zip(transcription_segments, enriched_segments, strict=True):
                tr_seg.text = t_seg.text

        if pronunciation_file:
            pronunciation_segments = load_transcript(Path(pronunciation_file))
            # Update pronunciation
            for p_seg, tr_seg in zip(pronunciation_segments, enriched_segments, strict=True):
                tr_seg.pronunciation = p_seg.text

    # Get the task ID for the current stage
    task_id = None
    if pipeline_progress.current_stage:
        task_id = pipeline_progress.stage_tasks.get(pipeline_progress.current_stage)

    # Get the original audio file path from kwargs
    input_audio_file = kwargs.get("input_audio_file")
    if input_audio_file:
        input_audio_file = Path(input_audio_file)

    # Create the Anki deck
    deck_dir = create_anki_deck(
        enriched_segments,
        deck_dir,  # Use the specified output directory
        task_id,
        pipeline_progress.progress,
        input_audio_file=input_audio_file,
        source_language=kwargs.get("source_language"),
        target_language=kwargs.get("target_language"),
    )

    return deck_dir

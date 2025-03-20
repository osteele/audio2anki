"""Anki deck generation module."""

import csv
import os
import platform
import shutil
from pathlib import Path
from typing import Any

from rich.progress import Progress, TaskID

from .audio_utils import split_audio
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
- Launch Anki if it's not already running
- Import the cards directly into your selected deck using the `add2anki` tool

### Option 2: Manual Installation with uv
1. Install `uv` from https://docs.astral.sh/uv/getting-started/installation/
2. Run: `uv tool git+https://github.com/osteele/add2anki@release deck.csv --tags audio2anki`

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
        source_language: Source language (e.g. "chinese", "japanese")
        target_language: Target language (e.g. "english", "french")

    Returns:
        Path to the created deck directory
    """
    # Create deck directory structure
    deck_dir = output_dir / "deck"
    deck_dir.mkdir(parents=True, exist_ok=True)
    media_dir = deck_dir / "media"
    media_dir.mkdir(exist_ok=True)

    # Split audio into segments if input file is provided
    if input_audio_file and progress and task_id:
        segments = split_audio(input_audio_file, segments, media_dir, task_id, progress)

    # Create deck.txt file
    deck_file = deck_dir / "deck.txt"
    with open(deck_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        target_language_name = (target_language or "Translation").capitalize()
        if source_language == "chinese":
            columns = ["Hanzi", "Pinyin", target_language_name, "Audio"]
        elif source_language == "japanese":
            columns = ["Japanese", "Pronunciation", target_language_name, "Audio"]
        else:
            columns = ["Text", "Pronunciation", target_language_name, "Audio"]
        writer.writerow(columns)

        # Write segments
        total = len(segments)
        if progress and task_id:
            progress.update(task_id, total=total)

        for segment in segments:
            writer.writerow(
                [
                    segment.text,
                    segment.pronunciation or "",
                    segment.translation or "",
                    f"[sound:{segment.audio_file}]" if segment.audio_file else "",
                ]
            )
            if progress and task_id:
                progress.update(task_id, advance=1)

    # Create deck.csv file
    csv_file = deck_dir / "deck.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # CSV header based on language
        if source_language == "chinese":
            writer.writerow(["Hanzi", "Pinyin", "English", "Audio"])
        elif source_language == "japanese":
            writer.writerow(["Japanese", "Pronunciation", "English", "Audio"])
        else:
            writer.writerow(["Text", "Pronunciation", "English", "Audio"])

        # Write segments to CSV
        for segment in segments:
            audio_path = f"media/{segment.audio_file}" if segment.audio_file else ""
            writer.writerow([segment.text, segment.pronunciation or "", segment.translation or "", audio_path])

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
    input_data: str | Path,
    progress: PipelineProgress,
    **kwargs: Any,
) -> Path:
    """Process deck generation stage in the pipeline.

    Args:
        input_data: Path to the input file (transcript with audio files)
        progress: Pipeline progress tracker
        **kwargs: Additional arguments passed from the pipeline

    Returns:
        Path to the generated deck directory
    """
    from .transcribe import load_transcript

    # Type assertion to handle the progress object correctly
    pipeline_progress = progress
    if not pipeline_progress:
        raise TypeError("Expected PipelineProgress object")

    input_path = Path(input_data)

    # Use the output_path from kwargs if provided, otherwise use current directory
    output_path = kwargs.get("output_path")
    deck_dir = Path.cwd() if output_path is None else Path(output_path)

    # Load segments from the translation file
    translation_segments = load_transcript(input_path)
    # Store translations before they get overwritten
    for seg in translation_segments:
        seg.translation = seg.text

    # Get transcription and pronunciation files from context
    transcription_file = kwargs.get("transcription_file")
    pronunciation_file = kwargs.get("pronunciation_file")

    # Load transcription and pronunciation if available
    if transcription_file:
        transcription_segments = load_transcript(Path(transcription_file))
        # Update text from transcription
        for t_seg, tr_seg in zip(transcription_segments, translation_segments, strict=True):
            tr_seg.text = t_seg.text

    if pronunciation_file:
        pronunciation_segments = load_transcript(Path(pronunciation_file))
        # Update pronunciation
        for p_seg, tr_seg in zip(pronunciation_segments, translation_segments, strict=True):
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
        translation_segments,
        deck_dir,  # Use the specified output directory
        task_id,
        pipeline_progress.progress,
        input_audio_file=input_audio_file,
        source_language=kwargs.get("source_language"),
        target_language=kwargs.get("target_language"),
    )

    return deck_dir

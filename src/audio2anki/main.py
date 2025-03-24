"""Main entry point for audio2anki."""

import locale
import logging
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from .config import edit_config, load_config, reset_config, set_config_value
from .pipeline import PipelineOptions, run_pipeline
from .utils import is_deck_folder, is_empty_directory

# Setup basic logging configuration
console = Console()


def configure_logging(debug: bool = False) -> None:
    """Configure logging based on debug flag."""
    level = logging.DEBUG if debug else logging.WARNING

    # Configure root logger with basic format
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    # Set the logger for our package to the same level
    logger = logging.getLogger("audio2anki")
    logger.setLevel(level)

    # If in debug mode, use a more detailed formatter
    if debug:
        # Create console handler if none exists
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            logger.addHandler(console_handler)

        # Set formatter for all handlers
        for handler in logger.handlers:
            handler.setFormatter(logging.Formatter("%(levelname)s: %(name)s - %(message)s"))


def get_system_language() -> str:
    """Get the system language code, falling back to 'english' if not determinable."""
    try:
        # Try to get the system locale
        lang_code = locale.getdefaultlocale()[0]
        if not lang_code:
            return "en"

        # Extract primary language code (e.g., "en" from "en_US")
        primary_code = lang_code.split("_")[0].lower()
        return primary_code
    except Exception:
        return "en"


def language_name_to_code(language_name: str) -> str | None:
    """
    Translates a language name to its corresponding language code.

    Args:
        language_name (str): The name of the language (e.g., "English", "Chinese").

    Returns:
        str: The corresponding language code (e.g., "en", "zh"), or None if not found.
    """
    import langcodes

    if len(language_name) == 2:
        return language_name

    try:
        # Normalize input by lowercasing
        normalized_name = language_name.lower().strip()

        # Get the language object from name
        lang = langcodes.find(normalized_name)

        # Return the ISO 639-1 code (2-letter code)
        return lang.to_tag()
    except (LookupError, AttributeError):
        # Return None if language is not found
        return None


class LeftAlignedMarkdown(Markdown):
    """Markdown with left-aligned h2-h6 headers, but keeping h1 centered."""

    def __init__(self, markup: str, **kwargs: Any) -> None:
        """Initialize with left-aligned heading style for h2-h6."""
        super().__init__(markup, **kwargs)

    def _get_heading_text(self, text: Text, level: int) -> Text:
        """Override to left-align h2-h6 headings while keeping h1 centered."""
        # Only change justification for h2-h6, leave h1 with default (centered)
        if level > 1:
            text.justify = "left"
        return text


def determine_output_path(base_path: Path, output_folder: str | None, input_file: Path) -> Path:
    """Determine the output path for the Anki deck based on provided options.

    Args:
        base_path: Base path for the output (typically current directory)
        output_folder: CLI-specified output folder or None
        input_file: Input audio/video file path

    Returns:
        Path: The determined output directory path

    Raises:
        click.ClickException: If target directory exists and is neither empty nor a deck folder
    """
    # If no output_folder is specified, use decks/input_filename
    if output_folder is None:
        path_a = base_path / "decks" / input_file.stem
        if path_a.exists() and not (is_empty_directory(path_a) or is_deck_folder(path_a)):
            raise click.ClickException(f"Output directory '{path_a}' exists and is neither empty nor a deck folder")
        return path_a

    # Convert to Path and handle absolute vs relative paths
    path_a = Path(output_folder)
    if not path_a.is_absolute():
        path_a = base_path / output_folder

    # If path exists and is either empty or a deck folder, use it directly
    if path_a.exists():
        if is_empty_directory(path_a) or is_deck_folder(path_a):
            return path_a
        # Otherwise, try appending input filename
        path_b = path_a / input_file.stem
        if path_b.exists() and not (is_empty_directory(path_b) or is_deck_folder(path_b)):
            raise click.ClickException(f"Output directory '{path_b}' exists and is neither empty nor a deck folder")
        return path_b

    # Path doesn't exist, use it directly
    return path_a


@click.group()
def cli():
    """Audio2Anki - Generate Anki cards from audio files."""
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--target-language", help="Target language for translation")
@click.option("--source-language", default="chinese", help="Source language for transcription")
@click.option("--output-folder", help="Specify the output folder for the deck")
@click.option("--skip-voice-isolation", is_flag=True, help="Skip the voice isolation step")
@click.option(
    "--translation-provider",
    type=click.Choice(["openai", "deepl"], case_sensitive=False),
    default="openai",
    help="Translation service provider to use (OpenAI or DeepL)",
)
@click.option("--no-cache", is_flag=True, help="Disable using the persistent artifact cache")
@click.option("--skip-cache-cleanup", is_flag=True, help="Skip cleaning up old items from the cache")
def process(
    input_file: str,
    debug: bool = False,
    target_language: str | None = None,
    source_language: str = "chinese",
    output_folder: str | None = None,
    skip_voice_isolation: bool = False,
    translation_provider: str = "openai",
    no_cache: bool = False,
    skip_cache_cleanup: bool = False,
) -> None:
    """Process an audio/video file and generate Anki flashcards."""
    configure_logging(debug)

    if not target_language:
        target_language = get_system_language()

    # Determine output path
    input_file_path = Path(input_file)
    resolved_output_path = determine_output_path(
        base_path=Path.cwd(), output_folder=output_folder, input_file=input_file_path
    )

    # Convert translation_provider string to enum
    from .translate import TranslationProvider

    translation_provider_enum = TranslationProvider.from_string(translation_provider)

    # Translate source_language from a language name to a language code

    options = PipelineOptions(
        source_language=language_name_to_code(source_language),
        target_language=language_name_to_code(target_language),
        debug=debug,
        output_folder=resolved_output_path,
        skip_voice_isolation=skip_voice_isolation,
        translation_provider=translation_provider_enum,
        use_artifact_cache=not no_cache,
        skip_cache_cleanup=skip_cache_cleanup,
    )
    deck_dir = str(run_pipeline(Path(input_file), console, options))

    # Print deck location and instructions
    console.print(f"\n[green]Deck created at:[/] {deck_dir}")

    # Direct user to the README.md file instead of printing it
    readme_path = Path(deck_dir) / "README.md"
    if readme_path.exists():
        console.print(f"[green]See[/] {readme_path} [green]for instructions on how to import the deck into Anki.[/]")


@cli.group()
def config():
    """Manage application configuration."""
    pass


@config.command()
def edit():
    """Open configuration file in default editor."""
    success, message = edit_config()
    if success:
        console.print(f"[green]{message}[/]")
    else:
        raise click.ClickException(message)


@config.command()
@click.argument("key")
@click.argument("value")
def set(key: str, value: str):
    """Set a configuration value."""
    success, message = set_config_value(key, value)
    if success:
        console.print(f"[green]{message}[/]")
    else:
        raise click.ClickException(message)


@config.command()
def list():
    """List all configuration settings."""
    config = load_config()
    config_dict = config.to_dict()

    table = Table(title="Configuration Settings")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Type", style="blue")

    for key, value in config_dict.items():
        table.add_row(key, str(value), type(value).__name__)

    console.print(table)


@config.command()
def reset():
    """Reset configuration to default values."""
    if click.confirm("Are you sure you want to reset all configuration to defaults?"):
        success, message = reset_config()
        if success:
            console.print(f"[green]{message}[/]")
        else:
            raise click.ClickException(message)


# Cache management commands
@cli.group()
def cache():
    """Manage the artifact cache."""
    pass


@cache.command()
def clear():
    """Clear all items from the artifact cache."""
    from . import artifact_cache

    if click.confirm("Are you sure you want to clear the entire cache?"):
        try:
            files_removed, bytes_freed = artifact_cache.clear_cache()
            if files_removed > 0:
                console.print(f"[green]Removed {files_removed} files from cache.[/]")
                console.print(f"[green]Freed up {format_size(bytes_freed)} of disk space.[/]")
            else:
                console.print("[yellow]No files were removed from the cache.[/]")
        except Exception as e:
            console.print(f"[red]Error clearing cache: {e}[/]")


@cache.command()
@click.option("--days", type=int, default=14, help="Remove items older than this many days")
def cleanup(days: int):
    """Remove old items from the artifact cache."""
    from . import artifact_cache

    try:
        files_removed, bytes_freed = artifact_cache.clean_old_artifacts(days)
        if files_removed > 0:
            console.print(f"[green]Removed {files_removed} files from cache older than {days} days.[/]")
            console.print(f"[green]Freed up {format_size(bytes_freed)} of disk space.[/]")
        else:
            console.print(f"[yellow]No files older than {days} days were found in the cache.[/]")
    except Exception as e:
        console.print(f"[red]Error cleaning up cache: {e}[/]")


@cache.command(name="info")
def cache_info():
    """Display information about the artifact cache."""
    from . import artifact_cache

    try:
        stats = artifact_cache.get_cache_stats()

        console.print("\n[bold]Artifact Cache Information:[/]")
        console.print(f"Location: {stats['cache_path']}")
        console.print(f"Size: {stats['total_size_human']} ({stats['file_count']} files)")

        if stats["oldest_artifact"]:
            console.print(f"Oldest artifact: {stats['oldest_artifact']}")
        if stats["newest_artifact"]:
            console.print(f"Newest artifact: {stats['newest_artifact']}")

        if stats["artifact_counts"]:
            table = Table(title="Cached Artifacts by Type")
            table.add_column("Artifact Type", style="cyan")
            table.add_column("Count", style="magenta")

            for name, count in stats["artifact_counts"].items():
                table.add_row(name, str(count))

            console.print(table)
    except Exception as e:
        console.print(f"[red]Error getting cache information: {e}[/]")


def format_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable format."""
    size = float(size_bytes)  # Convert to float for division
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} GB"  # Fallback return for completeness


@cli.command()
def paths() -> None:
    """Show locations of configuration files and cache."""
    configure_logging()
    from . import artifact_cache, config

    # Get configuration paths
    paths = config.get_app_paths()
    console.print("\n[bold]Application Paths:[/]")
    for name, path in paths.items():
        exists = path.exists()
        status = "[green]exists[/]" if exists else "[yellow]not created yet[/]"
        console.print(f"  [cyan]{name}[/]: {path} ({status})")

    # Get cache path
    cache_dir = artifact_cache.get_cache_dir()
    exists = cache_dir.exists()
    status = "[green]exists[/]" if exists else "[yellow]not created yet[/]"
    console.print(f"  [cyan]artifact_cache[/]: {cache_dir} ({status})")
    console.print()


def main():
    """CLI entry point."""
    import sys
    from pathlib import Path

    # If first argument is a file, treat it as the process command
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        # Check if it's a file and not a command
        arg_path = Path(sys.argv[1])
        if arg_path.exists() and arg_path.is_file():
            # Insert 'process' command before the file argument
            sys.argv.insert(1, "process")

    cli()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    main()

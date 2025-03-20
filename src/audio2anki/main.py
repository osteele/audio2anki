"""Main entry point for audio2anki."""

import locale
import logging
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from .config import edit_config, load_config, reset_config, set_config_value
from .pipeline import PipelineOptions, run_pipeline

# Setup basic logging configuration
console = Console()


def configure_logging(debug: bool = False) -> None:
    """Configure logging based on debug flag."""
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(level=level, format="%(message)s")


def get_system_language() -> str:
    """Get the system language code, falling back to 'english' if not determinable."""
    try:
        # Try to get the system locale
        lang_code = locale.getdefaultlocale()[0]
        if not lang_code:
            return "english"

        # Map common language codes to full names
        language_map: dict[str, str] = {
            "en": "english",
            "zh": "chinese",
            "ja": "japanese",
            "ko": "korean",
            "fr": "french",
            "de": "german",
            "es": "spanish",
            "it": "italian",
            "pt": "portuguese",
            "ru": "russian",
        }

        # Extract primary language code (e.g., "en" from "en_US")
        primary_code = lang_code.split("_")[0].lower()
        return language_map.get(primary_code, "english")
    except Exception:
        return "english"


@click.group()
def cli():
    """Audio2Anki - Generate Anki cards from audio files."""
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--bypass-cache", is_flag=True, help="Skip cache lookup and force reprocessing")
@click.option("--keep-cache", is_flag=True, help="Keep temporary cache directory after processing (for debugging)")
@click.option("--target-language", help="Target language for translation")
@click.option("--source-language", default="chinese", help="Source language for transcription")
def process(
    input_file: str,
    debug: bool = False,
    bypass_cache: bool = False,
    keep_cache: bool = False,
    target_language: str | None = None,
    source_language: str = "chinese",
) -> None:
    """Process an audio/video file and generate Anki flashcards."""
    configure_logging(debug)

    if not target_language:
        target_language = get_system_language()

    options = PipelineOptions(
        target_language=target_language,
        source_language=source_language,
        bypass_cache=bypass_cache,
        keep_cache=keep_cache,
        debug=debug,
    )
    deck_dir = str(run_pipeline(Path(input_file), console, options))

    # Print deck location and instructions
    console.print(f"\n[green]Deck created at:[/] {deck_dir}")

    # Read and render README content
    readme_path = Path(deck_dir) / "README.md"
    if readme_path.exists():
        content = readme_path.read_text(encoding="utf-8")

        # Replace "symbolic link" with platform-specific term
        import platform

        alias_term = (
            "alias"
            if platform.system() == "Darwin"
            else "shortcut"
            if platform.system() == "Windows"
            else "symbolic link"
        )
        content = content.replace("symbolic link", alias_term)

        # Render markdown content
        md = Markdown(content)
        console.print(md)


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


# Cache-related commands have been removed as we now use a temporary directory for each run


@cli.command()
def paths() -> None:
    """Show locations of configuration files."""
    configure_logging()
    from . import config

    paths = config.get_app_paths()
    console.print("\n[bold]Application Paths:[/]")
    for name, path in paths.items():
        if name == "cache_dir":
            continue  # Skip cache_dir since we now use temp directories
        exists = path.exists()
        status = "[green]exists[/]" if exists else "[yellow]not created yet[/]"
        console.print(f"  [cyan]{name}[/]: {path} ({status})")
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

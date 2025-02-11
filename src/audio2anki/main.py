"""Main entry point for audio2anki."""

import locale
import logging
from pathlib import Path

import click
from rich.console import Console

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
        language_map = {
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


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Audio2Anki - Generate Anki cards from audio files."""
    # If no command is specified, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command(name="process", help="Process an audio/video file (default command)")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--debug", is_flag=True, help="Enable debug output")
@click.option("--bypass-cache", is_flag=True, help="Bypass the cache and force recomputation")
@click.option("--clear-cache", is_flag=True, help="Clear the cache before starting")
@click.option(
    "--source-language",
    help="Source language of the audio (defaults to Chinese)",
    default="chinese",
)
@click.option(
    "--target-language",
    help="Target language for translation (defaults to system language or English)",
    default=None,
)
def process_command(
    input_file: str,
    debug: bool = False,
    bypass_cache: bool = False,
    clear_cache: bool = False,
    target_language: str | None = None,
    source_language: str = "chinese",
) -> None:
    """Process an audio/video file and generate Anki flashcards."""
    configure_logging(debug)

    # Initialize cache system
    from . import cache

    if clear_cache:
        cache.clear_cache()
    cache.init_cache(bypass=bypass_cache)

    if target_language is None:
        target_language = get_system_language()

    # Create pipeline options
    options = PipelineOptions(
        debug=debug,
        bypass_cache=bypass_cache,
        clear_cache=clear_cache,
        source_language=source_language,
        target_language=target_language,
    )

    # Run the pipeline
    run_pipeline(Path(input_file), console=console, options=options)


@cli.command()
def paths() -> None:
    """Show locations of configuration and cache files."""
    configure_logging()
    from . import config

    paths = config.get_app_paths()
    console.print("\n[bold]Application Paths:[/]")
    for name, path in paths.items():
        exists = path.exists()
        status = "[green]exists[/]" if exists else "[yellow]not created yet[/]"
        console.print(f"  [cyan]{name}[/]: {path} ({status})")
    console.print()


def main() -> None:
    """CLI entry point."""
    # If no arguments, show help
    import sys

    if len(sys.argv) == 1:
        cli.main(["--help"])
        return

    # If first arg is a file, treat it as the process command
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-") and sys.argv[1] not in ["paths", "process"]:
        # Insert 'process' command before the file argument
        sys.argv.insert(1, "process")

    cli()


if __name__ == "__main__":
    main()

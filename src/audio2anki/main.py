import logging
import os
from dataclasses import dataclass

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
console = Console()

# Configuration defaults
DEFAULT_CONFIG = {
    "clean_files": True,
    "use_cache": True,
    "cache_expiry_days": 7,
    "voice_isolation_provider": "eleven_labs",
    "transcription_provider": "openai_whisper",
}


@dataclass
class PipelineProgress:
    """Manages progress tracking for the pipeline and its stages."""

    progress: Progress  # Rich Progress instance
    pipeline_task: TaskID
    current_stage: TaskID | None = None

    @classmethod
    def create(cls) -> "PipelineProgress":
        """Create a new pipeline progress tracker."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        pipeline_task = progress.add_task("[bold blue]Processing audio file...", total=5)
        return cls(progress=progress, pipeline_task=pipeline_task)

    def __enter__(self) -> "PipelineProgress":
        """Start the progress display when entering the context."""
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the progress display when exiting the context."""
        self.progress.stop()

    def start_stage(self, description: str) -> TaskID:
        """Start a new pipeline stage."""
        self.current_stage = self.progress.add_task(f"  [cyan]{description}...", total=100)
        return self.current_stage

    def update_stage(self, completed: float) -> None:
        """Update the current stage's progress."""
        if self.current_stage is not None:
            self.progress.update(self.current_stage, completed=completed)

    def complete_stage(self) -> None:
        """Mark the current stage as complete and advance the pipeline."""
        if self.current_stage is not None:
            self.progress.update(self.current_stage, completed=100)
            self.progress.advance(self.pipeline_task)
            self.current_stage = None


def load_config() -> dict[str, bool | int | str]:
    """Load configuration from $HOME/.config/audio2anki/config.toml and return a config dictionary."""
    config_path = os.path.join(os.environ.get("HOME", "."), ".config", "audio2anki", "config.toml")
    config = DEFAULT_CONFIG.copy()
    try:
        with open(config_path) as f:
            # For now, we use a simple parsing: ignore comments and simple key = value pairs
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        # A simple conversion: if value is a digit, convert to int, else leave as string or bool
                        if value.lower() in ["true", "false"]:
                            config[key] = value.lower() == "true"
                        elif value.isdigit():
                            config[key] = int(value)
                        else:
                            config[key] = value
    except FileNotFoundError:
        logging.info(f"Config file not found at {config_path}. Using default configuration.")
    return config


def init_cache() -> str:
    """Ensure the cache directory exists and return its path."""
    cache_dir = os.path.join(os.environ.get("HOME", "."), ".cache", "audio2anki")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        logging.info(f"Cache directory created at {cache_dir}")
    else:
        logging.info(f"Cache directory exists at {cache_dir}")
    return cache_dir


def transcode(input_file: str, pipeline: "PipelineProgress") -> str:
    """Transcode an audio/video file to an audio file suitable for processing."""
    pipeline.start_stage("Transcoding audio")
    # Simulate transcoding progress
    pipeline.update_stage(50)  # In real implementation, update based on ffmpeg progress
    # Placeholder: In a real implementation, call ffmpeg to extract audio
    audio_file = input_file + ".audio.mp3"
    pipeline.complete_stage()
    return audio_file


def voice_isolation(audio_file: str, pipeline: "PipelineProgress") -> str:
    """Perform voice isolation using the selected provider."""
    pipeline.start_stage("Isolating voice using Eleven Labs API")
    # Simulate API call progress
    pipeline.update_stage(50)  # In real implementation, update based on API progress
    # Placeholder: In a real implementation, call the Eleven Labs API
    cleaned_audio = audio_file.replace(".audio.mp3", ".cleaned.mp3")
    pipeline.complete_stage()
    return cleaned_audio


def transcribe(audio_file: str, pipeline: "PipelineProgress") -> str:
    """Transcribe the audio file using OpenAI Whisper and produce an SRT file."""
    pipeline.start_stage("Transcribing with OpenAI Whisper")
    # Simulate transcription progress
    pipeline.update_stage(50)  # In real implementation, update based on Whisper progress
    # Placeholder: In a real implementation, call OpenAI Whisper
    srt_file = audio_file.replace(".cleaned.mp3", ".srt")
    pipeline.complete_stage()
    return srt_file


def sentence_selection(transcript: str, pipeline: "PipelineProgress") -> str:
    """Process the transcript to perform sentence selection."""
    pipeline.start_stage("Selecting sentences")
    # Simulate selection progress
    pipeline.update_stage(50)  # In real implementation, update based on actual progress
    # Placeholder: In a real implementation, apply sentence splitting/filtering
    selected_transcript = transcript.replace(".srt", ".selected.txt")
    pipeline.complete_stage()
    return selected_transcript


def generate_deck(processed_data: str, pipeline: "PipelineProgress") -> None:
    """Generate an Anki flashcard deck from the processed data."""
    pipeline.start_stage("Generating Anki deck")
    # Simulate deck generation progress
    pipeline.update_stage(50)  # In real implementation, update based on actual progress
    # Placeholder: In a real implementation, create directory structure with deck.txt, etc.
    pipeline.complete_stage()


def run_pipeline(input_file: str) -> None:
    """Run the entire pipeline for the given input file."""
    with PipelineProgress.create() as pipeline:
        # Transcode input file
        audio_file = transcode(input_file, pipeline)
        # Voice Isolation
        cleaned_audio = voice_isolation(audio_file, pipeline)
        # Transcription
        srt_file = transcribe(cleaned_audio, pipeline)
        # Optional Sentence Selection
        processed_transcript = sentence_selection(srt_file, pipeline)
        # Card Deck Generation
        generate_deck(processed_transcript, pipeline)


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--debug", is_flag=True, help="Enable debug output")
def main(input_file: str, debug: bool) -> None:
    """Process an audio/video file and generate Anki flashcards."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug mode enabled.")

    config = load_config()
    cache_dir = init_cache()

    if debug:
        console.print(f"[blue]Loaded configuration:[/] {config}")
        console.print(f"[blue]Using cache directory:[/] {cache_dir}")

    run_pipeline(input_file)


if __name__ == "__main__":
    main()

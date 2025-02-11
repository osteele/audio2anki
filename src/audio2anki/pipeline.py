"""Audio processing pipeline module."""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

from .transcoder import transcode_audio


@dataclass
class PipelineOptions:
    """Options that control pipeline behavior."""

    bypass_cache: bool = False
    clear_cache: bool = False
    debug: bool = False
    source_language: str = "chinese"
    target_language: str | None = None


@dataclass
class PipelineContext:
    """Holds the state and artifacts produced by each pipeline stage."""

    # Single "primary" artifact used by each next stage
    primary: Path

    # These fields store each artifact for later stages like Anki
    isolated_audio: Path | None = None
    transcription_srt: Path | None = None
    translation_srt: Path | None = None
    pronunciation_srt: Path | None = None

    # Language settings
    source_language: str = "chinese"
    target_language: str | None = None

    @classmethod
    def from_options(cls, input_path: Path, options: PipelineOptions) -> "PipelineContext":
        """Create a new pipeline context from options."""
        return cls(
            primary=input_path,
            source_language=options.source_language,
            target_language=options.target_language,
        )


@dataclass
class PipelineProgress:
    """Manages progress tracking for the pipeline and its stages."""

    progress: Progress
    pipeline_task: TaskID
    console: Console
    stage_tasks: dict[str, TaskID] = field(default_factory=dict)
    current_stage: str | None = None

    @classmethod
    def create(cls, console: Console) -> "PipelineProgress":
        """Create a new pipeline progress tracker."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        pipeline_task = progress.add_task("[bold blue]Creating Anki deck...", total=5)  # 5 stages
        return cls(progress=progress, pipeline_task=pipeline_task, console=console)

    def __enter__(self) -> "PipelineProgress":
        """Start the progress display when entering the context."""
        self.progress.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop the progress display when exiting the context."""
        self.progress.stop()

    def start_stage(self, description: str) -> TaskID:
        """Start a new stage with the given description."""
        if self.current_stage and self.current_stage in self.stage_tasks:
            self.progress.update(self.stage_tasks[self.current_stage], visible=False)

        task_id = self.progress.add_task(f"[cyan]{description}", total=100, visible=True)
        self.stage_tasks[description] = task_id
        self.current_stage = description
        return task_id

    def complete_stage(self) -> None:
        """Mark the current stage as complete and advance the pipeline."""
        if self.current_stage and self.current_stage in self.stage_tasks:
            self.progress.update(self.stage_tasks[self.current_stage], completed=100)
            self.progress.advance(self.pipeline_task)
            self.current_stage = None

    def update_progress(self, completed: float) -> None:
        """Update the current stage's progress."""
        if self.current_stage and self.current_stage in self.stage_tasks:
            self.progress.update(self.stage_tasks[self.current_stage], completed=completed)


def run_pipeline(input_path: Path, console: Console, options: PipelineOptions) -> PipelineContext:
    """Run the audio processing pipeline with minimal stage coordination."""
    if options.clear_cache:
        from . import cache

        cache.clear_cache()
        logging.info("Cache cleared")

    with PipelineProgress.create(console) as progress:
        try:
            # Initialize context with input file and options
            context = PipelineContext.from_options(input_path, options)

            # Run each stage in sequence
            progress.start_stage("Transcoding audio")
            context = transcode(context, progress)
            progress.complete_stage()

            progress.start_stage("Isolating voice with Eleven Labs API")
            context = voice_isolation(context, progress)
            progress.complete_stage()

            progress.start_stage("Transcribing with OpenAI Whisper")
            context = transcribe(context, progress)
            progress.complete_stage()

            progress.start_stage("Translating transcript")
            context = translate(context, progress)
            progress.complete_stage()

            progress.start_stage("Creating Anki deck files")
            context = generate_deck(context, progress)
            progress.complete_stage()

            return context

        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            console.print(f"[red]Error: {str(e)}[/]")
            sys.exit(1)


def transcode(context: PipelineContext, progress: PipelineProgress) -> PipelineContext:
    """Transcode an audio/video file to an audio file suitable for processing."""
    try:
        output_path = transcode_audio(context.primary, progress_callback=progress.update_progress)
        context.primary = Path(output_path)
        return context
    except Exception as e:
        logging.error(f"Transcoding failed: {e}")
        raise


def voice_isolation(context: PipelineContext, progress: PipelineProgress) -> PipelineContext:
    """Isolate voice from background noise."""
    from .voice_isolation import VoiceIsolationError, isolate_voice

    try:
        output_path = isolate_voice(context.primary, progress_callback=progress.update_progress)
        context.isolated_audio = output_path
        context.primary = output_path
        return context
    except VoiceIsolationError as e:
        progress.console.print(f"[red]Voice isolation failed: {e}[/]")
        sys.exit(1)


def transcribe(context: PipelineContext, progress: PipelineProgress) -> PipelineContext:
    """Transcribe the audio file using OpenAI Whisper and produce an SRT file."""
    from .transcribe import transcribe_audio

    input_path = context.primary
    output_path = input_path.parent / f"transcribe_{input_path.stem}.srt"

    # Map full language names to Whisper codes
    language_codes = {
        "chinese": "zh",
        "japanese": "ja",
        "korean": "ko",
        "english": "en",
        "french": "fr",
        "german": "de",
        "spanish": "es",
        "italian": "it",
        "portuguese": "pt",
        "russian": "ru",
    }

    # Convert language name to code for Whisper
    language_code = language_codes.get(context.source_language.lower())

    task_id = progress.start_stage("Transcribing with OpenAI Whisper")
    transcribe_audio(
        audio_file=input_path,
        transcript_path=output_path,
        model="whisper-1",
        progress=progress.progress,
        task_id=task_id,
        language=language_code,
    )

    context.transcription_srt = output_path
    context.primary = output_path
    return context


def translate(context: PipelineContext, progress: PipelineProgress) -> PipelineContext:
    """Translate the SRT file to English and create pinyin if needed."""
    from .translate import translate_srt

    input_path = context.primary
    task_id = progress.start_stage("Translating transcript")

    # Default to English if no target language specified
    target_language = context.target_language or "english"

    translated_file, pronunciation_file = translate_srt(
        input_file=input_path,
        source_language=context.source_language,
        target_language=target_language,
        task_id=task_id,
        progress=progress.progress,
    )

    context.translation_srt = Path(translated_file)
    if pronunciation_file:
        context.pronunciation_srt = Path(pronunciation_file)
    context.primary = context.translation_srt
    return context


def generate_deck(context: PipelineContext, progress: PipelineProgress) -> PipelineContext:
    """Generate an Anki flashcard deck from the processed data."""
    from .anki import process_deck

    # Create deck directory in current working directory
    deck_dir = Path.cwd() / "deck"
    if deck_dir.exists():
        import shutil

        shutil.rmtree(deck_dir)
    deck_dir.mkdir(parents=True, exist_ok=True)

    # Create media directory
    media_dir = deck_dir / "media"
    media_dir.mkdir(exist_ok=True)

    # Process deck
    deck_dir = process_deck(
        context.primary,
        progress,
        input_audio_file=context.isolated_audio,
        transcription_file=context.transcription_srt,
        pronunciation_file=context.pronunciation_srt,
        source_language=context.source_language,
        target_language=context.target_language or "english",
    )
    context.primary = deck_dir
    return context

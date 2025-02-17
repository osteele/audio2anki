"""Audio processing pipeline module."""

import inspect
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar, Union

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

T = TypeVar("T")
Artifact = Any  # Replace with more specific type as needed
PipelineFunction = Callable[..., Artifact]


def produces_artifacts(**artifacts: Any) -> Callable[[PipelineFunction], PipelineFunction]:
    """Decorator that annotates a pipeline function with the artifacts it produces.

    Args:
        **artifacts: Mapping of artifact names to their types.
            e.g. @produces_artifacts(translation=Path, pronunciation=Path | None)

    Returns:
        A decorator function that adds the artifact information to the function.
    """

    def decorator(func: PipelineFunction) -> PipelineFunction:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.produced_artifacts = artifacts  # type: ignore
        return wrapper

    return decorator


@dataclass
class PipelineOptions:
    """Options that control pipeline behavior."""

    bypass_cache: bool = False
    clear_cache: bool = False
    debug: bool = False
    source_language: str = "chinese"
    target_language: str | None = None


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


@dataclass
class PipelineContext:
    """Holds pipeline state and configuration."""

    progress: PipelineProgress
    source_language: str = "chinese"
    target_language: str | None = None

    @classmethod
    def from_options(cls, progress: PipelineProgress, options: PipelineOptions) -> "PipelineContext":
        """Create a new pipeline context from options."""
        return cls(
            progress=progress,
            source_language=options.source_language,
            target_language=options.target_language,
        )


def validate_pipeline(pipeline: list[PipelineFunction], initial_artifacts: dict[str, Any]) -> None:
    """Validate that all required artifacts will be available when the pipeline runs.

    Args:
        pipeline: List of pipeline functions to validate
        initial_artifacts: Dictionary of artifacts available at the start

    Raises:
        ValueError: If any required artifacts are missing
    """
    available_artifacts = set(initial_artifacts.keys())

    for func in pipeline:
        # Get required artifacts from function parameters
        params = inspect.signature(func).parameters
        required_artifacts = {
            name for name, param in params.items() if name != "context" and param.default == param.empty
        }

        # Check if all required artifacts are available
        missing = required_artifacts - available_artifacts
        if missing:
            logging.error(
                f"Function {func.__name__} requires artifacts that won't be available: {missing}. "
                f"Available artifacts will be: {available_artifacts}"
            )
            raise ValueError(
                f"Function {func.__name__} requires artifacts that won't be available: {missing}. "
                f"Available artifacts will be: {available_artifacts}"
            )

        # Add this function's produced artifacts to available artifacts
        if hasattr(func, "produced_artifacts"):
            produced_artifacts = func.produced_artifacts  # type: ignore
            for artifact_name, _artifact_type in produced_artifacts.items():
                # We don't need to check the type, just add the name to available artifacts
                available_artifacts.add(artifact_name)
        else:
            available_artifacts.add(func.__name__)


def run_pipeline(input_file: Path, console: Console, options: PipelineOptions) -> Path:
    """Run the audio processing pipeline.

    Returns:
        Path: The path to the generated deck directory
    """
    if options.clear_cache:
        from . import cache

        cache.clear_cache()
        logging.info("Cache cleared")

    with PipelineProgress.create(console) as progress:
        try:
            # Initialize context
            context = PipelineContext.from_options(progress, options)

            # Define pipeline stages
            pipeline = [transcode, voice_isolation, transcribe, translate, generate_deck]
            initial_artifacts = {"input_path": input_file}

            # Validate pipeline before running
            validate_pipeline(pipeline, initial_artifacts)

            # Run pipeline
            artifacts = initial_artifacts
            for func in pipeline:
                # Start progress tracking for this stage
                progress.start_stage(func.__name__.replace("_", " ").title())

                # Get required arguments from artifacts
                params = inspect.signature(func).parameters
                kwargs = {name: artifacts[name] for name in params if name != "context" and name in artifacts}

                # Run the function
                try:
                    result = func(context=context, **kwargs)
                    if hasattr(func, "produced_artifacts"):
                        # If function has multiple artifacts, unpack them into the artifacts dict
                        if isinstance(result, tuple):
                            for name, value in zip(func.produced_artifacts.keys(), result, strict=False):  # type: ignore
                                artifacts[name] = value
                        else:
                            # Single artifact with a custom name
                            artifacts[next(iter(func.produced_artifacts.keys()))] = result  # type: ignore
                    else:
                        # Default behavior: use function name as artifact key
                        artifacts[func.__name__] = result
                except Exception as e:
                    logging.error(f"{func.__name__} failed: {str(e)}")
                    console.print(f"[red]Error in {func.__name__}: {str(e)}[/]")
                    raise

                progress.complete_stage()

            return artifacts["generate_deck"]

        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            console.print(f"[red]Error: {str(e)}[/]")
            sys.exit(1)


def transcode(context: PipelineContext, input_path: Path) -> Path:
    """Transcode an audio/video file to an audio file suitable for processing."""
    try:
        return transcode_audio(input_path, progress_callback=context.progress.update_progress)
    except Exception as e:
        logging.error(f"Transcoding failed: {e}")
        raise


def voice_isolation(context: PipelineContext, transcode: Path) -> Path:
    """Isolate voice from background noise."""
    from .voice_isolation import VoiceIsolationError, isolate_voice

    try:
        return isolate_voice(transcode, progress_callback=context.progress.update_progress)
    except VoiceIsolationError as e:
        context.progress.console.print(f"[red]Voice isolation failed: {e}[/]")
        sys.exit(1)


def transcribe(context: PipelineContext, voice_isolation: Path) -> Path:
    """Transcribe the audio file using OpenAI Whisper and produce an SRT file."""
    from .transcribe import transcribe_audio

    output_path = voice_isolation.parent / f"transcribe_{voice_isolation.stem}.srt"

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

    language = language_codes.get(context.source_language.lower())
    if not language:
        raise ValueError(f"Unsupported language: {context.source_language}")

    # Get task ID for progress tracking
    task_id = (
        context.progress.stage_tasks.get(context.progress.current_stage) if context.progress.current_stage else None
    )

    transcribe_audio(
        audio_file=voice_isolation,
        transcript_path=output_path,
        model="whisper-1",
        task_id=task_id,
        progress=context.progress.progress,
        language=language,
    )

    return output_path


@produces_artifacts(translation=Path, pronunciation=Union[Path, None])  # noqa: UP007
def translate(context: PipelineContext, transcribe: Path) -> tuple[Path, Path | None]:
    """Translate the SRT file to English and create pinyin if needed.

    Returns:
        tuple[Path, Path | None]: (translation_srt_path, pronunciation_srt_path)
            pronunciation_srt_path may be None for languages that don't need pronunciation
    """
    from .translate import translate_srt

    # Get task ID for progress tracking
    task_id = (
        context.progress.stage_tasks.get(context.progress.current_stage) if context.progress.current_stage else None
    )
    if task_id is None:
        raise RuntimeError("No task ID available for translation stage")

    translation_path, pronunciation_path = translate_srt(
        input_file=transcribe,
        target_language=context.target_language or "english",
        task_id=task_id,
        progress=context.progress.progress,
        source_language=context.source_language,
    )

    return translation_path, pronunciation_path


def generate_deck(
    context: PipelineContext,
    voice_isolation: Path,
    transcribe: Path,
    translation: Path,
    pronunciation: Path | None,
) -> Path:
    """Generate an Anki flashcard deck from the processed data.

    Args:
        context: Pipeline context
        voice_isolation: Path to the voice-isolated audio file
        transcribe: Path to the transcription file
        translation: Path to the translation file
        pronunciation: Path to the pronunciation file (optional)

    Returns:
        Path: Path to the generated Anki deck directory
    """
    from .anki import generate_anki_deck

    task_id = (
        context.progress.stage_tasks.get(context.progress.current_stage) if context.progress.current_stage else None
    )

    if task_id is None:
        raise RuntimeError("No task ID available for deck generation stage")

    return generate_anki_deck(
        input_data=translation,  # translation file contains the main content
        input_audio_file=voice_isolation,
        transcription_file=transcribe,
        pronunciation_file=pronunciation,
        source_language=context.source_language,
        target_language=context.target_language,
        task_id=task_id,
        progress=context.progress,
    )

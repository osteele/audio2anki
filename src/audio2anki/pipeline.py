"""Audio processing pipeline module."""

import inspect
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

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

T = TypeVar("T")
Artifact = Any  # Replace with more specific type as needed


@runtime_checkable
class PipelineFunction(Protocol):
    """Protocol for pipeline functions."""

    __name__: str
    produced_artifacts: dict[str, Any]

    def __call__(self, context: "PipelineContext", **kwargs: Any) -> None: ...


def produces_artifacts(**artifacts: Any) -> Callable[[Callable[..., None]], PipelineFunction]:
    """
    Decorator that annotates a pipeline function with the artifacts it produces.
    Example usage:
        @produces_artifacts(transcribe={"extension": "srt"})
        def transcribe(...): ...
    """

    def decorator(func: Callable[..., None]) -> PipelineFunction:
        # If the user just passes a type (like Path), we can default to "mp3"
        # If they pass dict with extension, we store that.
        new_artifacts = {}
        for name, value in artifacts.items():
            if isinstance(value, dict) and "extension" in value:
                new_artifacts[name] = value
            else:
                # e.g. user just used "Path"
                new_artifacts[name] = {"extension": "mp3"}

        func.produced_artifacts = new_artifacts  # type: ignore
        return func  # type: ignore

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
        pipeline_task = progress.add_task("Processing audio...", total=100)
        return cls(progress=progress, pipeline_task=pipeline_task, console=console)

    def __enter__(self) -> "PipelineProgress":
        """Start the progress display when entering the context."""
        self.progress.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop the progress display when exiting the context."""
        self.progress.stop()

    def start_stage(self, stage_name: str) -> None:
        """Start a new stage with the given description."""
        self.current_stage = stage_name
        self.stage_tasks[stage_name] = self.progress.add_task(f"[cyan]{stage_name}...", total=100)

    def complete_stage(self) -> None:
        """Mark the current stage as complete and advance the pipeline."""
        if self.current_stage and self.current_stage in self.stage_tasks:
            self.progress.update(self.stage_tasks[self.current_stage], completed=100)
            self.progress.update(self.pipeline_task, advance=20)

    def update_progress(self, completed: float) -> None:
        """Update the current stage's progress."""
        if self.current_stage and self.current_stage in self.stage_tasks:
            self.progress.update(self.stage_tasks[self.current_stage], completed=completed)


@dataclass
class PipelineContext:
    """Holds pipeline state and configuration."""

    progress: "PipelineProgress"
    source_language: str = "chinese"
    target_language: str | None = None
    _current_stage: str | None = None
    _current_stage_fn: PipelineFunction | None = None
    _stage_artifacts: dict[str, set[str]] = field(default_factory=dict)

    def for_stage(self, pipeline_fn: PipelineFunction) -> "PipelineContext":
        """Create a stage-specific context."""
        stage_name = pipeline_fn.__name__
        # Store artifact keys if function is decorated
        produced_artifacts = getattr(pipeline_fn, "produced_artifacts", None)
        if produced_artifacts is not None:
            self._stage_artifacts[stage_name] = set(produced_artifacts.keys())
        else:
            # Function produces single artifact keyed by function name
            self._stage_artifacts[stage_name] = {stage_name}

        # Start progress tracking for this stage
        self.progress.start_stage(stage_name)

        # Create new context with stage name and function set
        return replace(self, _current_stage=stage_name, _current_stage_fn=pipeline_fn)

    def get_artifact_path(self, artifact_name: str | None = None) -> Path:
        """Get path for an artifact within the current stage.

        Args:
            artifact_name: Name of the artifact. If None, uses the stage name.
        """
        if not self._current_stage or not self._current_stage_fn:
            raise ValueError("No current pipeline stage")

        valid_artifacts = self._stage_artifacts.get(self._current_stage, set(self._current_stage))

        # If no artifact_name provided, use stage name (for single-artifact stages)
        if not artifact_name:
            if len(valid_artifacts) != 1:
                msg = f"Expected exactly one artifact for stage '{self._current_stage}'"
                msg += f", but got {len(valid_artifacts)}"
                raise ValueError(msg)
            artifact_name = next(iter(valid_artifacts))

        # Validate artifact belongs to current stage
        if artifact_name not in valid_artifacts:
            raise ValueError(f"Invalid artifact '{artifact_name}' for stage '{self._current_stage}'")

        # Get extension from the function's produced_artifacts
        extension = "mp3"  # default
        produced_artifacts = getattr(self._current_stage_fn, "produced_artifacts", None)
        if produced_artifacts and artifact_name in produced_artifacts:
            extension = produced_artifacts[artifact_name].get("extension", "mp3")

        # Use cache to get the path
        from . import cache

        return Path(cache.get_cache_path(artifact_name, "temp", f".{extension}"))

    @property
    def stage_task_id(self) -> TaskID | None:
        """Get the task ID for the current stage."""
        if not self._current_stage:
            return None
        return self.progress.stage_tasks.get(self._current_stage)


def validate_pipeline(pipeline: Sequence[PipelineFunction], initial_artifacts: dict[str, Any]) -> None:
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
            context = PipelineContext(
                progress=progress,
                source_language=options.source_language,
                target_language=options.target_language,
            )

            # Define pipeline stages
            pipeline = [transcode, voice_isolation, transcribe, translate, generate_deck]
            initial_artifacts = {"input_path": input_file}

            # Validate pipeline before running
            validate_pipeline(pipeline, initial_artifacts)

            # Run pipeline
            artifacts = initial_artifacts
            for func in pipeline:
                # Create stage-specific context
                context = context.for_stage(func)

                # Get required arguments from artifacts
                params = inspect.signature(func).parameters
                kwargs = {name: artifacts[name] for name in params if name != "context" and name in artifacts}

                # Run the function
                try:
                    func(context=context, **kwargs)
                except Exception as e:
                    logging.error(f"{func.__name__} failed: {str(e)}")
                    console.print(f"[red]Error in {func.__name__}: {str(e)}[/]")
                    raise

                # Update artifacts with produced artifacts from this stage
                if hasattr(func, "produced_artifacts"):
                    for artifact_name in func.produced_artifacts:
                        artifacts[artifact_name] = context.get_artifact_path(artifact_name)

                progress.complete_stage()

            return context.get_artifact_path()

        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            console.print(f"[red]Error: {str(e)}[/]")
            sys.exit(1)


@produces_artifacts(transcode={"extension": "mp3"})
def transcode(context: PipelineContext, input_path: Path) -> None:
    """Transcode an audio/video file to an audio file suitable for processing."""
    from .transcoder import transcode_audio

    try:
        output_path = context.get_artifact_path()
        transcode_audio(input_path, output_path, progress_callback=context.progress.update_progress)
    except Exception as e:
        logging.error(f"Transcoding failed: {e}")
        raise


@produces_artifacts(voice_isolation={"extension": "mp3"})
def voice_isolation(context: PipelineContext, transcode: Path) -> None:
    """Isolate voice from background noise."""
    from .voice_isolation import VoiceIsolationError, isolate_voice

    try:
        output_path = context.get_artifact_path()
        isolate_voice(transcode, output_path, progress_callback=context.progress.update_progress)
    except VoiceIsolationError as e:
        context.progress.console.print(f"[red]Voice isolation failed: {e}[/]")
        sys.exit(1)


@produces_artifacts(transcribe={"extension": "srt"})
def transcribe(context: PipelineContext, voice_isolation: Path) -> None:
    """Transcribe the audio file using OpenAI Whisper and produce an SRT file."""
    from .transcribe import transcribe_audio

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

    transcribe_audio(
        audio_file=voice_isolation,
        transcript_path=context.get_artifact_path(),
        model="whisper-1",
        task_id=context.stage_task_id,
        progress=context.progress.progress,
        language=language,
    )


@produces_artifacts(translation={"extension": "srt"}, pronunciation={"extension": "srt"})
def translate(context: PipelineContext, transcribe: Path) -> None:
    """Translate the SRT file to English and create pinyin if needed."""
    from .translate import translate_srt

    translation_path = context.get_artifact_path("translation")
    pronunciation_path = context.get_artifact_path("pronunciation")

    translate_srt(
        input_file=transcribe,
        target_language=context.target_language or "english",
        task_id=cast(TaskID, context.stage_task_id),
        progress=context.progress.progress,
        source_language=context.source_language,
        translation_output=translation_path,
        pronunciation_output=pronunciation_path,
    )


def generate_deck(
    context: PipelineContext,
    voice_isolation: Path,
    transcribe: Path,
    translation: Path,
    pronunciation: Path | None,
) -> None:
    """Generate an Anki flashcard deck from the processed data."""
    from .anki import generate_anki_deck

    generate_anki_deck(
        input_data=translation,  # translation file contains the main content
        input_audio_file=voice_isolation,
        transcription_file=transcribe,
        pronunciation_file=pronunciation,
        source_language=context.source_language,
        target_language=context.target_language,
        task_id=context.stage_task_id,
        progress=context.progress,
        output_path="deck",
    )

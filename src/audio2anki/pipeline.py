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


@runtime_checkable
class PipelineFunction(Protocol):
    """Protocol for pipeline functions."""

    __name__: str
    produced_artifacts: dict[str, dict[str, Any]]

    def __call__(self, context: "PipelineContext", **kwargs: Any) -> None: ...


def produces_artifacts(**artifacts: dict[str, Any]) -> Callable[[Callable[..., None]], PipelineFunction]:
    """
    Decorator that annotates a pipeline function with the artifacts it produces.
    Example usage:
        @produces_artifacts(transcript={"extension": "srt"})
        def transcribe(...): ...
    """

    def decorator(func: Callable[..., None]) -> PipelineFunction:
        # Store the artifact definitions
        func.produced_artifacts = artifacts  # type: ignore
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
    current_stage: str | None = None
    stage_tasks: dict[str, TaskID] = field(default_factory=dict)

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
        """Start tracking progress for a new stage."""
        self.current_stage = stage_name
        self.stage_tasks[stage_name] = self.progress.add_task(f"{stage_name}...", total=100, start=False)

    def complete_stage(self) -> None:
        """Mark the current stage as complete."""
        if self.current_stage and self.current_stage in self.stage_tasks:
            self.progress.update(self.stage_tasks[self.current_stage], completed=100)

    def update_progress(self, percent: float) -> None:
        """Update progress for the current stage."""
        if self.current_stage and self.current_stage in self.stage_tasks:
            self.progress.update(self.stage_tasks[self.current_stage], completed=percent)


@dataclass
class PipelineContext:
    """Holds pipeline state and configuration."""

    progress: PipelineProgress
    source_language: str = "chinese"
    target_language: str | None = None
    _current_fn: PipelineFunction | None = None
    _input_file: Path | None = None
    _stage_inputs: dict[str, Path] = field(default_factory=dict)
    _artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Convert a filename to a safe format for use in cache filenames.

        Preserves Unicode characters including diacritics, since modern filesystems
        (APFS, ext4, NTFS) support them. Only replaces characters that are actually
        unsafe in filenames.
        """
        # Remove file extension
        name = Path(filename).stem

        # Replace unsafe characters with underscore
        # Unsafe: '/', '\0', '\\', ':', '*', '?', '"', '<', '>', '|'
        import re

        name = re.sub(r'[/\0\\:*?"<>|]', "_", name)

        # Replace spaces with single underscore
        name = re.sub(r"\s+", "_", name)

        # Remove leading/trailing underscores and collapse multiple underscores
        name = re.sub(r"^_+|_+$", "", name)
        name = re.sub(r"_+", "_", name)

        # Limit length while preserving the end of the name
        if len(name) > 32:
            name = name[:29] + "..." + name[-3:]

        return name

    def set_input_file(self, input_file: Path) -> None:
        """Set the input file."""
        self._input_file = input_file
        self._stage_inputs["input_path"] = input_file

    def update_stage_input(self, artifact_name: str, input_path: Path) -> None:
        """Update the input file for an artifact."""
        self._stage_inputs[artifact_name] = input_path

    def get_input_hash(self, artifact_name: str) -> str:
        """Get hash of input file for an artifact."""
        import hashlib

        input_path = self._stage_inputs.get(artifact_name)
        if not input_path:
            raise ValueError(f"No input file set for artifact {artifact_name}")
        with open(input_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        return file_hash[:8]  # Use first 8 characters

    def for_stage(self, pipeline_fn: PipelineFunction) -> "PipelineContext":
        """Create a stage-specific context."""
        # Get artifact definitions from the function
        produced_artifacts = getattr(pipeline_fn, "produced_artifacts", None)
        if produced_artifacts is None:
            # Function produces single artifact named after the function
            produced_artifacts = {pipeline_fn.__name__: {"extension": "mp3"}}

        # Update artifacts dictionary
        self._artifacts.update(produced_artifacts)

        # Start progress tracking
        self.progress.start_stage(pipeline_fn.__name__)

        # Create new context with function set
        return replace(self, _current_fn=pipeline_fn)

    def get_artifact_path(self, artifact_name: str | None = None) -> Path:
        """Get path for an artifact.

        Args:
            artifact_name: Name of the artifact. If None and the current function
                         produces only one artifact, uses that artifact's name.
        """
        if not self._current_fn:
            raise ValueError("No current pipeline function")

        if not self._input_file:
            raise ValueError("Input file not set")

        # Get artifact definitions for current function
        produced_artifacts = getattr(self._current_fn, "produced_artifacts", None)
        if produced_artifacts is None:
            produced_artifacts = {self._current_fn.__name__: {"extension": "mp3"}}

        # If no artifact_name provided, use the only artifact if there's just one
        if not artifact_name:
            if len(produced_artifacts) != 1:
                msg = f"Must specify artifact name for function '{self._current_fn.__name__}'"
                msg += f" which produces multiple artifacts: {list(produced_artifacts.keys())}"
                raise ValueError(msg)
            artifact_name = next(iter(produced_artifacts))

        # Validate artifact belongs to current function
        if artifact_name not in produced_artifacts:
            raise ValueError(f"Invalid artifact '{artifact_name}' for function '{self._current_fn.__name__}'")

        # Get extension from the artifact definition
        extension = produced_artifacts[artifact_name].get("extension", "mp3")

        # Use cache to get the path
        from . import cache

        # Create cache key with format: sanitized_name_artifact_name
        sanitized_name = self.sanitize_filename(self._input_file.name)
        cache_key = f"{sanitized_name}_{artifact_name}"
        input_hash = self.get_input_hash(artifact_name)
        return Path(cache.get_cache_path(cache_key, input_hash, f".{extension}"))

    def retrieve_from_cache(self, artifact_name: str, input_path: Path) -> bool:
        """Check if an artifact is in cache."""
        if not self._input_file:
            raise ValueError("Input file not set")

        from . import cache

        sanitized_name = self.sanitize_filename(self._input_file.name)
        cache_key = f"{sanitized_name}_{artifact_name}"
        extension = self._artifacts[artifact_name].get("extension", "mp3")
        return cache.cache_retrieve(cache_key, input_path, f".{extension}")

    def store_in_cache(self, artifact_name: str, output_path: Path, input_path: Path) -> None:
        """Store an artifact in cache."""
        if not self._input_file:
            raise ValueError("Input file not set")

        from . import cache

        sanitized_name = self.sanitize_filename(self._input_file.name)
        cache_key = f"{sanitized_name}_{artifact_name}"
        extension = self._artifacts[artifact_name].get("extension", "mp3")
        with open(output_path, "rb") as f:
            cache.cache_store(cache_key, input_path, f.read(), f".{extension}")

    @property
    def stage_task_id(self) -> TaskID | None:
        """Get the task ID for the current stage."""
        if not self._current_fn:
            return None
        return self.progress.stage_tasks.get(self._current_fn.__name__)


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
            context.set_input_file(input_file)

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
                        artifact_path = context.get_artifact_path(artifact_name)
                        artifacts[artifact_name] = artifact_path
                        # Also store using the stage name as a key
                        artifacts[func.__name__] = artifact_path

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

    # Update input tracking for cache
    context.update_stage_input("transcode", input_path)

    try:
        output_path = context.get_artifact_path()
        transcode_audio(input_path, output_path, progress_callback=context.progress.update_progress)

        # Update input for next stage
        context.update_stage_input("isolated_voice", output_path)
    except Exception as e:
        logging.error(f"Transcoding failed: {e}")
        raise RuntimeError("transcode failed") from e


@produces_artifacts(voice_isolation={"extension": "mp3"})
def voice_isolation(context: PipelineContext, transcode: Path) -> None:
    """Isolate voice from background noise."""
    from .voice_isolation import isolate_voice

    try:
        output_path = context.get_artifact_path()
        isolate_voice(context, transcode)
    except Exception as e:
        logging.error(f"Voice isolation failed: {e}")
        raise RuntimeError("voice isolation failed") from e

    context.update_stage_input("transcribe", output_path)


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

    try:
        output_path = context.get_artifact_path()
        transcribe_audio(
            audio_file=voice_isolation,
            transcript_path=output_path,
            model="whisper-1",
            task_id=context.stage_task_id,
            progress=context.progress.progress,
            language=language,
        )
    except Exception as e:
        print("Error in transcribe:", str(e), file=sys.stderr)
        raise RuntimeError("transcribe failed") from e

    context.update_stage_input("translate", output_path)


@produces_artifacts(translation={"extension": "srt"}, pronunciation={"extension": "srt"})
def translate(context: PipelineContext, transcribe: Path) -> None:
    """Translate the SRT file to English and create pinyin if needed."""
    from .translate import translate_srt

    translation_path = context.get_artifact_path("translation")
    pronunciation_path = context.get_artifact_path("pronunciation")

    try:
        translate_srt(
            input_file=transcribe,
            target_language=context.target_language or "english",
            task_id=cast(TaskID, context.stage_task_id),
            progress=context.progress.progress,
            source_language=context.source_language,
            translation_output=translation_path,
            pronunciation_output=pronunciation_path,
        )
    except Exception as e:
        print("Translation failed:", str(e), file=sys.stderr)
        raise RuntimeError("translation failed") from e

    context.update_stage_input("generate_deck", translation_path)


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

"""Audio processing pipeline module."""

import inspect
import logging
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
    TimeElapsedColumn,
)

T = TypeVar("T")


@runtime_checkable
class PipelineFunction(Protocol):
    """Protocol for pipeline functions."""

    __name__: str
    produced_artifacts: dict[str, dict[str, Any]]

    def __call__(self, context: "PipelineContext", **kwargs: Any) -> None: ...


def pipeline_function(**artifacts: dict[str, Any]) -> Callable[[Callable[..., None]], PipelineFunction]:
    """
    Decorator that annotates a pipeline function with the artifacts it produces.
    Example usage:
        @pipeline_function(transcript={"extension": "srt"})
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
    keep_cache: bool = False
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
            TimeElapsedColumn(),
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
        # Only create a new task if one doesn't already exist for this stage
        if stage_name not in self.stage_tasks:
            self.current_stage = stage_name
            # Create task with start=False to ensure proper timing
            task_id = self.progress.add_task(f"{stage_name}...", total=100, start=False)
            self.stage_tasks[stage_name] = task_id
            # Start the task explicitly
            self.progress.start_task(task_id)
        else:
            # Just set the current stage if the task already exists
            self.current_stage = stage_name
            # Restart the task to ensure proper timing
            task_id = self.stage_tasks[stage_name]
            self.progress.reset(task_id)
            self.progress.start_task(task_id)

    def complete_stage(self) -> None:
        """Mark the current stage as complete."""
        if self.current_stage and self.current_stage in self.stage_tasks:
            task_id = self.stage_tasks[self.current_stage]
            # Set progress to 100% and mark as completed to stop the spinner
            self.progress.update(task_id, completed=100, refresh=True)
            # Stop the task to prevent further updates
            self.progress.stop_task(task_id)
            # Ensure the task is marked as completed
            self.progress.update(task_id, completed=100, refresh=True)

    def update_progress(self, percent: float) -> None:
        """Update progress for the current stage."""
        if self.current_stage and self.current_stage in self.stage_tasks:
            self.progress.update(self.stage_tasks[self.current_stage], completed=percent, refresh=True)


PipelineFunctionType = PipelineFunction | Callable[..., None]


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

    def set_input_file(self, input_file: Path) -> None:
        """Set the input file."""
        self._input_file = input_file
        self._stage_inputs["input_path"] = input_file

    def update_stage_input(self, artifact_name: str, input_path: Path) -> None:
        """Update the input file for an artifact."""
        self._stage_inputs[artifact_name] = input_path

    def for_stage(self, pipeline_fn: PipelineFunctionType) -> "PipelineContext":
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

    def get_artifact_path(self, artifact_name: str = "") -> Path:
        """
        Get the path to where an artifact should be stored.

        Args:
            artifact_name: The name of the artifact to get the path for. If not provided and the function
                          only produces one artifact, that one is used.

        Returns:
            The path to the artifact file.
        """
        if not self._current_fn:
            raise ValueError("No current pipeline function")

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

        # Get the cache path using just the artifact name
        return cache.get_artifact_path(artifact_name, extension)

    def retrieve_from_cache(self, artifact_name: str) -> Path | None:
        """
        Check if an artifact exists in the temp directory and return its path if found.

        Args:
            artifact_name: The name of the artifact to retrieve

        Returns:
            Path to the cached artifact if found, None otherwise
        """
        # Skip cache for terminal artifacts
        if self._artifacts[artifact_name].get("terminal", False):
            return None

        from . import cache

        # Get extension from the artifact definition
        extension = self._artifacts[artifact_name].get("extension", "mp3")

        # Get the cache path
        cache_path = cache.get_artifact_path(artifact_name, extension)

        # Check if the file exists
        if cache_path.exists():
            return cache_path

        return None

    def store_in_cache(self, artifact_name: str, output_path: Path, input_path: Path | None = None) -> None:
        """
        Store an artifact in the cache.

        Args:
            artifact_name: The name of the artifact to store
            output_path: Path to the artifact file to store
            input_path: Unused, kept for API compatibility
        """
        # Skip cache for terminal artifacts
        if self._artifacts[artifact_name].get("terminal", False):
            return

        # Make sure the output file exists before trying to cache it
        if not output_path.exists():
            logging.warning(f"Cannot cache non-existent file: {output_path}")
            return

        from . import cache

        # Get extension from the artifact definition
        extension = self._artifacts[artifact_name].get("extension", "mp3")

        # If the output path is already in the cache directory, no need to store again
        cache_path = cache.get_artifact_path(artifact_name, extension)
        if output_path.samefile(cache_path):
            return

        # Read and store the file data
        with open(output_path, "rb") as f:
            cache.store_artifact(artifact_name, f.read(), extension)

    @property
    def stage_task_id(self) -> TaskID | None:
        """Get the task ID for the current stage."""
        if not self._current_fn:
            return None
        return self.progress.stage_tasks.get(self._current_fn.__name__)


def validate_pipeline(pipeline: Sequence[PipelineFunctionType], initial_artifacts: dict[str, Any]) -> None:
    """Validate that all required artifacts will be available when the pipeline runs."""
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
        produced_artifacts = getattr(func, "produced_artifacts", {func.__name__: {"extension": "mp3"}})
        for artifact_name, _artifact_type in produced_artifacts.items():
            available_artifacts.add(artifact_name)


@dataclass
class PipelineRunner:
    """Manages the execution of a pipeline including caching, artifact tracking, and error handling."""

    context: PipelineContext
    options: PipelineOptions
    console: Console
    artifacts: dict[str, Any]
    pipeline: list[PipelineFunctionType]

    @classmethod
    def create(cls, input_file: Path, console: Console, options: PipelineOptions) -> "PipelineRunner":
        """Create a new pipeline runner with initialized context."""
        # Initialize context
        progress = PipelineProgress.create(console)
        context = PipelineContext(
            progress=progress,
            source_language=options.source_language,
            target_language=options.target_language,
        )
        context.set_input_file(input_file)

        # Define pipeline stages
        pipeline = [transcode, voice_isolation, transcribe, translate, generate_deck]
        initial_artifacts = {"input_path": input_file}

        return cls(
            context=context,
            options=options,
            console=console,
            artifacts=initial_artifacts,
            pipeline=pipeline,
        )

    def should_use_cache(self, func: PipelineFunctionType) -> bool:
        """Determine if caching should be used for this function."""
        if self.options.bypass_cache:
            return False

        # Check if this is a terminal stage that should bypass cache
        if isinstance(func, PipelineFunction) and hasattr(func, "produced_artifacts"):
            for artifact_info in func.produced_artifacts.values():
                if artifact_info.get("terminal", False):
                    return False

        return True

    def get_cached_artifacts(self, func: PipelineFunctionType) -> tuple[bool, dict[str, Path]]:
        """
        Try to retrieve all artifacts for this function from cache.

        Returns:
            Tuple of (cache_hit, artifact_paths)
        """
        artifact_paths: dict[str, Path] = {}

        if not isinstance(func, PipelineFunction) or not hasattr(func, "produced_artifacts"):
            return False, {}

        cache_hit = True  # Assume cache hit until we find a miss
        stage_context = self.context.for_stage(func)

        for artifact_name in func.produced_artifacts:
            logging.debug(f"Checking cache for {artifact_name}")
            # Try to retrieve from cache
            cached_path = stage_context.retrieve_from_cache(artifact_name)
            if cached_path is None:
                logging.debug(f"Cache miss for {artifact_name}")
                cache_hit = False
                break
            else:
                logging.debug(f"Cache hit for {artifact_name} at {cached_path}")
                artifact_paths[artifact_name] = cached_path

        return cache_hit, artifact_paths

    def store_artifacts_in_cache(self, func: PipelineFunctionType, context: PipelineContext) -> None:
        """Store all artifacts produced by this function in the cache."""
        if not isinstance(func, PipelineFunction) or not hasattr(func, "produced_artifacts"):
            return

        for artifact_name in func.produced_artifacts:
            artifact_path = context.get_artifact_path(artifact_name)
            if artifact_path.exists():
                context.store_in_cache(artifact_name, artifact_path, artifact_path)

    def update_artifacts(self, func: PipelineFunctionType, artifact_paths: dict[str, Path]) -> None:
        """Update the artifacts dictionary with new paths."""
        produced_artifacts = getattr(func, "produced_artifacts", {}) if isinstance(func, PipelineFunction) else {}

        for artifact_name, path in artifact_paths.items():
            self.artifacts[artifact_name] = path
            # Also store using the stage name as a key if this is the primary artifact
            if len(produced_artifacts) == 1 or artifact_name == func.__name__:
                self.artifacts[func.__name__] = path

        # Special handling for terminal artifacts like 'deck'
        if func.__name__ == "generate_deck" and "deck" in artifact_paths:
            self.artifacts["deck"] = artifact_paths["deck"]

    def get_function_kwargs(self, func: PipelineFunctionType) -> dict[str, Any]:
        """Get the required arguments for this function from artifacts."""
        params = inspect.signature(func).parameters
        return {name: self.artifacts[name] for name in params if name != "context" and name in self.artifacts}

    def update_input_tracking(
        self, func: PipelineFunctionType, context: PipelineContext, kwargs: dict[str, Any]
    ) -> None:
        """Set up input tracking for all artifacts this function produces."""
        if not isinstance(func, PipelineFunction) or not hasattr(func, "produced_artifacts") or not kwargs:
            return

        # Get all input paths that aren't the context
        input_paths = {name: path for name, path in kwargs.items() if name != "context" and isinstance(path, Path)}

        # For each artifact, set up all inputs
        for artifact_name in func.produced_artifacts.keys():
            for _name, input_path in input_paths.items():
                context.update_stage_input(artifact_name, input_path)

    def execute_stage(self, func: PipelineFunctionType) -> None:
        """Execute a single pipeline stage with caching."""
        # Create stage-specific context
        stage_context = self.context.for_stage(func)

        # Get required arguments from artifacts
        kwargs = self.get_function_kwargs(func)

        # Set up input tracking for all artifacts this function produces
        self.update_input_tracking(func, stage_context, kwargs)

        # Check if we should use cache
        use_cache = self.should_use_cache(func)

        # Try to get cached artifacts if appropriate
        cache_hit = False
        artifact_paths: dict[str, Path] = {}

        if use_cache:
            cache_hit, artifact_paths = self.get_cached_artifacts(func)

        # Run the function if needed
        if not use_cache or not cache_hit:
            logging.debug(f"Running {func.__name__} (use_cache={use_cache}, cache_hit={cache_hit})")
            func(context=stage_context, **kwargs)

            # Store results in cache after running
            if use_cache:
                self.store_artifacts_in_cache(func, stage_context)

            # Update artifacts with paths from generated files
            if isinstance(func, PipelineFunction) and hasattr(func, "produced_artifacts"):
                generated_paths: dict[str, Path] = {}
                for artifact_name in func.produced_artifacts:
                    artifact_path = stage_context.get_artifact_path(artifact_name)
                    generated_paths[artifact_name] = artifact_path
                self.update_artifacts(func, generated_paths)
        else:
            self.console.print(f"[green]Using cached result for {func.__name__}[/]")
            # Update artifacts with paths from cache
            self.update_artifacts(func, artifact_paths)

        self.context.progress.complete_stage()

    def run(self) -> Path:
        """Run the entire pipeline and return the final artifact path."""
        try:
            # Validate pipeline before running
            validate_pipeline(self.pipeline, self.artifacts)

            # Run each stage
            for func in self.pipeline:
                try:
                    self.execute_stage(func)
                except Exception as e:
                    # Classify error type
                    error_type = "SYSTEM_ERROR"
                    if isinstance(e, ConnectionError | TimeoutError):
                        error_type = "SERVICE_ERROR"
                    elif isinstance(e, ValueError):
                        error_type = "VALIDATION_ERROR"

                    # Enhanced logging with context
                    logging.error(f"{error_type} in {func.__name__}: {str(e)}", exc_info=True)
                    self.console.print(f"[red]Error in {func.__name__} ({error_type}): {str(e)}[/]")
                    raise

            # Get the deck path from the cache
            from . import cache

            deck_path = cache.get_cache().deck_path

            if deck_path and deck_path.exists():
                # Print success message with the actual deck path
                self.console.print(f"\nDeck created at: {deck_path}")
                return deck_path
            else:
                raise ValueError("Pipeline completed but no deck artifact was produced")

        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            self.console.print(f"[red]Error: {str(e)}[/]")
            raise RuntimeError(f"Pipeline failed: {str(e)}") from e


def run_pipeline(input_file: Path, console: Console, options: PipelineOptions) -> Path:
    """Run the audio processing pipeline.

    Returns:
        Path: The path to the generated deck directory
    """
    from . import cache

    # Initialize a new temporary cache with keep_files option
    cache.init_cache(keep_files=options.keep_cache)
    logging.info("Initialized temporary cache")

    try:
        with PipelineProgress.create(console) as progress:
            # Create pipeline runner
            runner = PipelineRunner.create(input_file, console, options)
            runner.context.progress = progress

            # Run the pipeline
            result = runner.run()
            return result
    finally:
        # Clean up cache unless keep_cache is True
        if not options.keep_cache:
            cache.cleanup_cache()


@pipeline_function(transcode={"extension": "mp3"})
def transcode(context: PipelineContext, input_path: Path) -> None:
    """Transcode an audio/video file to an audio file suitable for processing."""
    from .transcoder import transcode_audio

    output_path = context.get_artifact_path()
    transcode_audio(input_path, output_path, progress_callback=context.progress.update_progress)


@pipeline_function(voice_isolation={"extension": "mp3"})
def voice_isolation(context: PipelineContext, transcode: Path) -> None:
    """Isolate voice from background noise."""
    from .voice_isolation import isolate_voice

    # Get the output path from context
    output_path = context.get_artifact_path()

    # Use context's progress update method directly
    isolate_voice(transcode, output_path, progress_callback=context.progress.update_progress)


@pipeline_function(transcribe={"extension": "srt"})
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

    language_code = language_codes.get(context.source_language.lower())

    # Get the output path from context
    output_path = context.get_artifact_path()

    # Call transcribe_audio with the output path
    transcribe_audio(
        audio_file=voice_isolation,
        transcript_path=output_path,
        model="whisper-1",
        task_id=context.stage_task_id,
        progress=context.progress.progress,
        language=language_code,
    )


@pipeline_function(translation={"extension": "srt"}, pronunciation={"extension": "srt"})
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


@pipeline_function(deck={"extension": "directory", "terminal": True})
def generate_deck(
    context: PipelineContext,
    voice_isolation: Path,
    transcribe: Path,
    translation: Path,
    pronunciation: Path | None,
) -> None:
    """Generate an Anki flashcard deck from the processed data."""
    from .anki import generate_anki_deck

    # For the output deck, use the current working directory directly
    # The create_anki_deck function will append "/deck" to this path
    output_path = Path.cwd()

    # Generate the Anki deck
    deck_dir = generate_anki_deck(
        input_data=translation,  # translation file contains the main content
        input_audio_file=voice_isolation,
        transcription_file=transcribe,
        pronunciation_file=pronunciation,
        source_language=context.source_language,
        target_language=context.target_language,
        task_id=context.stage_task_id,
        progress=context.progress,
        output_path=output_path,
    )

    # Store the actual deck path in the artifacts dictionary for the PipelineRunner to use
    from .cache import get_cache

    get_cache().deck_path = deck_dir

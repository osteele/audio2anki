"""Audio processing pipeline module."""

import inspect
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

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

from audio2anki.transcribe import get_transcription_version
from audio2anki.translate import get_translation_version
from audio2anki.voice_isolation import get_voice_isolation_version

from .transcoder import get_transcode_version
from .translate import TranslationProvider
from .types import LanguageCode

T = TypeVar("T")

logger = logging.getLogger(__name__)


@runtime_checkable
class PipelineFunction(Protocol):
    """Protocol for pipeline functions."""

    __name__: str
    produced_artifacts: dict[str, dict[str, Any]]

    def __call__(self, context: "PipelineContext", **kwargs: Any) -> None: ...


VersionType = int | str | Callable[..., int | str]


def resolve_version(version: VersionType, context: "PipelineContext") -> int | str:
    """
    Resolve a version value that can be an integer, string, or function.

    Args:
        version: The version value to resolve
        context: The pipeline context for function resolution

    Returns:
        The resolved version value (int or str)

    Raises:
        ValueError: If the version function requires arguments that aren't available in the context
    """
    if isinstance(version, int | str):
        return version

    # If it's a function, call it with context attributes
    if callable(version):
        # Get the function's parameter names
        sig = inspect.signature(version)
        params = sig.parameters

        # Build kwargs from context attributes
        kwargs: dict[str, Any] = {}
        for param_name in params:
            if param_name == "context":
                kwargs[param_name] = context
            elif hasattr(context, param_name):
                kwargs[param_name] = getattr(context, param_name)
            else:
                raise ValueError(
                    f"Version function {version.__name__} requires parameter '{param_name}' "
                    f"which is not available in the pipeline context"
                )

        return version(**kwargs)

    raise ValueError(f"Invalid version type: {type(version)}")


def pipeline_function(**artifacts: dict[str, Any]) -> Callable[[Callable[..., None]], PipelineFunction]:
    """
    Decorator that annotates a pipeline function with the artifacts it produces.

    Example usage:
        @pipeline_function(transcript={"extension": "srt"})
        def transcribe(...): ...

        @pipeline_function(translation={"extension": "srt", "cache": True, "version": 2})
        def translate(...): ...

        @pipeline_function(translation={"extension": "srt", "cache": True, "version": get_translation_version})
        def translate(...): ...

    Args:
        **artifacts: Dictionary of artifact definitions. Each artifact can have these properties:
            - extension: File extension for the artifact (required)
            - cache: Whether to cache the artifact output (default: False)
            - version: Version number/string/function for cache invalidation (default: 1)
            - terminal: Whether this is a terminal artifact that shouldn't be cached (default: False)
    """

    def decorator(func: Callable[..., None]) -> PipelineFunction:
        # Store the artifact definitions
        func.produced_artifacts = artifacts  # type: ignore

        # Set defaults for caching properties
        for _artifact_name, artifact_def in artifacts.items():
            if "cache" not in artifact_def:
                artifact_def["cache"] = False
            if "version" not in artifact_def:
                artifact_def["version"] = 1

        return func  # type: ignore

    return decorator


@dataclass
class PipelineOptions:
    """Options that control pipeline behavior."""

    debug: bool = False
    source_language: LanguageCode | None = None
    target_language: LanguageCode | None = None
    output_folder: Path | None = None
    skip_voice_isolation: bool = False
    translation_provider: TranslationProvider = TranslationProvider.OPENAI

    # Caching options
    use_artifact_cache: bool = True
    skip_cache_cleanup: bool = False


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
    source_language: LanguageCode | None = None
    target_language: LanguageCode | None = None
    output_folder: Path | None = None
    translation_provider: TranslationProvider = TranslationProvider.OPENAI
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
        artifact_path = cache.get_artifact_path(artifact_name, extension)

        # Log the artifact path at debug level
        logger.debug(f"{artifact_name} artifact will be stored at {artifact_path}")

        return artifact_path

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
            logger.debug(f"Found cached {artifact_name} at {cache_path}")
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
            stored_path = cache.store_artifact(artifact_name, f.read(), extension)
            logger.debug(f"Storing {artifact_name} in cache at {stored_path}")

    @property
    def stage_task_id(self) -> TaskID:
        """Get the task ID for the current stage."""
        if not self._current_fn:
            raise ValueError("No current pipeline function")
        task_id = self.progress.stage_tasks.get(self._current_fn.__name__)
        if task_id is None:
            raise ValueError("No task ID available for stage")
        return task_id


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
            output_folder=options.output_folder,
            translation_provider=options.translation_provider,
        )
        context.set_input_file(input_file)

        # Define pipeline stages, optionally skipping voice isolation
        pipeline = [transcode]
        if not options.skip_voice_isolation:
            pipeline.append(voice_isolation)
        pipeline.extend([transcribe, translate, generate_deck])
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
        # Check if this is a terminal stage that should bypass cache
        if isinstance(func, PipelineFunction) and hasattr(func, "produced_artifacts"):
            for artifact_info in func.produced_artifacts.values():
                if artifact_info.get("terminal", False):
                    return False

        # Check if artifact caching is disabled globally
        if not self.options.use_artifact_cache:
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
        use_cache = self.should_use_cache(func) and self.options.use_artifact_cache

        # Try to get cached artifacts if appropriate
        cache_hit = False
        artifact_paths: dict[str, Path] = {}

        if use_cache and isinstance(func, PipelineFunction) and hasattr(func, "produced_artifacts"):
            cache_hit, artifact_paths = self.check_persistent_cache(func, kwargs)

            if not cache_hit:
                # Fall back to temp cache if persistent cache misses
                temp_cache_hit, temp_artifact_paths = self.get_cached_artifacts(func)
                if temp_cache_hit:
                    cache_hit = True
                    artifact_paths = temp_artifact_paths

        # Run the function if needed
        if not use_cache or not cache_hit:
            logging.debug(f"Running {func.__name__} (use_cache={use_cache}, cache_hit={cache_hit})")
            func(context=stage_context, **kwargs)

            # Store results in cache after running
            if use_cache:
                self.store_artifacts_in_cache(func, stage_context)

                # Also store in persistent cache if configured
                if isinstance(func, PipelineFunction) and hasattr(func, "produced_artifacts"):
                    self.store_in_persistent_cache(func, stage_context, kwargs)

            # Update artifacts with paths from generated files
            if isinstance(func, PipelineFunction) and hasattr(func, "produced_artifacts"):
                generated_paths: dict[str, Path] = {}
                for artifact_name in func.produced_artifacts:
                    artifact_path = stage_context.get_artifact_path(artifact_name)
                    generated_paths[artifact_name] = artifact_path
                self.update_artifacts(func, generated_paths)
        else:
            # Log cache hit at debug level - use logging module directly to avoid scope issues
            logger.debug(f"Using cached result for {func.__name__}")
            if self.options.debug:
                self.console.print(f"[green]Using cached result for {func.__name__}[/]")
            # Update artifacts with paths from cache
            self.update_artifacts(func, artifact_paths)

        self.context.progress.complete_stage()

    def check_persistent_cache(self, func: PipelineFunction, kwargs: dict[str, Any]) -> tuple[bool, dict[str, Path]]:
        """
        Check for cached artifacts in the persistent cache.

        Args:
            func: The pipeline function
            kwargs: The function arguments

        Returns:
            Tuple of (cache hit, artifact paths dict)
        """
        from . import artifact_cache

        func_name = func.__name__
        logger.debug(f"Checking persistent cache for function: {func_name}")
        logger.debug(f"Function kwargs: {list(kwargs.keys())}")

        # Resolve any path arguments to absolute paths for consistent caching
        normalized_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, Path):
                try:
                    normalized_kwargs[k] = v.resolve()
                    if str(v) != str(normalized_kwargs[k]):  # type: ignore
                        logger.debug(f"Normalized path {k}: {v} -> {normalized_kwargs[k]}")
                except Exception:
                    normalized_kwargs[k] = v
            else:
                normalized_kwargs[k] = v

        # Use normalized kwargs from this point on
        kwargs = normalized_kwargs

        artifact_paths: dict[str, Path] = {}
        all_found = True

        # Check if all artifacts for this function are in the cache
        for artifact_name, artifact_def in func.produced_artifacts.items():
            if not artifact_def.get("cache", False):
                logger.debug(f"Artifact '{artifact_name}' has caching disabled")
                continue

            # Resolve the version value
            version = resolve_version(artifact_def.get("version", 1), self.context)
            # Convert version to int for cache compatibility
            version_int = int(str(version))
            extension = artifact_def.get("extension", "mp3")

            logger.debug(f"Looking for cached artifact '{artifact_name}' with version {version}")

            cached_path, cache_hit = artifact_cache.get_cached_artifact(artifact_name, version_int, kwargs, extension)

            if cache_hit and cached_path:
                logger.debug(f"✅ Cache HIT for '{artifact_name}' at {cached_path}")
                artifact_paths[artifact_name] = cached_path
            else:
                logger.debug(f"❌ Cache MISS for '{artifact_name}'")
                all_found = False
                break

        result = all_found and bool(artifact_paths)
        logger.debug(f"Overall cache {'✅ HIT' if result else '❌ MISS'} for {func_name}")

        # Only return success if all artifacts were found
        return result, artifact_paths

    def store_in_persistent_cache(
        self, func: PipelineFunction, context: PipelineContext, kwargs: dict[str, Any]
    ) -> None:
        """
        Store all artifacts produced by this function in the persistent cache.

        Args:
            func: The pipeline function
            context: The pipeline context
            kwargs: The function arguments used
        """
        from . import artifact_cache

        func_name = func.__name__
        logger.debug(f"Storing artifacts in persistent cache for function: {func_name}")

        # Normalize any path arguments for consistent caching
        normalized_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, Path):
                try:
                    normalized_kwargs[k] = v.resolve()
                except Exception:
                    normalized_kwargs[k] = v
            else:
                normalized_kwargs[k] = v

        # Use normalized kwargs from this point on
        kwargs = normalized_kwargs

        for artifact_name, artifact_def in func.produced_artifacts.items():
            if not artifact_def.get("cache", False):
                logger.debug(f"Skipping cache storage for '{artifact_name}' (caching disabled)")
                continue

            # Resolve the version value
            version = resolve_version(artifact_def.get("version", 1), self.context)
            # Convert version to int for cache compatibility
            version_int = int(str(version))
            extension = artifact_def.get("extension", "mp3")

            artifact_path = context.get_artifact_path(artifact_name)
            logger.debug(f"Checking if artifact exists at {artifact_path}")

            if artifact_path.exists():
                try:
                    stored_path = artifact_cache.store_artifact(
                        artifact_name, version_int, kwargs, artifact_path, extension
                    )
                    logger.debug(
                        f"✅ Stored '{artifact_name}' in persistent cache (version {version}) at {stored_path}"
                    )
                except Exception as e:
                    logger.warning(f"❌ Failed to store '{artifact_name}' in persistent cache: {e}")
            else:
                logger.warning(f"❌ Cannot store '{artifact_name}' in cache - file does not exist at {artifact_path}")

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
            raise RuntimeError(f"Pipeline failed: {str(e)}") from e


def run_pipeline(input_file: Path, console: Console, options: PipelineOptions) -> Path:
    """Run the audio processing pipeline.

    Returns:
        Path: The path to the generated deck directory
    """
    from . import artifact_cache, cache
    from .utils import format_bytes

    # Clean up old artifacts in the persistent cache if enabled
    if options.use_artifact_cache and not options.skip_cache_cleanup:
        try:
            files_removed, bytes_freed = artifact_cache.clean_old_artifacts(days=14)
            if files_removed > 0:
                readable_size = format_bytes(bytes_freed)
                logger.info(f"Cleaned {files_removed} old artifacts from cache ({readable_size})")
        except Exception as e:
            logger.warning(f"Error cleaning up old cache artifacts: {e}")

    # Initialize a new temporary cache for this run
    cache.init_cache(keep_files=options.debug)
    cache_dir = cache.get_cache().temp_dir
    logger.info(f"Initialized temporary cache at {cache_dir}")
    logger.debug(f"Cache directory location: {cache_dir}")

    try:
        with PipelineProgress.create(console) as progress:
            # Create pipeline runner
            runner = PipelineRunner.create(input_file, console, options)
            runner.context.progress = progress

            # Run the pipeline
            result = runner.run()
            return result
    finally:
        cache_dir = cache.get_cache().temp_dir

        # In debug mode, preserve files and log location
        if options.debug:
            logger.debug(f"Intermediate files are preserved in cache directory: {cache_dir}")

        # In non-debug mode, always clean up regardless of how we exit
        else:
            logger.debug("Cleaning up cache directory")
            try:
                cache.cleanup_cache()
            except Exception as e:
                # Log cleanup errors but don't raise - we're in finally block
                logger.warning(f"Error cleaning up cache: {e}")


@pipeline_function(transcode={"extension": "mp3", "cache": True, "version": get_transcode_version})
def transcode(context: PipelineContext, input_path: Path) -> None:
    """Transcode an audio/video file to an audio file suitable for processing."""
    from .transcoder import transcode_audio

    output_path = context.get_artifact_path()
    transcode_audio(input_path, output_path, progress_callback=context.progress.update_progress)


@pipeline_function(voice_isolation={"extension": "mp3", "cache": True, "version": get_voice_isolation_version})
def voice_isolation(context: PipelineContext, transcode: Path) -> None:
    """Isolate voice from background noise."""
    from .voice_isolation import isolate_voice

    output_path = context.get_artifact_path()
    isolate_voice(transcode, output_path, progress_callback=context.progress.update_progress)


@pipeline_function(transcribe={"extension": "srt", "cache": True, "version": get_transcription_version})
def transcribe(context: PipelineContext, voice_isolation: Path | None = None, transcode: Path | None = None) -> None:
    """Transcribe audio to text."""
    from .transcribe import transcribe_audio

    # Use voice-isolated audio if available, otherwise use transcoded audio
    input_path = voice_isolation or transcode
    if not input_path:
        raise ValueError("No input audio file available for transcription")

    transcribe_audio(
        audio_file=input_path,
        transcript_path=context.get_artifact_path(),
        task_id=context.stage_task_id,
        progress=context.progress.progress,
        language=context.source_language,
    )


@pipeline_function(segments={"extension": "json", "cache": True, "version": get_translation_version})
def translate(context: PipelineContext, transcribe: Path) -> None:
    """Translate transcribed text to target language."""
    from .translate import translate_segments_to_json

    translate_segments_to_json(
        input_file=transcribe,
        output_file=context.get_artifact_path("segments"),
        target_language=context.target_language or "en",
        task_id=context.stage_task_id,
        progress=context.progress.progress,
        source_language=context.source_language,
        translation_provider=context.translation_provider,
    )


@pipeline_function(deck={"extension": "directory", "terminal": True})
def generate_deck(
    context: PipelineContext,
    segments: Path,
    translation: Path | None = None,
    transcribe: Path | None = None,
    pronunciation: Path | None = None,
    voice_isolation: Path | None = None,
    transcode: Path | None = None,
) -> None:
    """Generate Anki deck from translated segments."""
    from .anki import generate_anki_deck

    # Get the input audio file path
    input_audio = voice_isolation or transcode
    if not input_audio:
        raise ValueError("No input audio file available for deck generation")

    generate_anki_deck(
        segments_file=segments,
        input_audio_file=input_audio,
        transcription_file=transcribe,
        translation_file=translation,
        pronunciation_file=pronunciation,
        source_language=context.source_language,
        target_language=context.target_language or "english",
        task_id=context.stage_task_id,
        progress=context.progress,
    )

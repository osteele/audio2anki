"""Audio processing pipeline module."""

import hashlib
import inspect
import logging
import shutil
import signal
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NotRequired, Protocol, TypedDict, TypeVar, runtime_checkable

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

from . import artifact_cache
from .artifact_cache import try_hard_link
from .transcoder import TRANSCODING_FORMAT, get_transcode_hash
from .transcribe import get_transcription_hash
from .translate import TranslationProvider, get_translation_hash
from .types import LanguageCode
from .voice_isolation import VOICE_ISOLATION_FORMAT, get_voice_isolation_version

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Type definitions for pipeline artifacts
HashType = int | str | Callable[..., int | str]
VersionType = int | str


class ArtifactSpec(TypedDict):
    """Specification for a pipeline artifact."""

    extension: str
    cache: bool
    version: NotRequired[VersionType]
    hash: NotRequired[HashType]


class ArtifactSpecWithName(ArtifactSpec):
    """Artifact specification with a name field."""

    name: str


def create_artifact_spec(
    extension: str,
    cache: bool | None = None,
    version: VersionType | None = None,
    hash: HashType | None = None,
) -> ArtifactSpec:
    """
    Create a validated artifact specification.

    Args:
        extension: File extension for the artifact
        cache: Whether to cache the artifact output
        version: Version number/string for cache invalidation
        hash: Hash function/value for cache invalidation

    Returns:
        A validated ArtifactSpec
    """
    # Either version or hash implies cache=True
    if version is not None or hash is not None:
        if cache is False:
            raise ValueError("Cannot specify version or hash without caching")
        cache = True

    spec: ArtifactSpec = {"extension": extension, "cache": cache or False}

    # Add version if specified
    if version is not None:
        spec["version"] = version

    # Add hash if specified
    if hash is not None:
        spec["hash"] = hash

    return spec


def create_artifact_spec_from_dict(data: dict[str, Any]) -> ArtifactSpec:
    """
    Create an ArtifactSpec from a dictionary.

    Args:
        data: Dictionary with artifact specification

    Returns:
        A validated ArtifactSpec
    """
    return create_artifact_spec(
        extension=data.get("extension", "mp3"),
        cache=data.get("cache", False),
        version=data.get("version"),
        hash=data.get("hash"),
    )


@runtime_checkable
class PipelineFunction(Protocol):
    """Protocol for pipeline functions."""

    __name__: str
    produced_artifacts: dict[str, ArtifactSpec]

    def __call__(self, context: "PipelineContext", **kwargs: Any) -> None: ...


def resolve_hash(hash_value: HashType, context: "PipelineContext") -> int | str:
    """
    Resolve a hash value that can be an integer, string, or function.

    Args:
        hash_value: The hash value to resolve
        context: The pipeline context for function resolution

    Returns:
        The resolved hash value (int or str)

    Raises:
        ValueError: If the hash function requires arguments that aren't available in the context
    """
    if isinstance(hash_value, int | str):
        return hash_value

    # If it's a function, call it with context attributes
    if callable(hash_value):
        # Get the function's parameter names
        sig = inspect.signature(hash_value)
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
                    f"Hash function {hash_value.__name__} requires parameter '{param_name}' "
                    f"which is not available in the pipeline context"
                )

        return hash_value(**kwargs)

    raise ValueError(f"Invalid hash type: {type(hash_value)}")


def pipeline_function(
    extension: str | None = None,
    cache: bool | None = None,
    version: VersionType | None = None,
    hash: HashType | None = None,
    artifacts: list[dict[str, Any]] | None = None,
) -> Callable[[Callable[..., None]], PipelineFunction]:
    """
    Decorator that annotates a pipeline function with the artifacts it produces.

    Example usage:
        # Simple usage with default artifact name (function name)
        @pipeline_function(extension="srt")
        def transcribe(...): ...

        # With explicit caching and version
        @pipeline_function(extension="srt", version=2)
        def translate(...): ...

        # With hash function for caching
        @pipeline_function(extension="srt", hash=get_translation_hash)
        def translate(...): ...

        # For terminal functions that don't produce artifacts
        @pipeline_function(artifacts=[])  # or artifacts=None
        def terminal_function(...): ...

    Args:
        extension: File extension for the artifact (when using simplified form)
        cache: Whether to cache the artifact output (default: False)
        version: Version number/string for cache invalidation
        hash: Hash function/value for cache invalidation
        artifacts: List of artifact definitions or None for terminal functions
    """

    def decorator(func: Callable[..., None]) -> PipelineFunction:
        produced_artifacts: dict[str, ArtifactSpec] = {}

        # Case 1: Using simplified form with extension/cache/version/hash
        if extension or cache or version is not None or hash is not None:
            # Use function name as artifact name
            artifact_name = func.__name__
            artifact_spec = create_artifact_spec(extension=extension or "mp3", cache=cache, version=version, hash=hash)
            produced_artifacts[artifact_name] = artifact_spec

        # Case 2: Handle artifacts list (including empty list or None)
        for artifact_data in artifacts or []:
            artifact_name = artifact_data.get("name", func.__name__)
            # Create a clean artifact definition using our helper
            artifact_spec = create_artifact_spec_from_dict(artifact_data)
            produced_artifacts[artifact_name] = artifact_spec

        # Store the artifact definitions
        func.produced_artifacts = produced_artifacts  # type: ignore

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


PipelineFunctionType = PipelineFunction


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
        # Empty artifacts list means terminal function (no caching)
        if not func.produced_artifacts:
            return False

        # Check if artifact caching is disabled globally
        return self.options.use_artifact_cache

    def get_cached_artifacts(self, func: PipelineFunctionType) -> tuple[bool, dict[str, Path]]:
        """
        Try to retrieve all artifacts for this function from cache.

        Returns:
            Tuple of (cache_hit, artifact_paths)
        """
        artifact_paths: dict[str, Path] = {}
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
        for artifact_name in func.produced_artifacts:
            artifact_path = context.get_artifact_path(artifact_name)
            if artifact_path.exists():
                context.store_in_cache(artifact_name, artifact_path, artifact_path)

    def update_artifacts(self, func: PipelineFunctionType, artifact_paths: dict[str, Path]) -> None:
        """Update the artifacts dictionary with new paths."""
        for artifact_name, path in artifact_paths.items():
            self.artifacts[artifact_name] = path

    def copy_to_temp_cache(self, source_path: Path, artifact_name: str, extension: str) -> Path:
        """Copy an artifact from the persistent cache to the temporary cache using hard links when possible.

        Args:
            source_path: Path to the source file in the persistent cache
            artifact_name: Name of the artifact
            extension: File extension for the artifact

        Returns:
            Path to the copied file in the temporary cache
        """
        from . import cache

        # Get the destination path in the temporary cache
        dest_path = cache.get_artifact_path(artifact_name, extension)

        # Try to create a hard link first
        if try_hard_link(source_path, dest_path):
            logger.debug(f"Created hard link from {source_path} to {dest_path}")
            return dest_path

        # Fall back to regular copying if hard linking fails
        try:
            logger.debug(f"Copying file from {source_path} to {dest_path} using regular copy")
            shutil.copy2(source_path, dest_path)
            return dest_path
        except Exception as e:
            error_msg = f"Failed to copy artifact from persistent cache to temporary cache: {e}"
            logger.error(error_msg)
            console = Console(stderr=True)
            console.print(f"[bold red]Error:[/] {error_msg}")
            console.print("[yellow]Hint:[/] Check disk space and permissions.")
            import sys

            sys.exit(1)

    def get_function_kwargs(self, func: PipelineFunctionType) -> dict[str, Any]:
        """Get the required arguments for this function from artifacts."""
        params = inspect.signature(func).parameters
        return {name: self.artifacts[name] for name in params if name != "context" and name in self.artifacts}

    def update_input_tracking(
        self, func: PipelineFunctionType, context: PipelineContext, kwargs: dict[str, Any]
    ) -> None:
        """Set up input tracking for all artifacts this function produces."""
        # Get all input paths that aren't the context
        input_paths = {name: path for name, path in kwargs.items() if name != "context" and isinstance(path, Path)}

        # For each artifact, set up all inputs
        for artifact_name in func.produced_artifacts:
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

        if use_cache:
            cache_hit, artifact_paths = self.check_persistent_cache(func, kwargs)

            if not cache_hit:
                # Fall back to temp cache if persistent cache misses
                cache_hit, artifact_paths = self.get_cached_artifacts(func)

        # Run the function if needed
        if cache_hit:
            # Log cache hit at debug level - use logging module directly to avoid scope issues
            logger.debug(f"Using cached result for {func.__name__}")
            if self.options.debug:
                self.console.print(f"[green]Using cached result for {func.__name__}[/]")
            # Update artifacts with paths from cache
            self.update_artifacts(func, artifact_paths)
        else:
            logging.debug(f"Running {func.__name__} (use_cache={use_cache}, cache_hit={cache_hit})")
            func(context=stage_context, **kwargs)

            # Store results in cache after running
            if use_cache:
                self.store_artifacts_in_cache(func, stage_context)
                self.store_in_persistent_cache(func, stage_context, kwargs)

            # Update artifacts with paths from generated files
            generated_paths: dict[str, Path] = {}
            for artifact_name in func.produced_artifacts:
                artifact_path = stage_context.get_artifact_path(artifact_name)
                generated_paths[artifact_name] = artifact_path
                self.update_artifacts(func, generated_paths)

        self.context.progress.complete_stage()

    def compute_output_cache_key(self, artifact_def: ArtifactSpec, inputs: dict[str, Path]) -> str:
        """
        Compute a cache key for an artifact.

        Args:
            artifact_def: The artifact definition
            inputs: The input files for the artifact

        Returns:
            An integer hash suitable for cache keys
        """

        # Start with file hash if we have input files
        content_hash = hashlib.md5()
        for _, v in sorted(inputs.items()):  # Sort kwargs for consistent hashing
            with open(v, "rb") as f:
                content_hash.update(f.read())

        # Add version if specified
        version = artifact_def.get("version")
        if version is not None:
            content_hash.update(str(version).encode())

        # Add resolved hash if specified
        hash_value = artifact_def.get("hash")
        if hash_value is not None:
            resolved_hash = resolve_hash(hash_value, self.context)
            content_hash.update(str(resolved_hash).encode())

        # Convert to integer for cache compatibility
        return content_hash.hexdigest()

    def check_persistent_cache(self, func: PipelineFunction, kwargs: dict[str, Any]) -> tuple[bool, dict[str, Path]]:
        """
        Check for cached artifacts in the persistent cache.

        Args:
            func: The pipeline function
            kwargs: The function arguments

        Returns:
            Tuple of (cache hit, artifact paths dict)
        """

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

            cache_key = self.compute_output_cache_key(artifact_def, kwargs)
            extension = artifact_def.get("extension", "mp3")

            logger.debug(f"Looking for cached artifact '{artifact_name}' with hash {cache_key}")

            cached_path, cache_hit = artifact_cache.get_cached_artifact(artifact_name, cache_key, kwargs, extension)

            if cache_hit and cached_path:
                logger.debug(f"✅ Cache HIT for '{artifact_name}' at {cached_path}")
                # Copy the artifact from the persistent cache to the temporary cache using hard links when possible
                temp_path = self.copy_to_temp_cache(cached_path, artifact_name, extension)
                artifact_paths[artifact_name] = temp_path
                logger.debug(f"Copied artifact from persistent cache to temporary cache at {temp_path}")
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

        for artifact_name, artifact_def in sorted(
            func.produced_artifacts.items()
        ):  # Sort artifacts for consistent order
            if not artifact_def.get("cache", False):
                logger.debug(f"Skipping cache storage for '{artifact_name}' (caching disabled)")
                continue

            cache_key = self.compute_output_cache_key(artifact_def, kwargs)
            extension = artifact_def.get("extension", "mp3")

            artifact_path = context.get_artifact_path(artifact_name)
            logger.debug(f"Checking if artifact exists at {artifact_path}")

            if artifact_path.exists():
                try:
                    stored_path = artifact_cache.store_artifact(
                        artifact_name, cache_key, kwargs, artifact_path, extension
                    )
                    logger.debug(f"✅ Stored '{artifact_name}' in persistent cache (hash {cache_key}) at {stored_path}")
                except Exception as e:
                    error_msg = f"❌ Failed to store '{artifact_name}' in persistent cache: {e}"
                    logger.error(error_msg)
                    from rich.console import Console

                    console = Console(stderr=True)
                    console.print(f"[bold red]Error:[/] {error_msg}")
                    console.print("[yellow]Hint:[/] Check disk space or disable artifact cache with --no-cache flag.")
                    import sys

                    sys.exit(1)
            else:
                error_msg = f"❌ Cannot store '{artifact_name}' in cache - file does not exist at {artifact_path}"
                logger.error(error_msg)
                from rich.console import Console

                console = Console(stderr=True)
                console.print(f"[bold red]Error:[/] {error_msg}")
                import sys

                sys.exit(1)

    def run(self) -> Path:
        """Run the entire pipeline and return the final artifact path."""
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
                logging.error(f"{error_type} in {func.__name__}: {e!s}", exc_info=True)
                self.console.print(f"[red]Error in {func.__name__} ({error_type}): {e!s}[/]")
                raise

        # Get the output folder for the final deck
        output_folder = self.context.output_folder

        if output_folder and output_folder.exists():
            # Print success message with the output folder path
            self.console.print(f"\nDeck created at: {output_folder}")
            return output_folder
        else:
            raise ValueError("Pipeline completed but no deck artifact was produced")


def run_pipeline(input_file: Path, console: Console, options: PipelineOptions) -> Path:
    """Run the audio processing pipeline.

    Returns:
        Path: The path to the generated deck directory
    """
    import atexit
    import sys as signal_sys  # Import sys with alias to avoid conflicts

    from . import cache
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

    # Define cleanup function for signal handlers and atexit
    def cleanup_temp_cache(signum: int | None = None, frame: Any = None) -> None:
        if not options.debug:
            logger.debug("Cleaning up cache directory due to program termination")
            try:
                cache.cleanup_cache()
                if signum is not None:
                    # If this was called from a signal handler, exit the program
                    signal_sys.exit(1)
            except Exception as e:
                logger.warning(f"Error cleaning up cache: {e}")

    # Register signal handlers and atexit hook for cleanup
    if not options.debug:
        # Register cleanup for normal exit
        atexit.register(cleanup_temp_cache)
        # Register cleanup for signals
        signal.signal(signal.SIGINT, cleanup_temp_cache)  # Ctrl+C
        signal.signal(signal.SIGTERM, cleanup_temp_cache)  # kill command

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


@pipeline_function(extension=TRANSCODING_FORMAT, hash=get_transcode_hash)
def transcode(context: PipelineContext, input_path: Path) -> None:
    """Transcode an audio/video file to an audio file suitable for processing."""
    from .transcoder import transcode_audio

    output_path = context.get_artifact_path()
    transcode_audio(input_path, output_path, progress_callback=context.progress.update_progress)


@pipeline_function(extension=VOICE_ISOLATION_FORMAT, version=get_voice_isolation_version())
def voice_isolation(context: PipelineContext, transcode: Path) -> None:
    """Isolate voice from background noise."""
    from .voice_isolation import isolate_voice

    output_path = context.get_artifact_path()
    isolate_voice(transcode, output_path, progress_callback=context.progress.update_progress)


@pipeline_function(extension="json", hash=get_transcription_hash)
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


@pipeline_function(extension="json", hash=get_translation_hash)
def translate(context: PipelineContext, transcribe: Path) -> None:
    """Translate transcribed text to target language."""
    from .translate import translate_segments_to_json

    translate_segments_to_json(
        input_file=transcribe,
        output_file=context.get_artifact_path(),
        target_language=context.target_language or LanguageCode("en"),
        task_id=context.stage_task_id,
        progress=context.progress.progress,
        source_language=context.source_language,
        translation_provider=context.translation_provider,
    )


@pipeline_function(artifacts=[])
def generate_deck(
    context: PipelineContext,
    translate: Path,
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
        segments_file=translate,
        input_audio_file=input_audio,
        transcription_file=transcribe,
        translation_file=None,  # This parameter is no longer used
        pronunciation_file=pronunciation,
        source_language=context.source_language,
        target_language=context.target_language or "english",
        task_id=context.stage_task_id,
        progress=context.progress,
        output_folder=context.output_folder,
    )

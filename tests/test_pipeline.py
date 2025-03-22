"""Tests for the pipeline module."""

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from rich.console import Console
from rich.progress import Progress, TaskID

from audio2anki.pipeline import (
    PipelineContext,
    PipelineFunctionType,
    PipelineOptions,
    PipelineProgress,
    PipelineRunner,
    generate_deck,
    pipeline_function,
    run_pipeline,
    validate_pipeline,
)


# Create a pipeline-compatible version of generate_deck just for testing
@pipeline_function(deck={"extension": "apkg"})
def pipeline_generate_deck(
    context: PipelineContext,
    voice_isolation: Path,
    transcribe: Path,
    translation: Path,
    pronunciation: Path | None,
) -> None:
    """Wrapper around generate_deck that conforms to PipelineFunction protocol."""
    return generate_deck(
        context=context,
        voice_isolation=voice_isolation,
        transcribe=transcribe,
        translation=translation,
        pronunciation=pronunciation,
    )


@pytest.fixture
def mock_progress() -> Progress:
    """Create a mock progress bar."""
    mock = Mock(spec=Progress)
    mock.update = Mock()  # Explicitly create update method
    return mock


@pytest.fixture
def mock_console() -> Console:
    """Create a mock console."""
    mock = Mock(spec=Console)
    mock.get_time = Mock(return_value=0.0)
    mock.print = Mock()
    return mock


@pytest.fixture
def mock_pipeline_progress(mock_progress: Progress, mock_console: Console) -> PipelineProgress:
    """Create a mock pipeline progress tracker."""
    progress = Mock(spec=PipelineProgress)
    progress.progress = mock_progress
    progress.console = mock_console
    progress.current_stage = "generate_deck"
    progress.stage_tasks = {"generate_deck": Mock(spec=TaskID)}
    progress.complete_stage = Mock()
    progress.start_stage = Mock()
    progress.update_progress = Mock()
    return progress


@pytest.fixture
def pipeline_runner(mock_pipeline_progress: PipelineProgress, mock_console: Console, tmp_path: Path) -> PipelineRunner:
    """Create a pipeline runner for testing."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("test content")

    options = PipelineOptions(source_language="chinese", target_language="english")

    # Create the runner without using class method to avoid Rich's Progress initialization
    context = PipelineContext(
        progress=mock_pipeline_progress,
        source_language=options.source_language,
        target_language=options.target_language,
    )
    context.set_input_file(input_file)

    # Define pipeline stages for testing with proper type annotation
    pipeline: list[PipelineFunctionType] = []
    initial_artifacts = {"input_path": input_file}

    runner = PipelineRunner(
        context=context,
        options=options,
        console=mock_console,
        artifacts=initial_artifacts,
        pipeline=pipeline,
    )

    return runner


def test_validate_pipeline() -> None:
    """Test pipeline validation."""

    @pipeline_function(output1={"extension": "txt"})
    def func1(context: PipelineContext, input_path: Path) -> None:
        pass

    @pipeline_function(output2={"extension": "txt"})
    def func2(context: PipelineContext, output1: Path) -> None:
        pass

    @pipeline_function(output3={"extension": "txt"}, output4={"extension": "txt"})
    def func3(context: PipelineContext, output2: Path) -> None:
        pass

    @pipeline_function(output5={"extension": "txt"})
    def func4(context: PipelineContext, missing: Path) -> None:
        pass

    # Test valid pipeline with single artifacts
    pipeline = [func1, func2]
    initial_artifacts = {"input_path": Path("input.txt")}
    validate_pipeline(pipeline, initial_artifacts)  # Should not raise

    # Test valid pipeline with multiple artifacts
    pipeline = [func1, func2, func3]
    validate_pipeline(pipeline, initial_artifacts)  # Should not raise

    # Test invalid pipeline (missing artifact)
    pipeline = [func4]
    with pytest.raises(ValueError, match="missing"):
        validate_pipeline(pipeline, initial_artifacts)


def test_temp_dir_cache(tmp_path: Path) -> None:
    """Test that the temporary directory cache works correctly."""
    from audio2anki.cache import TempDirCache

    # Create a test temporary cache
    cache = TempDirCache(keep_files=True)
    try:
        # Test getting paths for different artifacts
        audio_path = cache.get_path("audio", "mp3")
        transcript_path = cache.get_path("transcript", "srt")

        # Paths should be in the temp directory
        assert str(cache.temp_dir) in str(audio_path)
        assert str(cache.temp_dir) in str(transcript_path)

        # Paths should end with the correct names
        assert str(audio_path).endswith("audio.mp3")
        assert str(transcript_path).endswith("transcript.srt")

        # Test storing an artifact
        test_data = b"test artifact data"
        path = cache.store("test_artifact", test_data, "txt")

        # Check file exists and has correct content
        assert path.exists()
        with open(path, "rb") as f:
            assert f.read() == test_data
    finally:
        # Cleanup
        cache.cleanup()


def test_pipeline_cache_integration(mock_pipeline_progress: PipelineProgress, tmp_path: Path) -> None:
    """Test the integration of the temp cache in the pipeline context."""
    from audio2anki.cache import cleanup_cache, init_cache
    from audio2anki.pipeline import PipelineContext, pipeline_function

    # Initialize a test cache
    init_cache(keep_files=True)
    try:
        # Create a temporary input file
        input_file = tmp_path / "input.txt"
        input_file.write_text("test content for caching")

        # Create a context
        context = PipelineContext(progress=mock_pipeline_progress)
        context.set_input_file(input_file)

        # Create a test pipeline function
        @pipeline_function(test_artifact={"extension": "txt"})
        def test_function(context: PipelineContext) -> None:
            """Test function that produces a single artifact."""
            pass

        # Set up the context for this function
        stage_context = context.for_stage(test_function)

        # Get the artifact path
        artifact_path = stage_context.get_artifact_path("test_artifact")

        # The path should be in the temp directory and have the correct name
        assert "audio2anki_" in str(artifact_path)
        assert str(artifact_path).endswith("test_artifact.txt")
    finally:
        # Clean up
        cleanup_cache()


def test_pipeline_runner_should_use_cache(pipeline_runner: PipelineRunner) -> None:
    """Test the should_use_cache method of PipelineRunner."""

    # Create test pipeline functions
    @pipeline_function(test1={"extension": "txt"})
    def normal_func(context: PipelineContext) -> None:
        pass

    @pipeline_function(test2={"extension": "txt", "terminal": True})
    def terminal_func(context: PipelineContext) -> None:
        pass

    # Test normal function with bypass_cache=False
    assert pipeline_runner.should_use_cache(normal_func) is True

    # Test terminal function
    assert pipeline_runner.should_use_cache(terminal_func) is False

    # We no longer have bypass_cache option, but terminal functions still skip cache
    assert pipeline_runner.should_use_cache(terminal_func) is False


def test_pipeline_runner_get_cached_artifacts(pipeline_runner: PipelineRunner, tmp_path: Path) -> None:
    """Test the get_cached_artifacts method of PipelineRunner."""

    # Create a test pipeline function
    @pipeline_function(artifact1={"extension": "txt"})
    def test_func(context: PipelineContext) -> None:
        pass

    # Mock the retrieve_from_cache method to simulate a cache hit
    with patch.object(PipelineContext, "retrieve_from_cache") as mock_retrieve:
        # Set up the mock to return a path for the first call, None for the second
        mock_path = tmp_path / "cached_artifact.txt"
        mock_retrieve.return_value = mock_path

        # Test cache hit
        cache_hit, paths = pipeline_runner.get_cached_artifacts(test_func)
        assert cache_hit is True
        assert "artifact1" in paths
        assert paths["artifact1"] == mock_path

        # Set up for cache miss
        mock_retrieve.return_value = None

        # Test cache miss
        cache_hit, paths = pipeline_runner.get_cached_artifacts(test_func)
        assert cache_hit is False
        assert len(paths) == 0


def test_pipeline_runner_update_artifacts(pipeline_runner: PipelineRunner, tmp_path: Path) -> None:
    """Test the update_artifacts method of PipelineRunner."""

    # Create a test pipeline function with single artifact
    @pipeline_function(single_artifact={"extension": "txt"})
    def single_func(context: PipelineContext) -> None:
        pass

    # Create a test pipeline function with multiple artifacts
    @pipeline_function(artifact1={"extension": "txt"}, artifact2={"extension": "txt"})
    def multi_func(context: PipelineContext) -> None:
        pass

    # Test single artifact function
    paths = {"single_artifact": tmp_path / "single.txt"}
    pipeline_runner.update_artifacts(single_func, paths)

    # Check that both the artifact name and function name are keys
    assert pipeline_runner.artifacts["single_artifact"] == paths["single_artifact"]
    assert pipeline_runner.artifacts["single_func"] == paths["single_artifact"]

    # Test multiple artifact function
    paths = {"artifact1": tmp_path / "artifact1.txt", "artifact2": tmp_path / "artifact2.txt"}
    pipeline_runner.update_artifacts(multi_func, paths)

    # Check that all artifacts are added but function name is not a key
    assert pipeline_runner.artifacts["artifact1"] == paths["artifact1"]
    assert pipeline_runner.artifacts["artifact2"] == paths["artifact2"]
    assert "multi_func" not in pipeline_runner.artifacts


def test_pipeline_stages(test_audio_file: Path, tmp_path: Path) -> None:
    """Test that pipeline stages produce the expected artifacts."""
    # Create pipeline options
    options = PipelineOptions(
        source_language="chinese",
        target_language="english",
    )

    # Set up a console for the test
    console = Console()

    # Create a dummy regular file path for transcode output, not a directory
    dummy_regular_file = tmp_path / "test_output.mp3"
    dummy_regular_file.touch()

    # Create a dummy result path to mock the final return
    dummy_result_path = tmp_path / "deck"
    dummy_result_path.mkdir(exist_ok=True)

    # Create a mock cache for testing
    with patch("audio2anki.cache.init_cache") as mock_init_cache:
        # Set up a mock TempDirCache
        mock_cache = Mock()
        mock_cache.get_path.return_value = dummy_regular_file
        mock_init_cache.return_value = mock_cache
        # Mock store_in_cache to prevent trying to cache a directory
        with patch.object(PipelineContext, "store_in_cache", return_value=None):
            # Add patch for the final get_artifact_path call in run()
            with patch.object(PipelineContext, "get_artifact_path") as mock_get_path:
                # Return regular file for most calls, directory only for the final call
                def get_path_side_effect(artifact_name: str = ""):
                    if not artifact_name or artifact_name == "deck":
                        return dummy_result_path
                    return dummy_regular_file

                mock_get_path.side_effect = get_path_side_effect

                # Patch the implementation functions that would make external API calls
                with (
                    # Patch the PipelineRunner.create class method to avoid Progress init issues
                    patch(
                        "audio2anki.pipeline.PipelineRunner.create",
                        return_value=PipelineRunner.create(test_audio_file, console, options),
                    ),
                    # Transcode stage
                    patch("audio2anki.transcoder.transcode_audio") as mock_transcode_audio,
                    # Voice isolation - patch the base function, not the API call
                    patch("audio2anki.voice_isolation.isolate_voice") as mock_isolate_voice,
                    # Transcribe stage
                    patch("audio2anki.transcribe.transcribe_audio") as mock_transcribe_audio,
                    # Translate stage
                    patch("audio2anki.translate.translate_srt") as mock_translate_srt,
                    # Generate deck stage
                    patch("audio2anki.anki.generate_anki_deck") as mock_generate_anki_deck,
                ):
                    # Configure the mocks to create files when called
                    def transcode_side_effect(input_path: Path, output_path: Path, **kwargs: Any) -> None:
                        if not output_path.exists():
                            output_path.touch()
                        return None

                    mock_transcode_audio.side_effect = transcode_side_effect

                    # Mock voice isolation to create an output file
                    def mock_voice_isolation_impl(
                        context: PipelineContext | Path, input_path: Path, **kwargs: Any
                    ) -> Path:
                        # Return the dummy file path directly to avoid file operations
                        return dummy_regular_file

                    mock_isolate_voice.side_effect = mock_voice_isolation_impl

                    # Mock transcribe to create an output file
                    def transcribe_side_effect(audio_file: Path, transcript_path: Path, **kwargs: Any) -> Any:
                        if not transcript_path.exists():
                            transcript_path.touch()
                        return []  # Return empty list of segments

                    mock_transcribe_audio.side_effect = transcribe_side_effect

                    # Mock translate to create output files
                    def mock_translate_srt_impl(
                        input_file: Path, translation_output: Path, pronunciation_output: Path, **kwargs: Any
                    ) -> tuple[Path, Path]:
                        if not translation_output.exists():
                            translation_output.touch()
                        if not pronunciation_output.exists():
                            pronunciation_output.touch()
                        return translation_output, pronunciation_output

                    mock_translate_srt.side_effect = mock_translate_srt_impl

                    # Mock generate_anki_deck
                    def generate_deck_side_effect(**kwargs: Any) -> Path:
                        if not dummy_result_path.exists():
                            dummy_result_path.mkdir(exist_ok=True)
                        # Set the deck_path in the cache to the dummy_result_path
                        mock_cache.deck_path = dummy_result_path
                        return dummy_result_path

                    mock_generate_anki_deck.side_effect = generate_deck_side_effect

                    # Run the pipeline with the real pipeline functions
                    # Call run_pipeline and ignore the return value
                    run_pipeline(test_audio_file, console, options)

                    # Verify that artifacts were created - check in the tmp_path directory
                    assert dummy_result_path.exists()

                    # Check that the mocks were called
                    mock_transcode_audio.assert_called_once()
                    mock_isolate_voice.assert_called_once()
                    mock_transcribe_audio.assert_called_once()
                    mock_translate_srt.assert_called_once()
                    mock_generate_anki_deck.assert_called_once()


def test_execute_stage(pipeline_runner: PipelineRunner, tmp_path: Path) -> None:
    """Test the execute_stage method of PipelineRunner."""

    # Create a test pipeline function
    @pipeline_function(test_output={"extension": "txt"})
    def test_func(context: PipelineContext) -> None:
        # Since we can't actually write to the path in the test environment,
        # we'll just pass and mock the file existence check
        pass

    # Set up the context with a current_fn for get_artifact_path
    pipeline_runner.context = pipeline_runner.context.for_stage(test_func)

    # Make sure test_func is in the pipeline
    pipeline_runner.pipeline = [test_func]

    # Mock should_use_cache, get_cached_artifacts, and artifact path existence check
    with (
        patch.object(PipelineRunner, "should_use_cache", return_value=False) as mock_should_use_cache,
        patch.object(PipelineRunner, "get_cached_artifacts") as mock_get_cached_artifacts,
        patch.object(PipelineRunner, "store_artifacts_in_cache") as mock_store_artifacts,
        patch("pathlib.Path.exists", return_value=True),  # Make Path.exists() always return True
    ):
        # Set up cache miss scenario
        mock_get_cached_artifacts.return_value = (False, {})

        # Execute the stage with cache bypass
        pipeline_runner.execute_stage(test_func)

        # Check that the function was executed
        mock_should_use_cache.assert_called_once()
        mock_get_cached_artifacts.assert_not_called()

        # Reset mocks for cache hit test
        mock_should_use_cache.reset_mock()
        mock_should_use_cache.return_value = True

        # Set up cache hit scenario
        cached_path = tmp_path / "cached_output.txt"
        mock_get_cached_artifacts.return_value = (True, {"test_output": cached_path})

        # Execute the stage with cache hit
        pipeline_runner.execute_stage(test_func)

        # Check that function was not executed but artifacts were updated
        mock_should_use_cache.assert_called_once()
        mock_get_cached_artifacts.assert_called_once()
        mock_store_artifacts.assert_not_called()

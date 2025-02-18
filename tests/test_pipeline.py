"""Tests for the pipeline module."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from rich.console import Console
from rich.progress import Progress, TaskID

from audio2anki.pipeline import (
    PipelineContext,
    PipelineOptions,
    PipelineProgress,
    generate_deck,
    produces_artifacts,
    run_pipeline,
    transcode,
    transcribe,
    translate,
    validate_pipeline,
    voice_isolation,
)


# Create a pipeline-compatible version of generate_deck just for testing
@produces_artifacts(deck={"extension": "apkg"})
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
def mock_pipeline_progress(mock_progress: Progress) -> PipelineProgress:
    """Create a mock pipeline progress tracker."""
    progress = Mock(spec=PipelineProgress)
    progress.progress = mock_progress
    console = Mock(spec=Console)
    console.__enter__ = Mock(return_value=console)
    console.__exit__ = Mock()
    console.is_interactive = True
    console.is_jupyter = False
    console.clear_live = Mock()
    console.set_live = Mock()
    console.show_cursor = Mock()
    console.push_render_hook = Mock()
    console.get_time = Mock(return_value=0.0)
    progress.console = console
    progress.current_stage = "generate_deck"
    progress.stage_tasks = {"generate_deck": Mock(spec=TaskID)}
    return progress


def test_validate_pipeline() -> None:
    """Test pipeline validation."""

    @produces_artifacts(output1=Path)
    def func1(context: PipelineContext, input_path: Path) -> None:
        pass

    @produces_artifacts(output2=Path)
    def func2(context: PipelineContext, output1: Path) -> None:
        pass

    @produces_artifacts(output3=Path, output4=Path)
    def func3(context: PipelineContext, output2: Path) -> None:
        pass

    @produces_artifacts(output5=Path)
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


def test_pipeline_stages(test_audio_file: Path, mock_pipeline_progress: PipelineProgress) -> None:
    """Test that pipeline stages are called correctly by run_pipeline."""
    # Create pipeline options
    options = PipelineOptions(
        source_language="chinese",
        target_language="english",
    )

    # Create a new PipelineProgress for this test
    with patch("audio2anki.pipeline.PipelineProgress.create") as mock_create:
        # Create a MagicMock that supports context manager methods
        mock_progress = MagicMock()
        mock_progress.__enter__.return_value = mock_progress
        mock_progress.__exit__.return_value = None
        mock_create.return_value = mock_progress

        context = PipelineContext(
            progress=mock_progress,
            source_language=options.source_language,
            target_language=options.target_language,
        )

        # Define the pipeline using the *real* stage functions

        # Mock external dependencies of the pipeline stages
        with (
            patch("audio2anki.transcoder.transcode_audio", autospec=True) as mock_transcode_audio,
            patch("audio2anki.voice_isolation.isolate_voice", autospec=True) as mock_isolate_voice,
            patch("audio2anki.transcribe.transcribe_audio", autospec=True) as mock_transcribe_audio,
            patch("audio2anki.translate.translate_srt", autospec=True) as mock_translate_srt,
            patch("audio2anki.anki.generate_anki_deck", autospec=True) as mock_generate_anki_deck,
        ):
            # Set up mocks for each stage
            transcode_context = context.for_stage(transcode)
            voice_isolation_context = context.for_stage(voice_isolation)
            transcribe_context = context.for_stage(transcribe)
            translate_context = context.for_stage(translate)
            generate_deck_context = context.for_stage(pipeline_generate_deck)

            # Configure mocks with the appropriate context
            mock_transcode_audio.side_effect = lambda input_path, output_path, **kwargs: output_path.touch()
            mock_isolate_voice.side_effect = lambda input_path, output_path, **kwargs: output_path.touch()
            mock_transcribe_audio.side_effect = (
                lambda audio_file, transcript_path, *args, **kwargs: transcript_path.touch()
            )

            def mock_translate_srt_side_effect(
                input_file, translation_output, pronunciation_output, *args, source_language=None, **kwargs
            ):
                translation_output.touch()
                if source_language == "chinese":
                    pronunciation_output.touch()

            mock_translate_srt.side_effect = mock_translate_srt_side_effect

            # Mock generate_anki_deck.  It returns None, but we simulate file creation.
            mock_generate_anki_deck.side_effect = lambda *args, **kwargs: Path("deck").mkdir(exist_ok=True)

            # Call run_pipeline with the correct arguments
            run_pipeline(test_audio_file, mock_progress.console, options)

            # Verify the calls to each mock
            mock_transcode_audio.assert_called_once_with(
                test_audio_file,
                transcode_context.get_artifact_path("transcode"),
                progress_callback=transcode_context.progress.update_progress,
            )

            mock_isolate_voice.assert_called_once_with(
                transcode_context.get_artifact_path("transcode"),
                voice_isolation_context.get_artifact_path("voice_isolation"),
                progress_callback=voice_isolation_context.progress.update_progress,
            )

            mock_transcribe_audio.assert_called_once_with(
                audio_file=voice_isolation_context.get_artifact_path("voice_isolation"),
                transcript_path=transcribe_context.get_artifact_path("transcribe"),
                model="whisper-1",
                task_id=transcribe_context.stage_task_id,
                progress=transcribe_context.progress.progress,
                language="zh",
            )

            mock_translate_srt.assert_called_once_with(
                input_file=transcribe_context.get_artifact_path("transcribe"),
                translation_output=translate_context.get_artifact_path("translation"),
                pronunciation_output=translate_context.get_artifact_path("pronunciation"),
                target_language=translate_context.target_language or "english",
                task_id=translate_context.stage_task_id,
                progress=translate_context.progress.progress,
                source_language=translate_context.source_language,
            )

            mock_generate_anki_deck.assert_called_once_with(
                input_data=translate_context.get_artifact_path("translation"),
                input_audio_file=voice_isolation_context.get_artifact_path("voice_isolation"),
                transcription_file=transcribe_context.get_artifact_path("transcribe"),
                pronunciation_file=translate_context.get_artifact_path("pronunciation"),
                source_language=generate_deck_context.source_language,
                target_language=generate_deck_context.target_language,
                task_id=generate_deck_context.stage_task_id,
                progress=generate_deck_context.progress,
                output_path="deck",
            )

            # Verify that all output files were created
            assert transcode_context.get_artifact_path("transcode").exists()
            assert voice_isolation_context.get_artifact_path("voice_isolation").exists()
            assert transcribe_context.get_artifact_path("transcribe").exists()
            assert translate_context.get_artifact_path("translation").exists()
            assert translate_context.get_artifact_path("pronunciation").exists()

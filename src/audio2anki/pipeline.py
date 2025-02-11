import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol

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

# Setup basic logging configuration


class StageFunction(Protocol):
    """Type protocol for stage processing functions."""

    def __call__(self, input_data: Any, progress: "PipelineProgress", **kwargs: Any) -> Any: ...


@dataclass
class Stage:
    """A single stage in the audio processing pipeline."""

    name: str
    description: str
    process: StageFunction
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineOptions:
    """Options that control pipeline behavior."""

    bypass_cache: bool = False
    clear_cache: bool = False
    debug: bool = False


@dataclass
class PipelineProgress:
    """Manages progress tracking for the pipeline and its stages."""

    progress: Progress
    pipeline_task: TaskID
    console: Console
    stage_tasks: dict[str, TaskID] = field(default_factory=dict)
    current_stage: str | None = None

    @classmethod
    def create(cls, stages: list[Stage], console: Console) -> "PipelineProgress":
        """Create a new pipeline progress tracker with pre-allocated tasks for all stages."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        pipeline_task = progress.add_task("[bold blue]Processing audio file...", total=len(stages))

        # Pre-create tasks for all stages
        stage_tasks: dict[str, TaskID] = {}
        for stage in stages:
            task_id = progress.add_task(
                f"[cyan]{stage.description}",
                total=100,
                visible=False,
            )
            stage_tasks[stage.name] = task_id

        return cls(progress=progress, pipeline_task=pipeline_task, stage_tasks=stage_tasks, console=console)

    def __enter__(self) -> "PipelineProgress":
        """Start the progress display when entering the context."""
        self.progress.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop the progress display when exiting the context."""
        if self.current_stage:
            task_id = self.stage_tasks[self.current_stage]
            self.progress.update(task_id, visible=False)
        self.progress.stop()

    def start_stage(self, stage_name: str) -> None:
        """Start a pipeline stage."""
        # Hide previous stage if exists
        if self.current_stage:
            prev_task_id = self.stage_tasks[self.current_stage]
            self.progress.update(prev_task_id, visible=False)

        # Show and start new stage
        self.current_stage = stage_name
        task_id = self.stage_tasks[stage_name]
        self.progress.reset(task_id)
        self.progress.update(task_id, visible=True, completed=0)

    def update_stage(self, completed: float) -> None:
        """Update the current stage's progress."""
        if self.current_stage:
            task_id = self.stage_tasks[self.current_stage]
            self.progress.update(task_id, completed=completed)

    def complete_stage(self) -> None:
        """Mark the current stage as complete and advance the pipeline."""
        if self.current_stage:
            task_id = self.stage_tasks[self.current_stage]
            self.progress.update(task_id, completed=100)
            self.progress.advance(self.pipeline_task)
            self.current_stage = None


@dataclass
class Pipeline:
    """Audio processing pipeline that manages and executes stages."""

    stages: list[Stage] = field(default_factory=list)
    options: PipelineOptions = field(default_factory=PipelineOptions)

    def add_stage(self, stage: Stage) -> None:
        """Add a stage to the pipeline."""
        self.stages.append(stage)

    def run(self, input_data: Any, console: Console) -> Any:
        """Execute all stages in the pipeline."""
        if self.options.clear_cache:
            from . import cache

            cache.clear_cache()
            logging.info("Cache cleared")

        with PipelineProgress.create(self.stages, console) as progress:
            current_data = input_data
            for stage in self.stages:
                try:
                    progress.start_stage(stage.name)
                    current_data = stage.process(current_data, progress, **stage.params)
                    progress.complete_stage()
                except Exception as e:
                    logging.error(f"Error in stage '{stage.name}': {str(e)}")
                    console.print(f"[red]Pipeline failed at stage: {stage.name}[/]")
                    console.print(f"[red]Error: {str(e)}[/]")
                    sys.exit(1)
            return current_data


def transcode(input_data: str | Path, progress: PipelineProgress, **kwargs: Any) -> str:
    """Transcode an audio/video file to an audio file suitable for processing."""
    try:
        output_path = transcode_audio(input_data, progress_callback=progress.update_stage, **kwargs)
        return str(output_path)
    except Exception as e:
        logging.error(f"Transcoding failed: {e}")
        raise


def voice_isolation(input_data: Any, progress: PipelineProgress, **kwargs: Any) -> Path:
    """Isolate voice from background noise."""
    from .voice_isolation import VoiceIsolationError, isolate_voice

    try:
        return isolate_voice(input_data, progress.update_stage)
    except VoiceIsolationError as e:
        progress.console.print(f"[red]Voice isolation failed: {e}[/]")
        sys.exit(1)


def transcribe(input_data: str | Path, progress: PipelineProgress, **kwargs: Any) -> str:
    """Transcribe the audio file using OpenAI Whisper and produce an SRT file."""
    from pathlib import Path

    from .transcribe import transcribe_audio

    input_path = Path(input_data)
    output_path = input_path.with_suffix(".srt")

    # Get the task ID for the current stage, defaulting to None if not found
    current_stage = progress.current_stage
    task_id = progress.stage_tasks.get(current_stage) if current_stage else None

    # Get source language from kwargs, defaulting to Chinese
    source_language = kwargs.pop("source_language", "chinese")

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
    language_code = language_codes.get(source_language.lower())

    # Remove target_language from kwargs as it's not used by transcribe_audio
    kwargs.pop("target_language", None)

    transcribe_audio(
        audio_file=input_path,
        transcript_path=output_path,
        model="whisper-1",
        progress=progress.progress,
        task_id=task_id,
        language=language_code,
        **kwargs,
    )

    return str(output_path)


def translate(input_data: str | Path, progress: PipelineProgress, **kwargs: Any) -> str:
    """Translate the SRT file to English and create pinyin if needed."""
    from pathlib import Path

    from .translate import translate_srt

    input_path = Path(input_data)

    # Get the task ID for the current stage
    task_id = None
    if progress.current_stage:
        task_id = progress.stage_tasks.get(progress.current_stage)
    if not task_id:
        task_id = progress.progress.add_task("Translating...", total=100)

    # Extract and remove language options from kwargs
    source_language = kwargs.pop("source_language", "chinese")
    target_language = kwargs.pop("target_language", None)

    # Remove any potential duplicates (if they exist)
    remaining_kwargs = {k: v for k, v in kwargs.items() if k not in ("source_language", "target_language")}

    # Call translate_srt with explicit source and target language, passing only remaining kwargs
    translated_file, pinyin_file = translate_srt(
        input_file=input_path,
        source_language=source_language,
        target_language=target_language,
        task_id=task_id,
        progress=progress.progress,
        **remaining_kwargs,
    )

    if pinyin_file:
        print(f"Created pinyin file: {pinyin_file}")

    return str(translated_file)


def sentence_selection(input_data: str | Path, progress: PipelineProgress, **kwargs: Any) -> str:
    """Process the transcript to perform sentence selection."""
    # Simulate selection progress
    progress.update_stage(50)  # In real implementation, update based on actual progress
    # Placeholder: In a real implementation, apply sentence splitting/filtering
    return str(input_data).replace(".srt", ".selected.txt")


def generate_deck(input_data: str | Path, progress: PipelineProgress, **kwargs: Any) -> None:
    """Generate an Anki flashcard deck from the processed data."""
    # Simulate deck generation progress
    progress.update_stage(50)  # In real implementation, update based on actual progress
    # Placeholder: In a real implementation, create directory structure with deck.txt, etc.


def create_pipeline(options: PipelineOptions) -> Pipeline:
    """Create and configure the audio processing pipeline."""
    pipeline = Pipeline(options=options)

    # Add stages in order
    pipeline.add_stage(Stage("transcode", "Transcoding audio", transcode))
    pipeline.add_stage(Stage("voice_isolation", "Isolating voice with Eleven Labs API", voice_isolation))
    pipeline.add_stage(Stage("transcribe", "Transcribing with OpenAI Whisper", transcribe))
    pipeline.add_stage(Stage("translate", "Translating transcript", translate))
    pipeline.add_stage(Stage("sentence_selection", "Selecting sentences", sentence_selection))
    pipeline.add_stage(Stage("generate_deck", "Generating Anki deck", generate_deck))

    return pipeline

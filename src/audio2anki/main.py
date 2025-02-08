"""Main entry point for audio2anki."""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import click
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
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
console = Console()


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
class PipelineProgress:
    """Manages progress tracking for the pipeline and its stages."""

    progress: Progress
    pipeline_task: TaskID
    stage_tasks: dict[str, TaskID] = field(default_factory=dict)
    current_stage: str | None = None

    @classmethod
    def create(cls, stages: list[Stage]) -> "PipelineProgress":
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
        stage_tasks = {}
        for stage in stages:
            task_id = progress.add_task(
                f"[cyan]{stage.description}",
                total=100,
                visible=False,
            )
            stage_tasks[stage.name] = task_id

        return cls(progress=progress, pipeline_task=pipeline_task, stage_tasks=stage_tasks)

    def __enter__(self) -> "PipelineProgress":
        """Start the progress display when entering the context."""
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
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

    def add_stage(self, stage: Stage) -> None:
        """Add a stage to the pipeline."""
        self.stages.append(stage)

    def run(self, input_data: Any) -> Any:
        """Execute all stages in the pipeline."""
        with PipelineProgress.create(self.stages) as progress:
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


def voice_isolation(input_data: str | Path, progress: PipelineProgress, **kwargs: Any) -> str:
    """Perform voice isolation using the selected provider."""
    # Simulate API call progress
    progress.update_stage(50)  # In real implementation, update based on API progress
    # Placeholder: In a real implementation, call the Eleven Labs API
    return str(input_data).replace(".audio.mp3", ".cleaned.mp3")


def transcribe(input_data: str | Path, progress: PipelineProgress, **kwargs: Any) -> str:
    """Transcribe the audio file using OpenAI Whisper and produce an SRT file."""
    # Simulate transcription progress
    progress.update_stage(50)  # In real implementation, update based on Whisper progress
    # Placeholder: In a real implementation, call OpenAI Whisper
    return str(input_data).replace(".cleaned.mp3", ".srt")


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


def create_pipeline() -> Pipeline:
    """Create and configure the audio processing pipeline."""
    pipeline = Pipeline()

    # Add stages in order
    pipeline.add_stage(Stage("transcode", "Transcoding audio", transcode))
    pipeline.add_stage(Stage("voice_isolation", "Isolating voice using Eleven Labs API", voice_isolation))
    pipeline.add_stage(Stage("transcribe", "Transcribing with OpenAI Whisper", transcribe))
    pipeline.add_stage(Stage("sentence_selection", "Selecting sentences", sentence_selection))
    pipeline.add_stage(Stage("generate_deck", "Generating Anki deck", generate_deck))

    return pipeline


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--debug", is_flag=True, help="Enable debug output")
def main(input_file: str, debug: bool) -> None:
    """Process an audio/video file and generate Anki flashcards."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug mode enabled.")

    # Create and run the pipeline
    pipeline = create_pipeline()
    pipeline.run(input_file)


if __name__ == "__main__":
    main()

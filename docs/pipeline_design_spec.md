# Audio-to-Anki Pipeline Architecture Specification

**Author:** Oliver Steele (GitHub: [osteele](https://github.com/osteele))
**Year:** 2024

---

## Overview

This document describes a generalized pipeline architecture for the audio-to-anki application, which processes audio and video inputs to generate Anki card decks. The design uses an artifact-aware pipeline where each stage explicitly declares its inputs and outputs. This approach improves type safety, testability, and makes data dependencies clear.

## Pipeline Architecture

The pipeline is implemented as a sequence of Python functions, where each function:
1. Explicitly declares its inputs through parameter names
2. Produces one or more named artifacts that become inputs for subsequent stages
3. Receives a context object containing pipeline-wide configuration and progress tracking

### Key Features

- **Artifact-Centric Design:** Each pipeline function produces one or more named artifacts
- **Explicit Data Flow:** Each function's inputs and outputs are clearly defined through its signature and artifact declarations
- **Static Validation:** The pipeline validates all required artifacts are available before execution
- **Type Safety:** Comprehensive type hints throughout the codebase
- **Progress Tracking:** Integrated progress reporting for each pipeline stage
- **Error Handling:** Clear error messages when required artifacts are missing

## Pipeline Operations

Each operation in the pipeline has the following structure:

```python
@produces_artifacts(output_name={"extension": "mp3"})  # Optional: declare artifact name and properties
def operation_name(
    context: PipelineContext,
    required_artifact1: Path,  # Name matches a previous stage's artifact name
    optional_param: int | None = None,
) -> None:
    """Process artifacts.
    
    Args:
        context: Pipeline-wide configuration and progress
        required_artifact1: Required input artifact
        optional_param: Optional configuration
    """
```

If a function is not decorated with `@produces_artifacts`, it produces a single artifact named after the function.

### Core Components

1. **PipelineContext:**
   ```python
   @dataclass
   class PipelineContext:
       """Holds pipeline state and configuration."""
       progress: PipelineProgress
       source_language: str = "chinese"
       target_language: str | None = None
       _artifacts: dict[str, dict[str, Any]]  # Maps artifact names to their properties
   ```

2. **Pipeline Runner:**
   - Manages the execution of pipeline stages
   - Tracks artifacts in a dictionary keyed by artifact name
   - Validates artifact availability before execution
   - Handles errors and progress reporting

3. **Pipeline Validation:**
   ```python
   def validate_pipeline(
       pipeline: list[PipelineFunction],
       initial_artifacts: dict[str, Any]
   ) -> None:
       """Validate that all required artifacts will be available."""
   ```

### Artifact Naming and Temporary Caching

- Each artifact has a unique name, either:
  - Specified in the `@pipeline_function` decorator, or
  - Defaulting to the function name if not decorated
- Each pipeline stage specifies the names of its input artifacts through parameter names (other than `context`)
- For each pipeline run, a temporary directory is created to store all artifacts
- The artifact filename has the simple format `{artifact_name}.{extension}`, where:
  - `artifact_name` is the name of the artifact produced by the pipeline stage
- Each pipeline function can produce one or more artifacts
- Artifacts are referenced by name in subsequent pipeline stages' parameters
- The temporary directory is automatically cleaned up after the pipeline completes, unless the `keep_cache` option is specified

### Example Pipeline

```python
@produces_artifacts(voice_only={"extension": "mp3"})
def voice_isolation(context: PipelineContext, transcode: Path) -> None:
    """Isolate voice from background noise."""
    output_path = context.get_artifact_path("voice_only")
    # Process audio...

@produces_artifacts(
    transcript={"extension": "srt"},
    timestamps={"extension": "json"}
)
def transcribe(context: PipelineContext, voice_only: Path) -> None:
    """Transcribe audio and produce transcript with timestamps."""
    transcript_path = context.get_artifact_path("transcript")
    timestamps_path = context.get_artifact_path("timestamps")
    # Generate transcript...
```

### Caching Behavior

- Artifacts are stored in a temporary directory created for each pipeline run
- Each artifact is stored with a simple filename based on the artifact name
- The temporary directory is deleted after the pipeline completes unless the `keep_cache` option is set
- Processing modules don't handle caching; they always process their inputs
- Intermediate files within a stage should use the temporary directory
- Cache lookup strategy:
  1. Check if the artifact already exists in the temporary directory
  2. If found, use the existing artifact instead of reprocessing
  3. If not found, process the stage and store the result in the temporary directory

## Detailed Operations

1. **Audio Channel Transcoding:**
   ```python
   def transcode(context: PipelineContext, input_path: Path) -> None:
       """Transcode an audio/video file to an audio file."""
   ```

2. **Voice Isolation:**
   ```python
   def voice_isolation(context: PipelineContext, transcode: Path) -> None:
       """Isolate voice from background noise."""
   ```

3. **Transcription:**
   ```python
   def transcribe(context: PipelineContext, voice_isolation: Path) -> None:
       """Transcribe audio to text and produce an SRT file."""
   ```

4. **Translation:**
   ```python
   def translate(context: PipelineContext, transcribe: Path) -> None:
       """Translate text and generate pronunciation guide."""
   ```

5. **Card Deck Generation:**
   ```python
   def generate_deck(
       context: PipelineContext,
       voice_isolation: Path,
       transcribe: Path,
       translation_path: Path,
       pronunciation_path: Path,
   ) -> None:
       """Generate Anki flashcard deck."""
   ```

## Progress Tracking

Progress tracking is integrated into the pipeline through the `PipelineProgress` class:
- Each stage automatically gets progress tracking based on its function name
- Progress updates are accessible through the context object
- Supports both overall pipeline progress and individual stage progress

## Error Handling

The pipeline includes several layers of error handling:
1. **Static Validation:** Catches missing artifact errors before execution
2. **Runtime Errors:** Each stage has specific error handling
3. **Progress Updates:** Error states are reflected in progress tracking

## CLI Integration

The command-line interface provides:
- Full pipeline execution
- Individual stage execution
- Progress display
- Error reporting

## Testing

The artifact-aware design improves testability:
- Each stage can be tested in isolation
- Artifacts can be mocked or replaced
- Pipeline validation can be tested separately
- Progress tracking can be mocked

## Future Considerations

1. **Type Safety:**
   - Consider using generics for stronger artifact typing
   - Add specific types for different artifact categories

2. **Error Handling:**
   - Add specific exception types for different pipeline errors
   - Improve error messages and recovery options

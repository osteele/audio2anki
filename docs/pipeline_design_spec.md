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

### Artifact Naming and Caching

- Each artifact has a unique name, either:
  - Specified in the `@pipeline_function` decorator, or
  - Defaulting to the function name if not decorated
- Each pipeline stage specifies the names of its input artifacts through parameter names (other than `context`)
- The artifact filename has the format `{sanitized_input_name}_{artifact_name}_{hash}.{extension}`, where:
  - `sanitized_input_name` is derived from the original input file to the pipeline
  - `artifact_name` is the name of the artifact produced by the pipeline stage
  - `hash` is derived from the inputs to the pipeline stage:
    - If the pipeline stage has a single input artifact, hash is the first eight characters of the md5 hash of the input content
    - If the pipeline stage has multiple input artifacts, the full md5 hashes of all input contents are concatenated and hashed again, with the first eight characters used
- Each pipeline function can produce one or more artifacts
- Artifacts are referenced by name in subsequent pipeline stages' parameters

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

- Artifacts are cached based on their unique names and content-based hashes
- The hash is computed from the content of the input artifacts, not just their names
- If a cached artifact exists with the matching hash, the stage is skipped
- Processing modules don't handle caching; they always process their inputs
- Intermediate files within a stage should use temporary files
- Cache lookup strategy:
  1. Compute hash from input artifact content
  2. Look for artifact with the filename pattern `{sanitized_input_name}_{artifact_name}_{hash}.{extension}`
  3. If found, use the cached artifact instead of reprocessing
  4. If not found, process the stage and cache the result with the hash-based filename

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

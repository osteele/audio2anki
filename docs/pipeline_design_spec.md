# Audio-to-Anki Pipeline Architecture Specification

**Author:** Oliver Steele (GitHub: [osteele](https://github.com/osteele))
**Year:** 2024

---

## Overview

This document describes a generalized pipeline architecture for the audio-to-anki application, which processes audio and video inputs to generate Anki card decks. The design uses an artifact-aware pipeline where each stage explicitly declares its inputs and outputs. This approach improves type safety, testability, and makes data dependencies clear.

## Pipeline Architecture

The pipeline is implemented as a sequence of Python functions, where each function:
1. Explicitly declares its inputs through parameter names
2. Produces artifacts that become inputs for subsequent stages
3. Receives a context object containing pipeline-wide configuration and progress tracking

### Key Features

- **Explicit Data Flow:** Each function's inputs and outputs are clearly defined through its signature
- **Static Validation:** The pipeline validates all required artifacts are available before execution
- **Type Safety:** Comprehensive type hints throughout the codebase
- **Progress Tracking:** Integrated progress reporting for each pipeline stage
- **Error Handling:** Clear error messages when required artifacts are missing

## Pipeline Operations

Each operation in the pipeline has the following structure:

```python
def operation_name(
    context: PipelineContext,
    required_artifact1: Path,
    required_artifact2: str,
    optional_param: int | None = None,
) -> None:
    """Process artifacts.
    
    Args:
        context: Pipeline-wide configuration and progress
        required_artifact1: First required input
        required_artifact2: Second required input
        optional_param: Optional configuration
    """
```

### Core Components

1. **PipelineContext:**
   ```python
   @dataclass
   class PipelineContext:
       """Holds pipeline state and configuration."""
       progress: PipelineProgress
       source_language: str = "chinese"
       target_language: str | None = None
   ```

2. **Pipeline Runner:**
   - Manages the execution of pipeline stages
   - Tracks artifacts in a dictionary
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

### Detailed Operations

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

## Artifact Management

### Naming Convention
- Artifacts are named using the pattern `{basename}_{stage}_{hash}.{ext}`
  - `basename` is derived from the input file (e.g., "dialog" from "Dialogue.mp3")
  - `stage` is either the pipeline function name or a key specified in `get_artifact_path()`
  - `hash` is an MD5 hash of the input file content
  - `ext` is the appropriate file extension for the artifact type

### Path Generation
The `PipelineContext` provides two ways to get artifact paths:
1. `artifact_path` property - For pipeline functions that produce a single artifact
2. `get_artifact_path(key)` - For pipeline functions that produce multiple artifacts

Pipeline functions are decorated with `@produces_artifacts` to specify their outputs:
```python
# Single artifact
@produces_artifacts(output=Path)
def transcode(context: PipelineContext, input_path: Path) -> None:
    output_path = context.artifact_path
    # Process input_path to output_path

# Multiple artifacts
@produces_artifacts(translation=Path, pronunciation=Path)
def translate(context: PipelineContext, input_path: Path) -> None:
    translation_path = context.get_artifact_path('translation')
    pronunciation_path = context.get_artifact_path('pronunciation')
    # Process input_path to both output paths
```

### Caching
- The pipeline runner checks for existing artifacts before executing each stage
- If a cached artifact exists with the expected name, the stage is skipped
- Processing modules don't handle caching; they always process their inputs
- Intermediate files within a stage should use temporary files

### Stage-Specific Context
Each pipeline function receives a stage-specific context created by `context.for_stage(stage_name)`. This context:
- Validates artifact path access is within a pipeline stage
- Ensures `artifact_path` is only used for single-artifact stages
- Validates `get_artifact_path()` keys match the stage's `@produces_artifacts` declaration

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

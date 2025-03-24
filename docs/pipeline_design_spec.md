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
- **Caching:** Two-level caching system with both temporary and persistent caches

## Pipeline Operations

Each operation in the pipeline has the following structure:

```python
@pipeline_function(output_name={"extension": "mp3", "cache": True, "version": get_version})
def operation_name(
    context: PipelineContext,
    required_artifact1: Path,  # Name matches a previous stage's artifact name
    optional_param: Path | None = None,
) -> None:
    """Process artifacts.

    Args:
        context: Pipeline-wide configuration and progress
        required_artifact1: Required input artifact
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
       source_language: LanguageCode | None = None
       target_language: LanguageCode | None = None
       output_folder: Path | None = None
       translation_provider: TranslationProvider = TranslationProvider.OPENAI
       _current_fn: PipelineFunction | None = None
       _input_file: Path | None = None
       _stage_inputs: dict[str, Path] = field(default_factory=dict)
       _artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
   ```

2. **Pipeline Options:**
   ```python
   @dataclass
   class PipelineOptions:
       """Options that control pipeline behavior."""
       debug: bool = False
       source_language: LanguageCode | None = None
       target_language: LanguageCode | None = None
       output_folder: Path | None = None
       skip_voice_isolation: bool = False
       translation_provider: TranslationProvider = TranslationProvider.OPENAI
       use_artifact_cache: bool = True
       skip_cache_cleanup: bool = False
   ```

3. **Progress Tracking:**
   ```python
   @dataclass
   class PipelineProgress:
       """Manages progress tracking for the pipeline and its stages."""
       progress: Progress  # rich.progress.Progress
       pipeline_task: TaskID
       console: Console
       current_stage: str | None = None
       stage_tasks: dict[str, TaskID] = field(default_factory=dict)
   ```

### Caching System

The pipeline implements a two-level caching system:

1. **Temporary Cache:**
   - Created fresh for each pipeline run
   - Stores intermediate artifacts during processing
   - Cleaned up after pipeline completion unless debug mode is enabled
   - Uses simple filenames based on artifact names

2. **Persistent Cache:**
   - Stores artifacts across pipeline runs
   - Version-aware to handle algorithm updates
   - Supports cache invalidation based on input changes
   - Automatically cleans up old artifacts (default: 14 days)
   - Configurable through artifact decorators:
     ```python
     # Simple usage with default artifact name (function name)
     @pipeline_function(extension="mp3", hash=get_version_function)
     
     # For terminal functions that don't produce artifacts
     @pipeline_function(artifacts=[])
     
     # With multiple artifacts
     @pipeline_function(artifacts=[
         {"name": "artifact1", "extension": "mp3", "cache": True, "version": 1},
         {"name": "artifact2", "extension": "json", "hash": get_hash_function}
     ])
     ```

### Pipeline Stages

The standard pipeline includes these stages:

1. **Audio Transcoding:**
   ```python
   @pipeline_function(extension="mp3", hash=get_transcode_version)
   def transcode(context: PipelineContext, input_path: Path) -> None
   ```

2. **Voice Isolation:**
   ```python
   @pipeline_function(extension="mp3", hash=get_voice_isolation_version)
   def voice_isolation(context: PipelineContext, transcode: Path) -> None
   ```

3. **Transcription:**
   ```python
   @pipeline_function(extension="srt", hash=get_transcription_version)
   def transcribe(context: PipelineContext, voice_isolation: Path | None = None, transcode: Path | None = None) -> None
   ```

4. **Translation:**
   ```python
   @pipeline_function(extension="json", hash=get_translation_version)
   def translate(context: PipelineContext, transcribe: Path) -> None
   ```

5. **Deck Generation:**
   ```python
   @pipeline_function(artifacts=[])
   def generate_deck(context: PipelineContext, segments: Path, ...) -> None
   ```

### Error Handling

The pipeline includes several layers of error handling:
1. **Static Validation:** Catches missing artifact errors before execution
2. **Runtime Errors:** Each stage has specific error handling
3. **Progress Updates:** Error states are reflected in progress tracking
4. **Error Classification:**
   - `SYSTEM_ERROR`: General system errors
   - `SERVICE_ERROR`: Network/service connection issues
   - `VALIDATION_ERROR`: Input validation failures

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

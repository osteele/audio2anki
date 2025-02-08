# Audio-to-Anki Pipeline Architecture Specification

**Author:** Oliver Steele (GitHub: [osteele](https://github.com/osteele))
**Year:** 2024

---

## Overview

This document describes a generalized pipeline architecture for the audio-to-anki application, which processes audio and video inputs to generate Anki card decks. The design codifies each stage of the processing as an independent operation within a configurable list, and integrates a central caching mechanism and configuration system. This modular approach improves auditability, flexibility, and future extensibility.

## Pipeline Architecture

The application pipeline is reified as an ordered list of operations. Each operation represents a distinct processing stage, such as:

- **Audio Channel Transcoding:** Includes extraction of an audio channel from a video file.
- **Voice Isolation:** Noise removal and audio cleaning.
- **Transcription:** Converting audio to text.
- **Audio File Slicing:** Dividing the audio into manageable segments.
- **Card Deck Generation:** Generating the Anki flashcard deck from processed data.

### Key Features

- **Modularity:** Each operation is implemented as a self-contained unit that adheres to a common interface, allowing for dynamic composition of the pipeline via CLI options.
- **Progress Reporting:** The pipeline maintains and displays progress by enumerating the list of operations and their status (pending, in-progress, completed, or failed).
- **Caching Integration:** Intermediate results from each operation are cached through a centralized caching mechanism, enabling efficient re-runs and recovery from failures.

## Pipeline Operations

Each operation in the pipeline has the following structure:

- **Name:** A human-readable identifier (e.g., 'Transcoding', 'Voice Isolation').
- **Function:** The executable code for the operation.
- **Dependencies:** Any prerequisites or limits (e.g., maximum file size, duration) that are enforced before execution.
- **Output Handling:** Intermediate results are stored and retrieved from the cache if available.

### Detailed Operations

1. **Audio Channel Transcoding:**
   - Transcode an audio or video file to an audio file that is suitable for downstream processing.

2. **Voice Isolation:**
   - Performs noise removal and audio cleaning.
   - Offers two provider options: one wrapping DEMUCS to run locally and one leveraging the Eleven Labs API.

3. **Transcription:**
   - Converts audio to text and produces an SRT file.
   - Users can optionally supply an existing SRT file to bypass this stage for intermediate testing.

4. **Audio File Slicing:**
   - Splits the audio into segments.
   - Multiple instances can be configured with different file size and duration settings to accommodate provider restrictions.

5. **Optional Sentence Selection:**
   - An optional stage that processes the transcript to select sentences.
   - Strategies include selecting all sentences, filtering for a target language, or using an LLM service for selection.

6. **Card Deck Generation:**
   - Generates Anki flashcard decks from the processed data. A flashcard deck is stored as a directory with a tab-separated `deck.txt`, a folder of media files, a README that describes how to import the deck, and an alias or shortcut to the location where the media files should be copied. Media files have names that include a hash of the source file to prevent conflicts when importing multiple decks.

CLI options can be provided to modify the set or order of operations. For example, a user might skip certain stages (e.g., voice isolation) or insert custom operations.

## Caching Mechanism

A central caching mechanism has the following features:

- **Standard Location:** The cache is stored at `$HOME/.cache/audio2anki`, where intermediate results are cached.
- **Size and Expiry Management:** The cache system will support commands to report its location and size, as well as expire old entries. Expiration settings will be configurable.
- **Disable Option:** Users may disable caching via a CLI option if desired.
- **Automatic Expiry:** When using the cache, entries older than a configured duration are automatically expired.

## Configuration System

The application will use a configuration file stored at `$HOME/.config/audio2anki/config.toml`. Features include:

- **Central Config File:** It contains settings governing the default behavior of the pipeline (e.g., file cleaning defaults, cache usage, cache expiry durations, provider selection, etc.).
- **CLI Options:** Commands to display the config file location, to modify configuration settings, or to open the config file in an editor.
- **Dynamic Adjustments:** Changes made via CLI are applied immediately to subsequent runs.

## CLI Integration

The command-line interface will provide the following functionalities:

**Pipeline Execution Options:**
- Run the entire pipeline or select specific stages (e.g., only voice isolation, transcription, or card generation) via CLI flags.

**Provider Selection:**
- Choose specific providers for stages that support multiple implementations (e.g., choose between local DEMUCS or Eleven Labs API for voice isolation).

**SRT File Input:**
- Supply an existing SRT file to bypass the transcription stage when needed.

**Progress Display:**
- Show real-time progress as each operation is initiated and completed.

**Cache Management:**
- Provide commands to print cache information (location and size), expire old entries, or delete the entire cache.

**Configuration Management:**
- Commands to display, edit, and update the configuration file interactively.

**Testing and Linting:**
- Integration with the tooling commands (e.g., `just check`, `just test`) to run automated quality checks.

## Extensibility Considerations

- **Dynamic Operation Lists:** Future extensions could allow plug-in operations by scanning designated directories for additional pipeline modules.
- **Custom Attributes:** Each operation may include metadata (such as estimated processing time and resource requirements) to better plan execution.

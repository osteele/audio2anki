# Implementation Plan for Audio-to-Anki Pipeline

This document details a step-by-step implementation plan to achieve an end-to-end processing pipeline for the Audio-to-Anki application, targeting short files that don't require splitting. This uses the following:

- **Voice Isolation:** Eleven Labs API
- **Transcription:** OpenAI Whisper service (with SRT output)
- **Sentence Selection:** Simple (basic sentence splitting with minimal filtering)

The plan is split into manageable steps so that each can be developed, tested, and committed separately.

---

## Step 1: Establish the Pipeline Framework

- Create a new entry point (e.g., `src/audio2anki/main.py`) which:
  - Loads configuration from `$HOME/.config/audio2anki/config.toml` (with sensible defaults).
  - Initializes the centralized cache located at `$HOME/.cache/audio2anki`.
  - Constructs an ordered list (pipeline) of operations. Initial stages:
    1. Audio Channel Transcoding
    2. Voice Isolation
    3. Transcription
    4. (Optional) Sentence Selection
    5. Card Deck Generation
  - Displays progress throughout the pipeline execution.
  - Provides CLI flags to override the stages list (default order is sufficient for now).

## Step 2: Implement the Temporary Caching Mechanism

- Develop a caching module (e.g., `src/audio2anki/cache.py`) that:
  - Creates a temporary directory for each pipeline run using `tempfile.mkdtemp()`.
  - Supports storing and retrieving artifacts by their simple names within this temporary directory.
  - Automatically cleans up the temporary directory after processing is complete, with an option to keep it for debugging.

## Step 3: Set Up the Configuration Module

- Create a configuration module (e.g., `src/audio2anki/config.py`) that:
  - Reads a TOML file at `$HOME/.config/audio2anki/config.toml` and applies defaults if missing.
  - Stores settings such as:
    - File cleaning defaults
    - Cache usage defaults
    - Provider selections (preselect Eleven Labs for voice isolation and OpenAI Whisper for transcription)
  - Offers CLI commands or a separate module to show and edit the configuration.

## Step 4: Develop Pipeline Operation Modules

For each stage, create separate modules or classes so that each can be developed and tested independently:

1. **Audio Channel Transcoding Module**
   - **Module:** `src/audio2anki/transcode.py`
   - **Function:** Accept an audio or video file and use a tool (e.g., `ffmpeg`) to extract/convert to an audio file suitable for downstream processing.
   - **Testing:** Process a known small file and verify output is a correct audio file.

2. **Voice Isolation Module**
   - **Module:** `src/audio2anki/voice_isolation.py`
   - **Function:** Invoke the Eleven Labs API to perform noise removal and cleaning on the audio.
   - **Testing:** Verify the API call processes a short file and returns the cleaned audio.

3. **Transcription Module**
   - **Module:** `src/audio2anki/transcribe.py`
   - **Function:** Integrate with the OpenAI Whisper service to transcribe the cleaned audio file.
   - **Output:** Generate an SRT file along with a transcript.
   - **Testing:** Provide a small audio clip and validate both the transcript and generated SRT file.

4. **Optional Sentence Selection Module**
   - **Module:** `src/audio2anki/sentence_selection.py`
   - **Function:** Implement a basic strategy to split the transcript into sentences (and optionally filter by target language).
   - **Testing:** Verify that the module returns a sensible list of sentences when given a sample transcript.

5. **Card Deck Generation Module**
   - **Module:** `src/audio2anki/deck.py`
   - **Function:** Generate an Anki flashcard deck from the processed data. The deck should be stored as a directory containing:
     - A tab-separated `deck.txt`
     - A folder of media files
     - A README describing how to import the deck
     - An alias or shortcut specifying the media files location
     - Media file names should include a hash of the source file to avoid naming conflicts.
   - **Testing:** Generate a deck from a known transcript and verify the output structure.

## Step 5: Integrate and Test End-to-End Processing

- In `main.py`, wire up the pipeline so that each stage feeds its output to the next, skipping the audio slicing stage for short files:
  - Store intermediate results in the temporary directory.
  - Implement basic CLI commands to process a sample short file.
- Add integration tests (e.g., within the tests directory) that run an end-to-end scenario for a short file, ensuring a complete deck is generated correctly.

## Step 6: Incremental Commits and Testing

- Develop and commit each new module separately.
- Write unit tests for each module to ensure they function correctly in isolation.
- Integrate modules in `main.py` and perform end-to-end integration tests.
- Add logging statements to trace the status of each stage during execution.
- Ensure continuous integration for smooth rollout of each milestone.

---

### Summary

This implementation plan begins with establishing a modular pipeline framework and then incrementally adds and tests each processing stage. The initial target is to achieve an end-to-end process for short files, using the Eleven Labs API for voice isolation, the OpenAI Whisper service for transcription (with SRT output), a basic sentence selection strategy, and finally deck generation. Each step is designed to be developed, tested, and committed independently to ensure robust integration and easy rollback if needed.

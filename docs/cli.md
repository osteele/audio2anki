# Command Line Interface

`audio2anki` provides a command-line interface with several subcommands for processing audio files and managing configuration.

## Main Command

```bash
audio2anki [OPTIONS] [INPUT_FILE]
```

```text
audio2anki
├── process [INPUT_FILE]  # Default command for processing audio files
└── config
    ├── edit
    ├── set
    ├── list
    └── reset
```

### Global Options

- `--debug`: Enable debug logging
- `--bypass-cache`: Skip cache lookup and force reprocessing
- `--keep-cache`: Keep the temporary cache directory after processing (for debugging)
- `--target-language`: Target language for translation (default: system language)
- `--source-language`: Source language for transcription (default: chinese)

## Configuration Management

```text
# Configuration
audio2anki config edit              # Open config in editor
audio2anki config set use_cache true
audio2anki config list             # Show all settings
audio2anki config reset            # Reset to defaults
```

```bash
audio2anki config COMMAND [ARGS]
```

### Commands

- `edit`: Open configuration file in default editor
  ```bash
  audio2anki config edit
  ```

- `set`: Set a configuration value
  ```bash
  audio2anki config set KEY VALUE
  ```
  Example: `audio2anki config set use_cache true`

- `list`: Show all configuration settings
  ```bash
  audio2anki config list
  ```

- `reset`: Reset configuration to default values
  ```bash
  audio2anki config reset
  ```

### Configuration Options

- `clean_files`: Remove temporary files after processing (default: true)
- `use_cache`: Enable caching of processed files (default: true)
- `voice_isolation_provider`: Provider for voice isolation (default: "eleven_labs")
- `transcription_provider`: Provider for transcription (default: "openai_whisper")

## Examples

1. Process an audio file with custom language settings:
   ```bash
   audio2anki input.mp3 --source-language japanese --target-language english
   ```

2. Process a file with cache disabled:
   ```bash
   audio2anki input.mp3 --bypass-cache
   ```

3. Process a file and keep the temporary cache directory for debugging:
   ```bash
   audio2anki input.mp3 --keep-cache
   ```

4. Change configuration settings:
   ```bash
   audio2anki config set use_cache true
   ```
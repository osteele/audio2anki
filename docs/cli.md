# Command Line Interface

`audio2anki` provides a command-line interface with several subcommands for processing audio files, managing configuration, and handling the cache.

## Main Command

```bash
audio2anki [OPTIONS] [INPUT_FILE]
```

```text
audio2anki
├── process [INPUT_FILE]  # Default command for processing audio files
├── config
│   ├── edit
│   ├── set
│   ├── list
│   └── reset
└── cache
    ├── open
    ├── clear
    └── info
```

### Global Options

- `--debug`: Enable debug logging
- `--bypass-cache`: Skip cache lookup and force reprocessing
- `--clear-cache`: Clear the cache before processing
- `--target-language`: Target language for translation (default: system language)
- `--source-language`: Source language for transcription (default: chinese)

## Configuration Management

```text
# Configuration
audio2anki config edit              # Open config in editor
audio2anki config set use_cache true
audio2anki config list             # Show all settings
audio2anki config reset            # Reset to defaults

# Cache
audio2anki cache info              # Show cache stats
audio2anki cache open              # Open in Finder
audio2anki cache clear            # Clear all cached data
The `config` subcommand manages application settings.
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
- `cache_expiry_days`: Number of days before cache entries expire (default: 7)
- `voice_isolation_provider`: Provider for voice isolation (default: "eleven_labs")
- `transcription_provider`: Provider for transcription (default: "openai_whisper")

## Cache Management

The `cache` subcommand manages the application's cache of processed files.

```bash
audio2anki cache COMMAND
```

### Commands

- `info`: Display information about the cache
  ```bash
  audio2anki cache info
  ```
  Shows cache size, number of files, and last modification time.

- `open`: Open cache directory in system file explorer
  ```bash
  audio2anki cache open
  ```

- `clear`: Clear all cached data
  ```bash
  audio2anki cache clear
  ```

## Examples

1. Process an audio file with custom language settings:
   ```bash
   audio2anki input.mp3 --source-language japanese --target-language english
   ```

2. Process a file with cache disabled:
   ```bash
   audio2anki input.mp3 --bypass-cache
   ```

3. Change cache settings:
   ```bash
   audio2anki config set cache_expiry_days 14
   audio2anki config set use_cache true
   ```

4. Clear cache and process a file:
   ```bash
   audio2anki cache clear
   audio2anki input.mp3
   ```

# audio2anki

[![PyPI version](https://img.shields.io/pypi/v/audio2anki.svg)](https://pypi.org/project/audio2anki/)
[![Python Version](https://img.shields.io/pypi/pyversions/audio2anki.svg)](https://pypi.org/project/audio2anki/)
[![License](https://img.shields.io/github/license/osteele/audio2anki.svg)](https://github.com/osteele/audio2anki/blob/main/LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/osteele/audio2anki/actions/workflows/ci.yml/badge.svg)](https://github.com/osteele/audio2anki/actions/workflows/ci.yml)

Convert audio and video files into Anki flashcard decks with translations.

`audio2anki` helps language learners create study materials from audio and video content. It automatically:
- Transcribes audio using OpenAI Whisper
- Segments the audio into individual utterances
- Translates each segment using OpenAI or DeepL
- Generates pronunciation (currently supports pinyin for Mandarin)
- Creates Anki-compatible flashcards with audio snippets

![audio2anki](./docs/preview.png)

For related language learning resources, visit [Oliver Steele's Language
Learning Resources](https://osteele.com/topics/language-learning/).

## Features

- 🎵 Process audio files (mp3, wav, etc.) and video files
- 🤖 Automatic transcription using OpenAI Whisper
- 🔤 Automatic translation and pronunciation
- ✂️ Smart audio segmentation
- 📝 Optional manual transcript input
- 🎴 Anki-ready output with embedded audio

## Requirements

- Python 3.10 or later
- OpenAI API key (set as `OPENAI_API_KEY` environment variable) or DeepL API token (set as `DEEPL_API_TOKEN` environment variable)

## Installation

You can install audio2anki using either `uv`, `pipx`, or `pip`:

### Using uv

1. [Install `uv`](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it already.
2. Install `audio2anki`:
   ```bash
   uv tool install audio2anki
   ```

### Using pipx

1. [Install `pipx`](https://pipx.pypa.io/stable/installation/) if you don't have it already.

2. Install `audio2anki`:
   ```bash
   pipx install audio2anki
   ```

### Using pip (not recommended)

This method doesn't require a third-party tool, but it is not recommended as it will install audio2anki in the current Python environment, which may cause conflicts with other packages.

```bash
pip install audio2anki
```

## Usage

### Basic Usage

Create an Anki deck from an audio file:
```bash
export OPENAI_API_KEY=your-api-key-here
audio2anki audio.mp3
```

Use an existing transcript:
```bash
export OPENAI_API_KEY=your-api-key-here
audio2anki audio.mp3 --transcript transcript.txt
```

Specify which translation service to use:
```bash
# Use OpenAI for translation (default)
audio2anki audio.mp3 --translation-provider openai

# Use DeepL for translation
export DEEPL_API_TOKEN=your-deepl-token-here
audio2anki audio.mp3 --translation-provider deepl
```

For a complete list of commands, including cache and configuration management, see the [CLI documentation](./docs/cli.md).

### Common Use Cases

Process a noisy recording with more aggressive silence removal:
```bash
audio2anki audio.mp3 --silence-thresh -30
```

Process a quiet recording or preserve more background sounds:
```bash
audio2anki audio.mp3 --silence-thresh -50
```

Process a podcast with custom segment lengths and silence detection:
```bash
audio2anki podcast.mp3 --min-length 2.0 --max-length 20.0 --silence-thresh -35
```

### Command Line Options

```bash
audio2anki <input-file> [options]

Options:
  --transcript FILE    Use existing transcript
  --output DIR        Output directory (default: ./output)
  --model MODEL       Whisper model (tiny, base, small, medium, large)
  --debug            Generate debug information
  --min-length SEC   Minimum segment length (default: 1.0)
  --max-length SEC   Maximum segment length (default: 15.0)
  --language LANG    Source language (default: auto-detect)
  --silence-thresh DB Silence threshold (default: -40)
  --translation-provider {openai,deepl}  Translation service to use (default: openai)
```

### Environment Variables

Required:
- `OPENAI_API_KEY` - OpenAI API key (required if DeepL is not used)

Optional:
- `DEEPL_API_TOKEN` - DeepL API key (recommended for higher quality translations)

### Translation Services

The tool supports two translation services:

1. **DeepL**
   - Higher quality translations, especially for European languages
   - Get an API key from [DeepL Pro](https://www.deepl.com/pro-api)
   - Set environment variable: `export DEEPL_API_TOKEN=your-api-key`
   - Use with: `--translation-provider deepl`

2. **OpenAI** (Default)
   - Used by default or when DeepL is not configured or fails
   - Get an API key from [OpenAI](https://platform.openai.com/api-keys)
   - Set environment variable: `export OPENAI_API_KEY=your-api-key`
   - Use with: `--translation-provider openai`

Note: OpenAI is always used for generating pronunciations (Pinyin, Hiragana), even when DeepL is selected for translation.

### Output

The script creates:
1. A tab-separated deck file (`deck.txt`) containing:
   - Original text (e.g., Chinese characters)
   - Pronunciation (e.g., Pinyin with tone marks)
   - English translation
   - Audio reference
2. A `media` directory containing the audio segments

### Importing into Anki

1. **Import the Deck**:
   - Open Anki
   - Click `File` > `Import`
   - Select the generated `deck.txt` file
   - In the import dialog:
     - Set the Type to "Basic"
     - Check that fields are mapped correctly:
       - Field 1: Front (Original text)
       - Field 2: Pronunciation
       - Field 3: Back (Translation)
       - Field 4: Audio
     - Set "Field separator" to "Tab"
     - Check "Allow HTML in fields"

2. **Import the Audio**:
   - Copy all files from the `media` directory
   - Paste them into your Anki media collection:
     - On Mac: [~/Library/Application Support/Anki2/User 1/collection.media](file:///Users/$(whoami)/Library/Application%20Support/Anki2/User%201/collection.media)
     - On Windows: [%APPDATA%\Anki2\User 1\collection.media](file:///C:/Users/%USERNAME%/AppData/Roaming/Anki2/User%201/collection.media)
     - On Linux: [~/.local/share/Anki2/User 1/collection.media](file:///home/$(whoami)/.local/share/Anki2/User%201/collection.media)

3. **Verify the Import**:
   - The cards should show:
     - Front: Original text
     - Back: Pronunciation, translation, and a play button for audio
   - Test the audio playback on a few cards

**Note**: The audio filenames include a hash of the source file to prevent conflicts when importing multiple decks.

## Development

```bash
just check  # Run linting and type checking
just test   # Run tests
```

## Related Projects

- [add2anki](https://github.com/osteele/add2anki) - Convert vocabulary lists to Anki flashcard decks

## License

MIT License 2024 Oliver Steele

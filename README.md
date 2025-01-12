# audio2anki

Convert audio and video files into Anki flashcard decks with translations.

`audio2anki` helps language learners create study materials from audio and video content. It automatically:
- Transcribes audio using OpenAI Whisper
- Segments the audio into individual utterances
- Translates each segment using OpenAI
- Generates pronunciation (currently supports pinyin for Mandarin)
- Creates Anki-compatible flashcards with audio snippets

## Features

- 🎵 Process audio files (mp3, wav, etc.) and video files
- 🤖 Automatic transcription using OpenAI Whisper
- 🔤 Automatic translation and pronunciation
- ✂️ Smart audio segmentation
- 📝 Optional manual transcript input
- 🎴 Anki-ready output with embedded audio

## Installation

```bash
uv venv
uv pip install -e .
```

## Usage

### Basic Usage

Create an Anki deck from an audio file:
```bash
audio2anki audio.mp3
```

Use an existing transcript:
```bash
audio2anki audio.mp3 --transcript transcript.txt
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
```

### Output

The tool creates:
1. A CSV file ready for Anki import
2. A directory of audio snippets
3. (Optional) A debug file with detailed segment information

The CSV includes fields for:
- Source text (e.g., Hanzi for Mandarin)
- Pronunciation (e.g., Pinyin for Mandarin)
- English translation
- Audio filename

## Development

```bash
just check  # Run linting and type checking
just test   # Run tests
```

## License

MIT License 2024 Oliver Steele

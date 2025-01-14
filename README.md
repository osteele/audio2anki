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

## Requirements

- Python 3.10 or later
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)

## Installation

```bash
uv venv
uv pip install -e .
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

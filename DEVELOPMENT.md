# Development Guide for audio2anki

## Setup

1. [Install `uv`](https://docs.astral.sh/uv/getting-started/installation/)
2. [Install `just`](https://just.systems/man/en/pre-built-binaries.html)
3. Clone the repository and set up the development environment:

```bash
# Clone the repository
git clone https://github.com/osteele/audio2anki.git
cd audio2anki

# Install dependencies
just setup

# Install pre-commit hooks
uv run --dev pre-commit install
```

## Development Commands

```bash
# Run tests
just test

# Format code
just format

# Run type checking
just typecheck

# Run linting
just lint

# Fix linting issues automatically
just fix

# Run all checks (lint, typecheck, test)
just check
```

## Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to run code formatting before each commit.
The hooks will automatically run formatting tools to ensure code quality.

To install the pre-commit hooks:

```bash
uv run --dev pre-commit install
```

After installation, the hooks will run automatically on each commit.

## Publishing

To publish a new version to PyPI:

```bash
just publish
```

This will clean build artifacts, build the package, and publish it to PyPI.

## CI/CD

The project uses GitHub Actions for continuous integration. The workflow runs:
- Linting with ruff
- Type checking with pyright
- Tests with pytest

The workflow runs on multiple Python versions (3.10, 3.11, 3.12) to ensure compatibility.

## Development Utilities

### Cache Bypass for Pipeline Stages

When developing or debugging specific pipeline stages, you can bypass the cache for those stages using the `--bypass-cache-for` option:

```bash
# Bypass cache for the transcribe stage
audio2anki process input.mp3 --bypass-cache-for transcribe

# Bypass cache for multiple stages
audio2anki process input.mp3 --bypass-cache-for "transcribe,translate"

# Alternative syntax for multiple stages
audio2anki process input.mp3 --bypass-cache-for transcribe --bypass-cache-for translate
```

This option is hidden from standard help output since it's intended for development use. It forces the specified pipeline stages to run from scratch, ignoring any cached results.

Available pipeline stages:
- `transcode`: Audio/video conversion
- `voice_isolation`: Voice isolation (if enabled)
- `transcribe`: Speech-to-text transcription
- `translate`: Text translation

## Testing

The project includes comprehensive test coverage using pytest. To run tests:

```bash
just test
```

### Testing with Mock Services

The test suite uses mock implementations of the external API services (OpenAI and ElevenLabs) to avoid making real API calls during testing. This has several benefits:

1. Tests run faster without network calls
2. Tests don't incur API usage costs
3. Tests are more reliable and deterministic
4. No API keys are required to run tests

The mock service system works as follows:

- **Service Interfaces**: Abstract base classes for `TranscriptionService`, `TranslationService`, and `VoiceIsolationService`
- **Mock Implementations**: Mock implementations of these services with configurable responses
- **Service Provider**: A central provider that returns either real or mock services based on configuration
- **Test Data**: JSON files with predefined mock responses for common test cases

To run tests with mock services:

```bash
# Tests use mock services by default, but this can be explicitly set
AUDIO2ANKI_TEST_MODE=true just test
```

To add new mock responses for testing:

1. Edit `tests/data/mock_responses.json` to add new transcriptions, translations, or readings
2. Or programmatically add responses in test fixtures:

```python
def test_custom_transcription(mock_transcription_service):
    mock_transcription_service.add_response(
        "test.mp3",
        [
            {"start": 0.0, "end": 1.0, "text": "Custom test text"},
        ],
    )
    # Test with this custom response
```

## Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to run code formatting before each commit.
The hooks will automatically run formatting tools to ensure code quality.

To install the pre-commit hooks:

```bash
uv run --dev pre-commit install
```

After installation, the hooks will run automatically on each commit.

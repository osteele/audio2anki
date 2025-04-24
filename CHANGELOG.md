# Changelog

All notable changes to audio2anki will be documented in this file.

## [Unreleased]

### Added
- Improved audio segment extraction with padding and silence detection
- Added configuration options for silence threshold and audio padding
- Added deck summary display showing timestamps and content after deck creation
- Replaced `skip-voice-isolation` to disable voice isolation, with `--voice-isolation` to enable it. Voice isolation is
  now disabled by default.
- API errors during voice isolation now display as user-facing errors
- Added intelligent sentence selection/filtering system:
  - Filters out one-word segments, sentences ending with commas, and duplicates
  - Detects and filters by language to ensure consistent decks
  - Smart handling of multi-character CJK text and punctuation
- Enhanced add2anki integration:
  - Checks for add2anki in PATH and verifies it meets version requirement (>=0.1.2).
  - If add2anki is present but outdated, and uv is not available, prints upgrade instructions.
  - If add2anki is missing or too old and uv is available, offers to use uv as a fallback.
  - Adds ADD2ANKI_MIN_VERSION constant for version management.
  - The deck's `import_to_anki.sh` script is now smarter: it checks for `add2anki` at version >=0.1.2 and uses it if available, falling back to `uv` otherwise. If `add2anki` is present but too old, it prints an upgrade message. This logic was moved from the README to the script itself.
  - Updated README with import instructions, version check, and upgrade/fallback guidance.
  - Updated deck README.md to reference add2anki and uv options.
- Usage tracking records the number of minutes of audio processed for each API/model, in addition to tokens and
  character cost.

### Fixed
- Persistent cache keys now use artifact basenames (which include content hashes) instead of full temp file paths, ensuring cache hits across runs regardless of temp directory changes.

## [0.1.2] - 2025-04-08

### Changed
- Removed automatic Anki launch
- Updated dependencies
- Moved package from src/ directory to project root

### Fixed
- Removed unused "Color" field from Chinese deck format
- Improved target language display

## [0.1.1] - 2025-03-23

### Added
- Persistent artifact caching system for better performance between runs
- Enhanced audio filename generation using text hashes for better uniqueness
- Added langcodes>=3.5.0 dependency for robust language code handling

### Changed
- Switched from string-based language identification to ISO language codes
- Improved deck generation by combining translations and transcriptions in JSON
- Updated import script to use the latest audio2anki version
- Updated project description to better reflect its purpose

### Fixed
- Fixed several bugs and improved error handling throughout the codebase

## [0.1.0] - 2025-01-14

### Added
- Initial release
- Convert audio and video files into Anki flashcard decks with translations
- OpenAI Whisper API integration for transcription
- DeepL API integration with fallback to OpenAI for translation
- Pinyin support for Chinese
- Silence trimming with configurable threshold
- Parallelized translation processing

[Unreleased]: https://github.com/osteele/audio2anki/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/osteele/audio2anki/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/osteele/audio2anki/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/osteele/audio2anki/releases/tag/v0.1.0

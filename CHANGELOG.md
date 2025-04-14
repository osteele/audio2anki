# Changelog

All notable changes to audio2anki will be documented in this file.

## [Unreleased]

### Added
- Improved audio segment extraction with padding and silence detection
- Added configuration options for silence threshold and audio padding

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

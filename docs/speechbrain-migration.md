# SpeechBrain Migration Roadmap

## Phase 1: Voice Isolation Replacement
**Current Implementation**: 
- Likely uses custom/simplified audio processing (needs investigation)
- Potential pain points: noise handling, speaker separation

**SpeechBrain Alternative**:
- Use `speechbrain.pretrained.VAD` for voice activity detection
- Leverage `speechbrain.processing.speech_enhancement` for noise reduction
- Maintain existing interface (`isolate_voice()` function)

**Benefits**:
- State-of-the-art pre-trained models
- Built-in GPU acceleration
- Better handling of background noise

## Phase 2: Transcription Enhancement
**Current Implementation**:
- Appears to use basic ASR (needs investigation of `transcribe.py`)

**SpeechBrain Alternative**:
- Implement `speechbrain.pretrained.TransformerASR`
- Add language detection for multilingual support

## Phase 3: Translation Integration
**Current Implementation**:
- Current translation approach needs investigation (`translate.py`)

**SpeechBrain Alternative**:
- Use `speechbrain.pretrained.Translator`
- Implement end-to-end speech translation

## Adoption Strategy
1. Maintain existing interfaces during transition
2. Feature-flag SpeechBrain implementations
3. Parallel validation of old/new implementations
4. Phase out legacy code after testing

## Dependency Plan
- Add SpeechBrain to `pyproject.toml` dependencies
- Update `justfile` with new test targets
- Document environment setup with UV

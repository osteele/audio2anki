"""Data models for audio2anki."""

import torch  # Required for device detection
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

try:
    from audio2anki.voice_isolation.speechbrain_impl import SpeechBrainVoiceIsolator
    HAS_SPEECHBRAIN = True
except ImportError:
    HAS_SPEECHBRAIN = False

@dataclass
class AudioSegment:
    """A segment of audio with text and optional translation."""
    start: float
    end: float
    text: str
    translation: str | None = None
    pronunciation: str | None = None
    audio_file: str | None = None

class VoiceIsolator(ABC):
    @abstractmethod
    def isolate_voice(self, input_path: Path, output_path: Path) -> None:
        """Process audio file to isolate voice"""
        pass

class LegacyVoiceIsolator(VoiceIsolator):
    def __init__(self, **kwargs):
        pass
    
    def isolate_voice(self, input_path: Path, output_path: Path) -> None:
        raise NotImplementedError("Legacy implementation not available in this prototype")

class AudioProcessor:
    @classmethod
    def create_voice_isolator(cls, config: dict) -> VoiceIsolator:
        impl_type = config.get('voice_isolation_impl', 
                             'speechbrain' if HAS_SPEECHBRAIN else 'legacy')
        
        if impl_type == 'speechbrain':
            device = config.get('torch_device', 
                              'cuda' if torch.cuda.is_available() else 'cpu')
            return SpeechBrainVoiceIsolator(device=device)
            
        return LegacyVoiceIsolator(**config.get('legacy_params', {}))

from pathlib import Path
import logging
import tempfile
import os
import subprocess
from speechbrain.inference import SepformerSeparation
import torchaudio
import torch
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

class SpeechBrainVoiceIsolator:
    MODEL_SOURCE = "speechbrain/sepformer-whamr"
    
    def __init__(self, device: str = 'cpu'):
        """Initialize SpeechBrain components with pretrained models"""
        # Configure speechbrain logger to only show warnings and above
        logging.getLogger('speechbrain').setLevel(logging.WARNING)
        
        self.device = torch.device(device)
        
        # Use XDG_CACHE_HOME if available, otherwise use user's home directory
        if cache_dir := os.environ.get("XDG_CACHE_HOME"):
            self.model_dir = Path(cache_dir) / "audio2anki" / "models" / "speechbrain"
        else:
            self.model_dir = Path.home() / ".cache" / "audio2anki" / "models" / "speechbrain"
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Loading SpeechBrain voice separation model...")
        
        # Check if model is already cached
        if self._is_model_cached():
            logger.info(f"Using cached model from {self.model_dir}")
        else:
            logger.info(f"Model not found in {self.model_dir}, downloading from {self.MODEL_SOURCE}")
            
        try:
            self.separator = SepformerSeparation.from_hparams(
                source=self.MODEL_SOURCE,
                savedir=self.model_dir,
                run_opts={"device": device}
            )
            logger.info("SpeechBrain model loaded successfully")
        except Exception as e:
            # Provide more helpful error message
            if "Connection failed" in str(e):
                raise RuntimeError(
                    "Failed to download SpeechBrain model. Please check your internet connection. "
                    "If you're behind a proxy, make sure it's configured correctly."
                ) from e
            elif "Disk quota exceeded" in str(e):
                raise RuntimeError(
                    f"Not enough disk space to download SpeechBrain model to {self.model_dir}. "
                    "Please free up some space and try again."
                ) from e
            else:
                raise RuntimeError(f"Failed to load SpeechBrain model: {str(e)}") from e

    def _is_model_cached(self) -> bool:
        """Check if the model files are already cached."""
        required_files = ["hyperparams.yaml", "custom.py", "masknet.ckpt", "encoder.ckpt", "decoder.ckpt"]
        return all((self.model_dir / file).exists() for file in required_files)

    def _convert_to_wav(self, input_path: Path) -> Path:
        """Convert audio file to WAV format using soundfile."""
        logger.debug(f"Converting {input_path} to WAV format")
        
        try:
            # Load audio using soundfile
            data, sample_rate = sf.read(str(input_path))
            logger.debug(f"Loaded audio - Shape: {data.shape}, Sample rate: {sample_rate}, Dtype: {data.dtype}")
            
            # Convert to mono if needed
            if len(data.shape) > 1 and data.shape[1] > 1:
                data = np.mean(data, axis=1)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                logger.debug(f"Resampling from {sample_rate}Hz to 16000Hz")
                # Convert to torch tensor for resampling
                waveform = torch.from_numpy(data).float()
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                data = waveform.numpy()
                sample_rate = 16000
            
            # Save as WAV
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_path = Path(temp_wav.name)
                sf.write(str(temp_path), data, sample_rate, subtype='PCM_16')
                
                # Verify the output file exists and is not empty
                if not temp_path.exists() or temp_path.stat().st_size == 0:
                    raise RuntimeError("Failed to create WAV file")
                    
                return temp_path
                
        except Exception as e:
            logger.error(f"Audio conversion failed: {str(e)}")
            if "No audio streams" in str(e):
                raise RuntimeError(f"No audio found in {input_path}") from e
            elif "Error opening" in str(e):
                raise RuntimeError(f"Could not open audio file {input_path}. The file may be corrupted or in an unsupported format.") from e
            else:
                raise RuntimeError(f"Failed to convert audio: {str(e)}") from e

    def _load_audio(self, input_path: Path) -> tuple[torch.Tensor, int]:
        """Load audio file and convert to proper format."""
        logger.debug(f"Loading audio file: {input_path}")
        
        # Convert to WAV first
        wav_path = self._convert_to_wav(input_path)
        
        try:
            # Load the WAV file
            data, sample_rate = sf.read(str(wav_path))
            logger.debug(f"Initial load - Shape: {data.shape}, Sample rate: {sample_rate}, Dtype: {data.dtype}")
            
            # Convert to torch tensor
            waveform = torch.from_numpy(data).float()
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            
            # Normalize if needed
            if waveform.abs().max() > 1:
                logger.debug("Normalizing audio")
                waveform /= waveform.abs().max()
            
            logger.debug(f"After processing - Shape: {waveform.shape}, Sample rate: {sample_rate}, Dtype: {waveform.dtype}")
            return waveform, sample_rate
            
        except Exception as e:
            logger.error(f"Error loading audio: {str(e)}")
            raise
        finally:
            # Clean up temporary file
            try:
                wav_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {wav_path}: {e}")

    def _save_audio(self, waveform: torch.Tensor, sample_rate: int, output_path: Path) -> None:
        """Save audio data to WAV file."""
        logger.debug(f"Saving audio - Shape: {waveform.shape}, Sample rate: {sample_rate}, Dtype: {waveform.dtype}")
        
        try:
            # Ensure proper tensor shape and normalization
            waveform = waveform.squeeze()  # Remove unnecessary dimensions
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # Add back channel dimension
                
            # Convert to CPU and clamp values to valid range
            waveform = waveform.cpu()
            waveform = torch.clamp(waveform, -1.0, 1.0)
            
            # Save using torchaudio with explicit format parameters
            torchaudio.save(
                str(output_path),
                waveform,
                sample_rate,
                format="wav",
                encoding="PCM_S16",
                bits_per_sample=16
            )
            
        except Exception as e:
            logger.error(f"Error saving audio: {str(e)}")
            raise

    def isolate_voice(self, input_path: Path, output_path: Path) -> None:
        """Process audio through SpeechBrain source separation pipeline"""
        try:
            # Load audio
            waveform, rate = self._load_audio(input_path)
            logger.debug(f"Loaded audio - Shape: {waveform.shape}, Sample rate: {rate}")
            
            # Resample to model's expected rate if needed
            if rate != 8000:
                logger.debug(f"Resampling from {rate}Hz to 8000Hz")
                resampler = torchaudio.transforms.Resample(rate, 8000)
                waveform = resampler(waveform)
                rate = 8000
            
            # Move to correct device
            waveform = waveform.to(self.device)
            
            # Separate sources - returns a tuple of tensors [batch, time]
            logger.debug("Running source separation")
            est_sources = self.separator.separate_batch(waveform)
            logger.debug(f"Source separation output shape: {est_sources.shape}")
            
            # Take the first source (speech) and ensure correct shape
            speech = est_sources[:, :, 0]  # Select first source
            logger.debug(f"Selected speech source shape: {speech.shape}")
            
            # Convert back to 16kHz for output
            if rate != 16000:
                logger.debug(f"Resampling from {rate}Hz to 16000Hz")
                resampler = torchaudio.transforms.Resample(rate, 16000)
                speech = resampler(speech)
                rate = 16000
            
            # Save enhanced audio
            self._save_audio(speech, rate, output_path)
            
        except Exception as e:
            logger.error(f"Error in voice isolation: {str(e)}")
            raise RuntimeError(f"SpeechBrain processing failed: {str(e)}") from e

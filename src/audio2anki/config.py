"""Configuration management for audio2anki."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    "clean_files": True,
    "use_cache": True,
    "cache_expiry_days": 7,
    "voice_isolation_provider": "eleven_labs",
    "transcription_provider": "openai_whisper",
}

CONFIG_DIR = os.path.join(os.environ.get("HOME", "."), ".config", "audio2anki")
CONFIG_FILE = "config.toml"


@dataclass
class Config:
    """Configuration settings for audio2anki."""

    clean_files: bool
    use_cache: bool
    cache_expiry_days: int
    voice_isolation_provider: str
    transcription_provider: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create a Config instance from a dictionary."""
        return cls(
            clean_files=data["clean_files"],
            use_cache=data["use_cache"],
            cache_expiry_days=data["cache_expiry_days"],
            voice_isolation_provider=data["voice_isolation_provider"],
            transcription_provider=data["transcription_provider"],
        )

    def to_dict(self) -> dict[str, bool | int | str]:
        """Convert Config to a dictionary."""
        return {
            "clean_files": self.clean_files,
            "use_cache": self.use_cache,
            "cache_expiry_days": self.cache_expiry_days,
            "voice_isolation_provider": self.voice_isolation_provider,
            "transcription_provider": self.transcription_provider,
        }


def get_config_path() -> Path:
    """Get the path to the configuration file."""
    return Path(CONFIG_DIR) / CONFIG_FILE


def get_app_paths() -> dict[str, Path]:
    """Get all application paths."""
    from . import cache  # Import here to avoid circular imports

    return {
        "config_dir": Path(CONFIG_DIR),
        "config_file": Path(CONFIG_DIR) / CONFIG_FILE,
        "cache_dir": Path(cache.CACHE_DIR),
    }


def ensure_config_dir() -> None:
    """Ensure the configuration directory exists."""
    os.makedirs(CONFIG_DIR, exist_ok=True)


def create_default_config() -> None:
    """Create a default configuration file if it doesn't exist."""
    config_path = get_config_path()
    if not config_path.exists():
        ensure_config_dir()
        with open(config_path, "w") as f:
            # Write a commented example configuration
            f.write("# Audio-to-Anki Configuration\n\n")
            f.write("# Whether to clean up intermediate files\n")
            f.write("clean_files = true\n\n")
            f.write("# Cache settings\n")
            f.write("use_cache = true\n")
            f.write("cache_expiry_days = 7\n\n")
            f.write("# API providers\n")
            f.write('voice_isolation_provider = "eleven_labs"\n')
            f.write('transcription_provider = "openai_whisper"\n')
        logger.info(f"Created default configuration file at {config_path}")


def load_config() -> Config:
    """Load configuration from file or return defaults.

    The configuration is loaded from $HOME/.config/audio2anki/config.toml.
    If the file doesn't exist or can't be parsed, default values are used.

    Returns:
        Config: The configuration object with all settings.
    """
    config_path = get_config_path()
    config_dict = DEFAULT_CONFIG.copy()

    if not config_path.exists():
        logger.info(f"Config file not found at {config_path}. Using default configuration.")
        create_default_config()
    else:
        try:
            with open(config_path, "rb") as f:
                file_config = tomllib.load(f)
                # Update only the keys that exist in our default config
                for key in DEFAULT_CONFIG:
                    if key in file_config:
                        config_dict[key] = file_config[key]
        except (tomllib.TOMLDecodeError, OSError) as e:
            logger.warning(f"Error reading config file: {e}. Using default configuration.")

    return Config.from_dict(config_dict)


def validate_config(config: Config) -> list[str]:
    """Validate configuration values.

    Args:
        config: The configuration object to validate.

    Returns:
        list[str]: List of validation error messages. Empty if valid.
    """
    errors = []

    # Validate cache expiry days
    if config.cache_expiry_days < 1:
        errors.append("cache_expiry_days must be at least 1")

    # Validate provider names
    valid_voice_providers = ["eleven_labs"]
    if config.voice_isolation_provider not in valid_voice_providers:
        errors.append(f"voice_isolation_provider must be one of: {', '.join(valid_voice_providers)}")

    valid_transcription_providers = ["openai_whisper"]
    if config.transcription_provider not in valid_transcription_providers:
        errors.append(f"transcription_provider must be one of: {', '.join(valid_transcription_providers)}")

    return errors

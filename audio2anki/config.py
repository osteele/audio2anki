"""Configuration management for audio2anki."""

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli_w
import tomllib

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG: dict[str, Any] = {
    "clean_files": True,
    "use_cache": True,
    "cache_expiry_days": 14,
    "voice_isolation_provider": "eleven_labs",
    "transcription_provider": "openai_whisper",
    "use_artifact_cache": True,
    "max_artifact_cache_size_mb": 2000,  # 2GB default limit
}

CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "audio2anki"
CONFIG_FILE = "config.toml"


@dataclass
class Config:
    """Configuration settings for audio2anki."""

    clean_files: bool
    use_cache: bool
    cache_expiry_days: int
    voice_isolation_provider: str
    transcription_provider: str
    use_artifact_cache: bool
    max_artifact_cache_size_mb: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create a Config instance from a dictionary."""
        return cls(
            clean_files=data["clean_files"],
            use_cache=data["use_cache"],
            cache_expiry_days=data["cache_expiry_days"],
            voice_isolation_provider=data["voice_isolation_provider"],
            transcription_provider=data["transcription_provider"],
            use_artifact_cache=data.get("use_artifact_cache", True),
            max_artifact_cache_size_mb=data.get("max_artifact_cache_size_mb", 2000),
        )

    def to_dict(self) -> dict[str, bool | int | str]:
        """Convert Config to a dictionary."""
        return {
            "clean_files": self.clean_files,
            "use_cache": self.use_cache,
            "cache_expiry_days": self.cache_expiry_days,
            "voice_isolation_provider": self.voice_isolation_provider,
            "transcription_provider": self.transcription_provider,
            "use_artifact_cache": self.use_artifact_cache,
            "max_artifact_cache_size_mb": self.max_artifact_cache_size_mb,
        }


def get_config_path() -> Path:
    """Get the path to the configuration file."""
    return Path(CONFIG_DIR) / CONFIG_FILE


def get_app_paths() -> dict[str, Path]:
    """Get all application paths."""

    return {
        "config_dir": Path(CONFIG_DIR),
        "config_file": Path(CONFIG_DIR) / CONFIG_FILE,
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
            f.write("# Temporary cache settings (per-run)\n")
            f.write("use_cache = true\n\n")
            f.write("# Artifact cache settings (persistent between runs)\n")
            f.write("use_artifact_cache = true\n")
            f.write("cache_expiry_days = 14\n")
            f.write("max_artifact_cache_size_mb = 2000\n\n")
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
    errors: list[str] = []

    # Validate cache expiry days
    if config.cache_expiry_days < 1:
        errors.append("cache_expiry_days must be at least 1")

    # Validate max cache size
    if config.max_artifact_cache_size_mb < 100:
        errors.append("max_artifact_cache_size_mb must be at least 100 MB")

    # Validate provider names
    valid_voice_providers = ["eleven_labs"]
    if config.voice_isolation_provider not in valid_voice_providers:
        errors.append(f"voice_isolation_provider must be one of: {', '.join(valid_voice_providers)}")

    valid_transcription_providers = ["openai_whisper"]
    if config.transcription_provider not in valid_transcription_providers:
        errors.append(f"transcription_provider must be one of: {', '.join(valid_transcription_providers)}")

    return errors


def set_config_value(key: str, value: str) -> tuple[bool, str]:
    """Set a configuration value.

    Args:
        key: The configuration key to set
        value: The value to set (will be type-converted based on default config type)

    Returns:
        tuple of (success, message)
    """
    config = load_config()
    config_dict = config.to_dict()

    if key not in DEFAULT_CONFIG:
        return False, f"Unknown configuration key: {key}"

    default_type: Any = type(DEFAULT_CONFIG[key])
    converted_value: Any = None
    try:
        if default_type is bool:
            converted_value = value.lower() in ("true", "1", "yes", "on")
        elif default_type is int:
            converted_value = int(value)
        elif default_type is str:
            converted_value = value  # Assuming the default is str
    except ValueError:
        return False, f"Invalid value for {key}. Expected {default_type.__name__}"

    if converted_value is None:
        return False, f"Invalid value for {key}. Expected {default_type.__name__}"

    config_dict[key] = converted_value

    # Validate the new configuration
    new_config = Config.from_dict(config_dict)
    errors = validate_config(new_config)
    if errors:
        return False, f"Invalid configuration: {', '.join(errors)}"

    # Write the configuration
    config_path = get_config_path()
    ensure_config_dir()
    try:
        with open(config_path, "wb") as f:
            tomli_w.dump(config_dict, f)
        return True, f"Successfully set {key} to {converted_value}"
    except Exception as e:
        return False, f"Error writing configuration: {e}"


def edit_config() -> tuple[bool, str]:
    """Open the configuration file in the default editor.

    Returns:
        tuple of (success, message)
    """
    config_path = get_config_path()
    ensure_config_dir()

    if not config_path.exists():
        create_default_config()

    editor = os.environ.get("EDITOR", "nano")
    try:
        subprocess.run([editor, str(config_path)], check=True)

        # Validate the edited configuration
        try:
            config = load_config()
            errors = validate_config(config)
            if errors:
                return False, f"Invalid configuration: {', '.join(errors)}"
            return True, "Configuration updated successfully"
        except Exception as e:
            return False, f"Error in configuration file: {e}"
    except subprocess.CalledProcessError:
        return False, "Editor exited with an error"
    except Exception as e:
        return False, f"Error opening editor: {e}"


def reset_config() -> tuple[bool, str]:
    """Reset the configuration to default values.

    Returns:
        tuple of (success, message)
    """
    config_path = get_config_path()
    try:
        if config_path.exists():
            config_path.unlink()
        create_default_config()
        return True, "Configuration reset to defaults"
    except Exception as e:
        return False, f"Error resetting configuration: {e}"

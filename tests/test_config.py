"""Tests for the configuration module."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from pytest import MonkeyPatch
from tests.conftest import CacheTestEnv

from audio2anki.config import (
    CONFIG_FILE,
    Config,
    create_default_config,
    ensure_config_dir,
    get_config_path,
    load_config,
    validate_config,
)


@pytest.fixture
def temp_config_dir(monkeypatch: MonkeyPatch) -> Generator[Path, None, None]:
    """Create a temporary config directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Override the CONFIG_DIR constant for testing
        monkeypatch.setattr("audio2anki.config.CONFIG_DIR", temp_dir)
        yield Path(temp_dir)


@pytest.fixture
def temp_config_file(temp_config_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary config file with test content."""
    config_path = temp_config_dir / CONFIG_FILE
    config_content = """
    # Test configuration
    clean_files = false
    use_cache = true
    cache_expiry_days = 14
    voice_isolation_provider = "eleven_labs"
    transcription_provider = "openai_whisper"
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_content)
    yield config_path
    if config_path.exists():
        config_path.unlink()


def test_get_config_path(temp_config_dir: Path) -> None:
    """Test that get_config_path returns the correct path."""
    expected_path = temp_config_dir / CONFIG_FILE
    assert get_config_path() == expected_path


def test_ensure_config_dir(temp_config_dir: Path) -> None:
    """Test that ensure_config_dir creates the directory."""
    # Remove the directory if it exists
    if temp_config_dir.exists():
        temp_config_dir.rmdir()

    ensure_config_dir()
    assert temp_config_dir.exists()
    assert temp_config_dir.is_dir()


def test_create_default_config(temp_config_dir: Path) -> None:
    """Test creation of default config file."""
    config_path = temp_config_dir / CONFIG_FILE
    create_default_config()

    assert config_path.exists()
    content = config_path.read_text()
    assert "clean_files = true" in content
    assert "cache_expiry_days = 14" in content
    assert 'voice_isolation_provider = "eleven_labs"' in content
    assert 'transcription_provider = "openai_whisper"' in content
    assert "use_artifact_cache = true" in content
    assert "max_artifact_cache_size_mb = 2000" in content


def test_load_config_with_file(temp_config_file: Path) -> None:
    """Test loading configuration from an existing file."""
    config = load_config()
    assert isinstance(config, Config)
    assert config.clean_files is False  # Overridden in test file
    assert config.cache_expiry_days == 14  # Overridden in test file
    assert config.voice_isolation_provider == "eleven_labs"
    assert config.transcription_provider == "openai_whisper"


def test_load_config_no_file(temp_config_dir: Path) -> None:
    """Test loading configuration when no file exists."""
    config = load_config()
    assert isinstance(config, Config)
    assert config.clean_files is True  # Default value
    assert config.cache_expiry_days == 14  # Default value
    assert config.voice_isolation_provider == "eleven_labs"
    assert config.transcription_provider == "openai_whisper"
    assert config.use_artifact_cache is True
    assert config.max_artifact_cache_size_mb == 2000


def test_load_config_invalid_file(temp_config_dir: Path) -> None:
    """Test loading configuration with an invalid TOML file."""
    config_path = temp_config_dir / CONFIG_FILE
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("invalid = toml [ file")

    config = load_config()
    assert isinstance(config, Config)
    # Should fall back to defaults
    assert config.clean_files is True
    assert config.cache_expiry_days == 14
    assert config.use_artifact_cache is True
    assert config.max_artifact_cache_size_mb == 2000


def test_config_validation_valid() -> None:
    """Test configuration validation with valid values."""
    config = Config(
        clean_files=True,
        use_cache=True,
        cache_expiry_days=7,
        voice_isolation_provider="eleven_labs",
        transcription_provider="openai_whisper",
        use_artifact_cache=True,
        max_artifact_cache_size_mb=2000,
    )
    errors = validate_config(config)
    assert not errors


def test_config_validation_invalid() -> None:
    """Test configuration validation with invalid values."""
    config = Config(
        clean_files=True,
        use_cache=True,
        cache_expiry_days=0,  # Invalid value
        voice_isolation_provider="invalid_provider",  # Invalid value
        transcription_provider="invalid_provider",  # Invalid value
        use_artifact_cache=True,
        max_artifact_cache_size_mb=50,  # Invalid value (too small)
    )
    errors = validate_config(config)
    assert len(errors) == 4


def test_config_to_from_dict() -> None:
    """Test conversion between Config and dictionary."""
    original_config = Config(
        clean_files=True,
        use_cache=True,
        cache_expiry_days=7,
        voice_isolation_provider="eleven_labs",
        transcription_provider="openai_whisper",
        use_artifact_cache=True,
        max_artifact_cache_size_mb=2000,
    )

    # Convert to dict and back
    config_dict = original_config.to_dict()
    restored_config = Config.from_dict(config_dict)

    # Check all attributes match
    assert restored_config.clean_files == original_config.clean_files
    assert restored_config.use_cache == original_config.use_cache
    assert restored_config.cache_expiry_days == original_config.cache_expiry_days
    assert restored_config.voice_isolation_provider == original_config.voice_isolation_provider
    assert restored_config.transcription_provider == original_config.transcription_provider


@pytest.fixture
def config(test_env: CacheTestEnv) -> Config:
    """Create a test configuration."""
    # Now the config module will use the test directory
    from audio2anki import config

    return config.load_config()


def test_load_config(test_env: CacheTestEnv) -> None:
    """Test loading configuration."""
    # Import config after environment is set up
    from audio2anki import config

    cfg = config.load_config()
    assert isinstance(cfg, Config)
    assert cfg.clean_files is True  # default value


def test_validate_config(test_env: CacheTestEnv) -> None:
    """Test configuration validation."""
    from audio2anki import config

    cfg = Config(
        clean_files=True,
        use_cache=True,
        cache_expiry_days=7,
        voice_isolation_provider="eleven_labs",
        transcription_provider="openai_whisper",
        use_artifact_cache=True,
        max_artifact_cache_size_mb=2000,
    )

    errors = config.validate_config(cfg)
    assert not errors

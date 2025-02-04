"""Pytest configuration file."""

import pytest
import torch
import torchaudio
from rich.progress import Progress


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")


@pytest.fixture
def progress() -> Progress:
    """Progress bar for testing."""
    return Progress()


@pytest.fixture
def test_audio(tmp_path):
    """Generate a valid WAV file for testing"""
    path = tmp_path / "test.wav"
    # Generate 1 second of random audio at 16kHz, ensuring 2D tensor [channels, time]
    waveform = torch.randn(1, 16000)  # [1 channel, 16000 samples]
    torchaudio.save(str(path), waveform, 16000)
    return path

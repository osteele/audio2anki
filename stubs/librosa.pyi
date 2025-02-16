"""
Enhanced type stubs for the parts of librosa used by audio2anki.
Provides more specific NDArray[...] types to remove "partially unknown" warnings.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from numpy._typing import NDArray

# If your code universally uses float32 for librosa, declare a specialized type here.
Float32Array = NDArray[np.float32]

class AudioFile: ...
class SoundFile: ...

def load(
    path: str | int | os.PathLike[Any] | AudioFile | SoundFile,
    *,
    sr: float | None = 22050,
    mono: bool = True,
    offset: float = 0.0,
    duration: float | None = None,
    dtype: Any = np.float32,
    res_type: str = "soxr_hq",
) -> tuple[Float32Array, float | int]:
    """
    Stub matching librosa.load's signature more precisely:
    returns (audio array, sample rate).
    The audio array is typed as NDArray[np.float32].
    """
    ...

def get_duration(
    *,
    y: Float32Array | None = None,
    sr: float = 22050,
    S: Float32Array | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    path: str | os.PathLike[Any] | None = None,
    filename: Any = None,
) -> float:
    """
    Stub matching librosa.get_duration's signature more precisely:
    returns a float, with array parameters typed as NDArray[np.float32].
    """
    ...

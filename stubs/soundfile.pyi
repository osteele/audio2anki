"""
Minimal type stub for the PySoundFile (soundfile) module.

This stub provides only the definitions needed by the audio2anki project.
"""

from typing import Any, Literal

import numpy as np

def write(
    file: Any,
    data: Any,
    samplerate: int,
    *,
    subtype: Any = None,
    endian: Any = None,
    format: Any = None,
    closefd: bool = True,
    compression_level: Any = None,
    bitrate_mode: Any = None,
) -> None: ...

class SoundFile:
    def __init__(self, file: Any, mode: str = "r", **kwargs: Any) -> None: ...
    def __enter__(self) -> SoundFile: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...
    def read(
        self,
        frames: int = -1,
        dtype: Literal["float64"] | Literal["float32"] | Literal["int32"] | Literal["int16"] = "float64",
        always_2d: bool = False,
        fill_value: float | None = None,
        out: np.ndarray[Any, Any] | None = None,
    ) -> Any: ...
    def write(self, data: Any) -> None: ...

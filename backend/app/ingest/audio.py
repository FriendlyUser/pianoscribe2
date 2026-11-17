"""PCM helpers for already-decoded (trusted) audio."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


def read_wav(path: Path) -> tuple[FloatArray, int]:
    """Read a WAV as float32 shaped ``(channels, samples)``."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.T), int(sr)


def write_wav(path: Path, audio: FloatArray, sr: int) -> None:
    """Atomically write ``(channels, samples)`` or mono ``(samples,)`` as 16-bit PCM."""
    frames = audio.T if audio.ndim == 2 else audio
    tmp = path.with_suffix(".tmp.wav")
    sf.write(str(tmp), np.clip(frames, -1.0, 1.0), sr, subtype="PCM_16")
    tmp.replace(path)

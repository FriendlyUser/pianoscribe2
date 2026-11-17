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


def peak_normalize(audio: FloatArray, peak_dbfs: float = -1.0) -> FloatArray:
    """Remove DC offset and scale linearly so the peak sits at ``peak_dbfs``.

    A linear gain (not a compressor) keeps the recording's dynamics intact for
    the models that read loudness from it.
    """
    centered = audio - audio.mean(axis=-1, keepdims=True)
    peak = float(np.max(np.abs(centered))) if centered.size else 0.0
    if peak < 1e-6:
        return centered.astype(np.float32)
    target = 10.0 ** (peak_dbfs / 20.0)
    return np.asarray(centered * (target / peak), dtype=np.float32)

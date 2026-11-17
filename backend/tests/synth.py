"""Deterministic piano-like audio synthesis used to build test fixtures.

The synthesiser is intentionally dependency-light (numpy only). Each note is a
sum of slightly inharmonic partials with per-partial exponential decay and a
short hammer transient, which is close enough to a real piano for the
transcription model to recognise pitches, onsets and offsets reliably.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44_100


@dataclass(frozen=True)
class SynthNote:
    pitch: int  # MIDI note number
    start_beat: float
    duration_beats: float
    velocity: int = 80


def _note_wave(pitch: int, seconds: float, velocity: int, sr: int) -> np.ndarray:
    f0 = 440.0 * 2.0 ** ((pitch - 69) / 12.0)
    release = 0.12
    n = int((seconds + release) * sr)
    t = np.arange(n) / sr
    out = np.zeros(n)
    inharmonicity = 0.0004
    brightness = 0.35 + 0.5 * (velocity / 127.0)
    for k in range(1, 13):
        fk = f0 * k * np.sqrt(1.0 + inharmonicity * k * k)
        if fk > sr / 2 - 500:
            break
        amp = (brightness ** (k - 1)) / k
        decay = 1.2 + 0.9 * k + (pitch - 60) * 0.03
        out += amp * np.exp(-decay * t) * np.sin(2 * np.pi * fk * t + 0.3 * k)
    attack = np.minimum(1.0, t / 0.004)
    rng = np.random.default_rng(pitch)
    hammer = rng.standard_normal(n) * np.exp(-t / 0.006) * 0.08
    out = (out + hammer) * attack
    rel = np.ones(n)
    rel_start = int(seconds * sr)
    rel[rel_start:] = np.exp(-(t[rel_start:] - seconds) / (release / 5))
    return out * rel * (velocity / 127.0) ** 1.5


def render(notes: list[SynthNote], bpm: float, sr: int = SAMPLE_RATE, lead_in: float = 0.25) -> np.ndarray:
    """Render ``notes`` to a mono float waveform in [-1, 1]."""
    spb = 60.0 / bpm
    total = max(n.start_beat + n.duration_beats for n in notes) * spb + lead_in + 1.0
    audio = np.zeros(int(total * sr))
    for n in notes:
        w = _note_wave(n.pitch, n.duration_beats * spb * 0.95, n.velocity, sr)
        i = int((lead_in + n.start_beat * spb) * sr)
        audio[i : i + len(w)] += w[: len(audio) - i]
    peak = float(np.max(np.abs(audio))) or 1.0
    return (audio / peak * 0.8).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> Path:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(pcm.tobytes())
    return path


# --------------------------------------------------------------------------
# Fixture piece: "Ode to Joy" (Beethoven, public domain) in G major, 4/4.
# --------------------------------------------------------------------------

ODE_BPM = 112.0
ODE_KEY_FIFTHS = 1  # G major
ODE_METER = (4, 4)

_G4, _A4, _B4, _C5, _D5 = 67, 69, 71, 72, 74
_G2, _B2, _D3, _FS3, _A3, _A2 = 43, 47, 50, 54, 57, 45


def ode_to_joy() -> list[SynthNote]:
    """Eight bars: melody in the right hand, broken chords in the left."""
    notes: list[SynthNote] = []

    def melody(bar: int, seq: list[tuple[int, float]]) -> None:
        beat = bar * 4.0
        for pitch, dur in seq:
            notes.append(SynthNote(pitch, beat, dur, 92))
            beat += dur

    def bass(bar: int, seq: list[tuple[tuple[int, ...], float]]) -> None:
        beat = bar * 4.0
        for pitches, dur in seq:
            for p in pitches:
                notes.append(SynthNote(p, beat, dur, 66))
            beat += dur

    phrase_a = [(_B4, 1), (_B4, 1), (_C5, 1), (_D5, 1)]
    phrase_b = [(_D5, 1), (_C5, 1), (_B4, 1), (_A4, 1)]
    phrase_c = [(_G4, 1), (_G4, 1), (_A4, 1), (_B4, 1)]
    g_arp = [((_G2,), 1), ((_D3,), 1), ((_B2,), 1), ((_D3,), 1)]
    d_arp = [((_D3,), 1), ((_A3,), 1), ((_FS3,), 1), ((_A3,), 1)]

    for offset in (0, 4):
        melody(offset + 0, phrase_a)
        melody(offset + 1, phrase_b)
        melody(offset + 2, phrase_c)
        bass(offset + 0, g_arp)
        bass(offset + 1, d_arp)
        bass(offset + 2, g_arp)
    melody(3, [(_B4, 1.5), (_A4, 0.5), (_A4, 2)])
    bass(3, [((_D3, _A3), 2), ((_D3, _FS3), 2)])
    melody(7, [(_A4, 1.5), (_G4, 0.5), (_G4, 2)])
    bass(7, [((_A2, _FS3), 2), ((_G2, _D3), 2)])
    return notes


def ode_to_joy_wav(path: Path) -> Path:
    return write_wav(path, render(ode_to_joy(), ODE_BPM))


if __name__ == "__main__":  # python -m tests.synth OUTPUT.wav
    import sys

    print(ode_to_joy_wav(Path(sys.argv[1] if len(sys.argv) > 1 else "ode_to_joy.wav")))

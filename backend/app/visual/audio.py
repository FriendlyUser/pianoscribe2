"""Check the notes read from the picture against the sound of the same video.

The picture gives every note's key and timing, but two things it cannot give by itself:

* **The octave**, when the keyboard shown is not a full 88 keys: a row of keys looks
  the same an octave up. The audio settles it (``best_octave``): the notes are laid
  over the spectrum of the recording at each candidate octave, and the true one is
  where their fundamentals and overtones meet the sound.
* **The exact moment** a falling bar counts as struck. Notes are timed against the
  attacks heard in the audio (``audio_offset``), so the score's notes line up with the
  beats the beat tracker hears in the same recording.

Both need nothing but a spectrum (numpy only, no model).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Float = NDArray[np.float64]
OCTAVES = (-24, -12, 0, 12, 24)
# Relative strength of a piano tone's partials, by harmonic number (fundamental first).
PARTIALS = ((0, 1.0), (12, 0.7), (19, 0.45), (24, 0.35), (28, 0.2), (31, 0.15))
LOWEST, HIGHEST = 21, 120  # semitone bands of the pitch spectrum (MIDI numbers)


def _mono(audio: NDArray[np.float32]) -> Float:
    return audio.mean(axis=0).astype(np.float64) if audio.ndim == 2 else audio.astype(np.float64)


def _spectrogram(signal: Float, size: int, hop: int) -> Float:
    """Magnitude spectrogram [frames, bins]; frame i is centred on sample i * hop."""
    padded = np.concatenate((np.zeros(size // 2), signal, np.zeros(size)))
    count = 1 + (len(signal) + size // 2) // hop
    window = np.hanning(size)
    out = np.empty((count, size // 2 + 1))
    for start in range(0, count, 256):
        stop = min(count, start + 256)
        index = (np.arange(start, stop) * hop)[:, None] + np.arange(size)[None, :]
        out[start:stop] = np.abs(np.fft.rfft(padded[index] * window, axis=1))
    return out


def onset_strength(audio: NDArray[np.float32], rate: int, *, size: int = 1024, hop: int = 256) -> tuple[Float, float]:
    """(how sharply the sound rises in each frame, frames per second)."""
    magnitude = np.log1p(100.0 * _spectrogram(_mono(audio), size, hop) / size)
    rise = np.maximum(magnitude[1:] - magnitude[:-1], 0.0).sum(axis=1)
    return np.concatenate(([0.0], rise)), rate / hop


def attack_times(audio: NDArray[np.float32], rate: int, *, apart: float = 0.08, within: float = 0.06) -> list[float]:
    """Seconds of the strong attacks in ``audio``: where the sound's rise jumps, inside ``within`` seconds,
    by more than its usual level (its mean plus two standard deviations); the strongest of any
    closer together than ``apart``.

    The jump, not the level: while a chord rings the rise ripples on at about that level for
    some tenths of a second, and a peak of the ripple is no note.
    """
    strength, per_second = onset_strength(audio, rate)
    level = float(strength.mean() + 2.0 * strength.std())
    back = max(1, int(round(within * per_second)))
    kept: list[int] = []
    for i in range(1, len(strength) - 1):
        if strength[i] < strength[i - 1] or strength[i] <= strength[i + 1]:
            continue
        if strength[i] - float(strength[max(0, i - back) : i].min()) < level:
            continue
        if kept and (i - kept[-1]) / per_second < apart:
            if strength[i] > strength[kept[-1]]:
                kept[-1] = i
        else:
            kept.append(i)
    # As in ``audio_offset``: a frame's rise is measured between two windows, half a hop before its centre.
    return [(i - 0.5) / per_second for i in kept]


@dataclass(frozen=True)
class Offset:
    seconds: float  # add this to the notes' times to put them on the audio's clock
    sharpness: float  # how far the best lag stands out from the others (standard deviations)

    @property
    def reliable(self) -> bool:
        return self.sharpness >= 6.0


def audio_offset(
    onsets: Sequence[float], audio: NDArray[np.float32], rate: int, *, search: float = 0.3
) -> Offset | None:
    """How much later (seconds) the audio's attacks come than the given note onsets."""
    if len(onsets) < 8:
        return None
    strength, per_second = onset_strength(audio, rate)
    strength = strength - strength.mean()
    lags = np.arange(-int(search * per_second), int(search * per_second) + 1)
    positions = np.asarray(onsets, dtype=np.float64) * per_second
    scores = np.empty(len(lags))
    for i, shift in enumerate(lags):
        where = positions + shift
        scores[i] = np.interp(where, np.arange(len(strength)), strength, left=0.0, right=0.0).sum()
    best = int(np.argmax(scores))
    lag = float(lags[best])
    if 0 < best < len(lags) - 1:
        before, at, after = scores[best - 1 : best + 2]
        bend = before - 2 * at + after
        if bend < 0:
            lag += float(0.5 * (before - after) / bend)
    away = np.abs(lags - lags[best]) > 0.04 * per_second
    spread = float(scores[away].std()) or 1.0
    sharpness = float((scores[best] - scores[away].mean()) / spread)
    # A frame's rise is measured between two windows; the attack sits half a hop before its centre.
    return Offset(seconds=(lag - 0.5) / per_second, sharpness=sharpness)


def pitch_spectrum(audio: NDArray[np.float32], rate: int, *, size: int = 8192, hop: int = 2048) -> tuple[Float, float]:
    """(energy in each semitone band [frames, HIGHEST - LOWEST + 1], frames per second)."""
    magnitude = _spectrogram(_mono(audio), size, hop)
    frequency = np.fft.rfftfreq(size, 1.0 / rate)
    pitch = np.full(len(frequency), -1.0)
    pitch[1:] = 69.0 + 12.0 * np.log2(frequency[1:] / 440.0)
    bands = np.zeros((len(magnitude), HIGHEST - LOWEST + 1))
    for band in range(LOWEST, HIGHEST + 1):
        inside = np.abs(pitch - band) <= 0.5
        if inside.any():
            bands[:, band - LOWEST] = magnitude[:, inside].max(axis=1)
        else:  # the lowest bands are narrower than one bin: read the nearest bin
            nearest = int(np.argmin(np.abs(pitch - band)))
            bands[:, band - LOWEST] = magnitude[:, nearest]
    return bands, rate / hop


def octave_scores(notes: Sequence[tuple[float, float, int]], audio: NDArray[np.float32], rate: int) -> dict[int, float]:
    """Agreement (0..1) of ``notes`` (start, end, pitch) with the audio, moved by each candidate octave."""
    bands, per_second = pitch_spectrum(audio, rate)
    heard = np.sqrt(bands)
    heard -= np.median(heard, axis=1, keepdims=True)  # remove the broadband floor of each frame
    heard = np.maximum(heard, 0.0)
    frames = len(heard)
    scores: dict[int, float] = {}
    for shift in OCTAVES:
        expected = np.zeros_like(heard)
        for start, end, pitch in notes:
            first = int(np.floor(start * per_second))
            last = int(np.ceil(min(end, start + 1.5) * per_second))
            if first >= frames:
                continue
            for interval, weight in PARTIALS:
                band = pitch + shift + interval - LOWEST
                if 0 <= band < expected.shape[1]:
                    expected[first : max(last, first + 1), band] += weight
        norm = float(np.linalg.norm(expected) * np.linalg.norm(heard))
        scores[shift] = float((expected * heard).sum() / norm) if norm > 0 else 0.0
    return scores


def best_octave(scores: dict[int, float], *, margin: float = 0.02) -> int | None:
    """The octave shift the audio supports, or None when no candidate stands clear of the others."""
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) < 2 or ranked[0][1] <= 0 or ranked[0][1] - ranked[1][1] < margin:
        return None
    return ranked[0][0]


RINGS = 0.15  # share of a note's attack still sounding well after its key is up, above which the pedal holds the notes


def rings_on(
    notes: Sequence[tuple[float, float, int]], audio: NDArray[np.float32], rate: int, *, after: float = 0.3
) -> float | None:
    """How much of a short note still sounds ``after`` seconds after its key is released, as a share of its attack.

    The median over the notes (start, end, pitch, on the audio's clock) it can be measured
    for: short ones, not too low for their pitch to be told from its neighbours, whose pitch
    and whose overtones nothing else strikes until then. None when there are too few.

    A damper stops a string within a tenth of a second of the key's release. A note that
    goes on sounding is held by the pedal: its key says when it was struck, not how long it
    lasts, and a score writes it out to the next note rather than as a short note and a rest.
    """
    size, hop = 4096, 512
    magnitude = _spectrogram(_mono(audio), size, hop)
    per_second = rate / hop
    frequency = np.fft.rfftfreq(size, 1.0 / rate)
    pitch_of = np.full(len(frequency), -99.0)
    pitch_of[1:] = 69.0 + 12.0 * np.log2(frequency[1:] / 440.0)
    starts = np.array([note[0] for note in notes])
    pitches = np.array([note[2] for note in notes])
    bands: dict[int, Float] = {}
    shares = []
    for start, end, pitch in notes:
        held = end - start
        if pitch < 55 or held > 0.2:
            continue
        window = (starts > start + 0.02) & (starts < end + after + 0.05)
        if any((other - pitch) % 12 == 0 or pitch - other in (19, 28) for other in pitches[window]):
            continue  # struck again, or a note below whose overtone it is
        first, last = int(round((start - 0.06) * per_second)), int(round((end + after) * per_second))
        if first < 0 or last >= len(magnitude):
            continue
        if pitch not in bands:
            bands[pitch] = magnitude[:, np.abs(pitch_of - pitch) <= 0.4].max(axis=1)
        level = bands[pitch]
        before = float(level[first])
        attack = max(float(level[int(round((start + t) * per_second))]) for t in (0.05, 0.08)) - before
        if attack > 0:
            shares.append((float(level[last]) - before) / attack)
    return float(np.median(shares)) if len(shares) >= 20 else None

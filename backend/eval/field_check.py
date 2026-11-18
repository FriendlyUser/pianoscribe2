"""Check a finished job against the recording it was made from, without a reference score.

    python -m eval.field_check RECORDING.wav ARTIFACT_DIR

``ARTIFACT_DIR`` holds a job's downloads: ``performance.mid`` and ``score.musicxml``
(and ``transcription.mid`` for a song). Three things are reported:

1. **Harmony agreement.** The pitch-class content of the audio, from a plain short-time
   spectrum, against the notes of the MIDI, in half-second windows (cosine similarity,
   1.0 is a perfect match). No model is involved on the audio side. Two baselines show
   what a wrong answer scores on the same recording: the same notes eight seconds late,
   and the same notes transposed.
2. **Note fidelity** (``eval.fidelity``): the written score against the performance it was
   written from.
3. **Score facts**: key signatures, time signatures, bars.

For scale: an arrangement that scores 0.88 against a human reference gives a harmony
agreement of about 0.72 here, a correct solo transcription 0.84 to 0.89, and transposed
notes about 0.3. This cannot tell whether each melody note is right; it confirms that the
result follows the recording and that the notation is faithful to what was transcribed.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from eval.fidelity import note_fidelity
from eval.scoreio import read_musicxml

SAMPLE_RATE, HOP, FFT, WINDOW_SECONDS = 11_025, 1024, 4096, 0.5


def audio_chroma(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """(frame times, chroma [12, frames]) of a recording: spectral energy folded onto pitch classes."""
    from scipy.signal import resample_poly, stft

    samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if rate != SAMPLE_RATE:
        mono = resample_poly(mono, SAMPLE_RATE, rate)
    frequencies, times, spectrum = stft(mono, fs=SAMPLE_RATE, nperseg=FFT, noverlap=FFT - HOP, padded=False)
    magnitude = np.log1p(50 * np.abs(spectrum))
    audible = (frequencies >= 55) & (frequencies <= 2100)
    pitch = np.round(69 + 12 * np.log2(frequencies[audible] / 440.0)).astype(int)
    chroma = np.zeros((12, magnitude.shape[1]))
    for pitch_class in range(12):
        chroma[pitch_class] = magnitude[audible][pitch % 12 == pitch_class].sum(axis=0)
    return times, chroma


def midi_notes(path: str | Path) -> list[tuple[float, float, int]]:
    """(start, end, pitch) of every pitched note, in seconds."""
    now = 0.0
    sounding: dict[tuple[int, int], list[float]] = defaultdict(list)
    notes = []
    for message in mido.MidiFile(str(path)):
        now += message.time
        if not hasattr(message, "channel") or message.channel == 9:
            continue
        key = (message.channel, getattr(message, "note", -1))
        if message.type == "note_on" and message.velocity > 0:
            sounding[key].append(now)
        elif message.type in ("note_off", "note_on") and sounding[key]:
            start = sounding[key].pop(0)
            notes.append((start, max(now, start + 0.05), message.note))
    return sorted(notes)


def midi_chroma(
    notes: list[tuple[float, float, int]], times: np.ndarray, *, shift: float = 0.0, transpose: int = 0
) -> np.ndarray:
    chroma = np.zeros((12, len(times)))
    for start, end, pitch in notes:
        first = int(np.searchsorted(times, start + shift))
        last = int(np.searchsorted(times, min(end, start + 2.0) + shift))
        chroma[(pitch + transpose) % 12, first : max(last, first + 1)] += 1.0
    return chroma


def _windows(times: np.ndarray, chroma: np.ndarray) -> np.ndarray:
    index = (times // WINDOW_SECONDS).astype(int)
    out = np.zeros((12, int(index.max()) + 1))
    np.add.at(out.T, index, chroma.T)
    return out


def agreement(audio: np.ndarray, notes: np.ndarray) -> float:
    """Mean cosine similarity of two windowed chroma matrices, over windows where both have content."""
    used = (audio.sum(axis=0) > np.percentile(audio.sum(axis=0), 10)) & (notes.sum(axis=0) > 0)
    if not used.any():
        return 0.0
    a, n = audio[:, used], notes[:, used]
    a = a - a.min(axis=0, keepdims=True)  # remove the broadband floor
    return float(((a * n).sum(axis=0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(n, axis=0) + 1e-9)).mean())


def harmony_agreement(recording: str | Path, midi: str | Path) -> dict[str, float]:
    """Agreement of ``midi`` with ``recording``, and what misplaced and transposed notes would score."""
    times, chroma = audio_chroma(recording)
    audio = _windows(times, chroma)
    notes = midi_notes(midi)
    transposed = [agreement(audio, _windows(times, midi_chroma(notes, times, transpose=k))) for k in range(1, 12)]
    return {
        "agreement": agreement(audio, _windows(times, midi_chroma(notes, times))),
        "shifted_8s": agreement(audio, _windows(times, midi_chroma(notes, times, shift=8.0))),
        "transposed_mean": float(np.mean(transposed)),
        "transposed_best": float(np.max(transposed)),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    recording, directory = Path(argv[0]), Path(argv[1])
    for name in ("performance.mid", "transcription.mid"):
        if (directory / name).is_file():
            h = harmony_agreement(recording, directory / name)
            print(
                f"harmony agreement, {name}: {h['agreement']:.3f}  "
                f"(same notes 8 s late: {h['shifted_8s']:.3f}; transposed: {h['transposed_mean']:.3f} on average, "
                f"{h['transposed_best']:.3f} at best)"
            )
    score = read_musicxml(directory / "score.musicxml")
    fidelity = note_fidelity(directory / "performance.mid", score)
    print(
        f"note fidelity: {fidelity.performance_notes} notes performed, {fidelity.score_notes} written "
        f"({fidelity.extra:+.0%}); performed notes in the score {fidelity.kept:.3f}, written notes that were "
        f"performed {fidelity.backed:.3f}"
    )
    regions: list[list[int]] = []
    for bar in score.bars:
        if not regions or regions[-1][1] != bar.fifths:
            regions.append([bar.index + 1, bar.fifths, 0])
        regions[-1][2] += 1
    meters = Counter(f"{bar.meter[0]}/{bar.meter[1]}" for bar in score.bars)
    print(f"score: {len(score.bars)} bars; time signatures {dict(meters.most_common(3))}")
    print(
        "key signatures (from bar, sharps + / flats -, bars): " + ", ".join(f"{a}: {k:+d} x{n}" for a, k, n in regions)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

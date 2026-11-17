"""How faithfully a generated piano cover carries the song's melody.

The cover model samples its output, so two runs differ, and now and then a
passage comes out rhythmically off. Each candidate is therefore checked
against something independent of it: the melody that SheetSage2-Refined
transcribed from the original recording. A candidate scores by the share of
those melody notes it plays (same pitch class, at the same moment), which
penalises both wrong notes and right notes at the wrong time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mido
import numpy as np

ONSET_TOLERANCE_SECONDS = 0.12


def read_note_onsets(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(onset seconds, pitches) of every note in a MIDI file, tempo map applied."""
    times: list[float] = []
    pitches: list[int] = []
    now = 0.0
    for message in mido.MidiFile(str(path)):  # iteration yields delta times in seconds
        now += message.time
        if message.type == "note_on" and message.velocity > 0:
            times.append(now)
            pitches.append(message.note)
    order = np.argsort(times, kind="stable")
    return np.asarray(times, dtype=np.float64)[order], np.asarray(pitches, dtype=np.int64)[order]


def melody_fidelity(cover_midi: Path, annotation: dict[str, Any], *, window: float | None = None) -> float:
    """Share (0..1) of the detected melody notes that the cover plays in time.

    ``window`` limits the check to the first ``window`` seconds (the opening
    is where a sampled cover most often goes wrong).
    """
    melody = [n for n in annotation.get("melody", []) if window is None or n["start"] < window]
    if not melody:
        return 0.0
    onsets, pitches = read_note_onsets(cover_midi)
    hits = 0.0
    total = 0.0
    for note in melody:
        weight = min(max(float(note["end"]) - float(note["start"]), 0.1), 1.0)
        total += weight
        lo = int(np.searchsorted(onsets, note["start"] - ONSET_TOLERANCE_SECONDS))
        hi = int(np.searchsorted(onsets, note["start"] + ONSET_TOLERANCE_SECONDS))
        if np.any((pitches[lo:hi] - int(note["pitch"])) % 12 == 0):
            hits += weight
    return hits / total if total else 0.0


def choose_cover(candidates: list[Path], annotation: dict[str, Any]) -> tuple[Path, list[float]]:
    """Pick the candidate most faithful to the melody; returns it and every candidate's score."""
    if not candidates:
        raise ValueError("no cover candidates")
    scores = [
        0.7 * melody_fidelity(c, annotation) + 0.3 * melody_fidelity(c, annotation, window=30.0) for c in candidates
    ]
    return candidates[int(np.argmax(scores))], scores

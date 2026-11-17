"""Score a piano transcription against exact note data (MAESTRO Disklavier MIDI).

Standard note-level metrics from ``mir_eval``: a note is correct when its pitch
matches and its onset is within 50 ms; the stricter variant also requires the
offset within 50 ms or 20 % of the note's length. Reference offsets follow the
MAESTRO convention (extended through the sustain pedal); systems differ in how
they define a release, so the onset score is the primary one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _notes(path: str | Path, *, sustain: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Notes of a MIDI file as (intervals, pitches, velocities).

    With ``sustain``, a note released while the sustain pedal is down lasts
    until the pedal comes up or the same key is struck again - the MAESTRO
    evaluation convention (the string keeps sounding).
    """
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(str(path))
    rows: list[list[float]] = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        notes = sorted(instrument.notes, key=lambda n: n.start)
        ends = np.array([n.end for n in notes])
        if sustain:
            pedal = sorted((c.time, c.value >= 64) for c in instrument.control_changes if c.number == 64)
            spans, down = [], None
            for time, pressed in pedal:
                if pressed and down is None:
                    down = time
                elif not pressed and down is not None:
                    spans.append((down, time))
                    down = None
            if down is not None:
                spans.append((down, midi.get_end_time()))
            next_same: dict[int, float] = {}
            for i in range(len(notes) - 1, -1, -1):
                note = notes[i]
                for start, stop in spans:
                    if start <= note.end < stop:
                        ends[i] = min(stop, next_same.get(note.pitch, stop))
                        break
                next_same[note.pitch] = note.start
        rows += [
            [n.start, max(float(e), n.start + 0.01), n.pitch, n.velocity] for n, e in zip(notes, ends, strict=True)
        ]
    rows.sort()
    data = np.array(rows).reshape(-1, 4)
    return data[:, :2], data[:, 2], data[:, 3]


def note_scores(reference_midi: str | Path, estimated_midi: str | Path) -> dict[str, float]:
    """Onset and onset+offset precision / recall / F1 of ``estimated_midi``."""
    import mir_eval

    ref_intervals, ref_pitches, _ = _notes(reference_midi, sustain=True)
    est_intervals, est_pitches, _ = _notes(estimated_midi)
    hz = lambda p: 440.0 * 2.0 ** ((p - 69.0) / 12.0)  # noqa: E731
    out: dict[str, float] = {"reference_notes": float(len(ref_pitches)), "estimated_notes": float(len(est_pitches))}
    if not len(est_pitches) or not len(ref_pitches):
        return {**out, "onset_f1": 0.0, "onset_precision": 0.0, "onset_recall": 0.0, "offset_f1": 0.0}
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, hz(ref_pitches), est_intervals, hz(est_pitches), onset_tolerance=0.05, offset_ratio=None
    )
    _, _, f_off, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, hz(ref_pitches), est_intervals, hz(est_pitches), onset_tolerance=0.05, offset_ratio=0.2
    )
    return {
        **out,
        "onset_precision": float(p),
        "onset_recall": float(r),
        "onset_f1": float(f),
        "offset_f1": float(f_off),
    }

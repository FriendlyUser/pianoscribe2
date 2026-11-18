"""Is the written score faithful to the performance it was written from?

The score model writes notation from a performance MIDI (a transcription, or an arrangement).
It may respell, tie and regroup, but it should neither invent notes nor lose them. This compares
the two as multisets of pitches: no timing is involved, so it holds across any tempo or beat grid.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import mido

from eval.scoreio import Score


@dataclass(frozen=True)
class NoteFidelity:
    performance_notes: int
    score_notes: int
    kept: float  # share of the performance's notes that are in the score
    backed: float  # share of the score's notes that are in the performance

    @property
    def extra(self) -> float:
        """Score notes beyond the performance, as a share of the performance (0.39: 39 % more notes)."""
        return self.score_notes / self.performance_notes - 1.0 if self.performance_notes else 0.0


def performance_pitches(midi: str | Path) -> Counter[int]:
    return Counter(
        message.note
        for message in mido.MidiFile(str(midi))
        if message.type == "note_on" and message.velocity > 0 and message.channel != 9
    )


def note_fidelity(performance_midi: str | Path, score: Score) -> NoteFidelity:
    played = performance_pitches(performance_midi)
    written = Counter(note.pitch for note in score.notes)
    common = sum((played & written).values())
    total_played, total_written = sum(played.values()), sum(written.values())
    return NoteFidelity(
        total_played,
        total_written,
        common / total_played if total_played else 0.0,
        common / total_written if total_written else 0.0,
    )

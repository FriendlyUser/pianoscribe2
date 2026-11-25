"""Is the written score the notes that were read from the video?

The last check of the video path, on its end product. The notes of the engraved score
(MusicXML, ties joined) are compared with the notes read from the picture, as two
multisets of pitches: no timing is involved, so the comparison holds whatever the beat
grid. With the rule-based writer both shares are 1.0 by construction; the check is there
so that this is measured on every job rather than assumed.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable

_STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def score_pitches(musicxml: str) -> Counter[int]:
    """Pitches of the notes of a score, a tied note counted once."""
    root = ET.fromstring(musicxml.encode("utf-8"))  # noqa: S314 - our own score writer's output
    pitches: Counter[int] = Counter()
    for note in root.iter("note"):
        pitch = note.find("pitch")
        if pitch is None or note.find("grace") is not None:
            continue
        if any(tie.get("type") == "stop" for tie in note.findall("tie")):
            continue
        octave = int(pitch.findtext("octave") or 4)
        alter = int(float(pitch.findtext("alter") or 0))
        pitches[(octave + 1) * 12 + _STEP[pitch.findtext("step") or "C"] + alter] += 1
    return pitches


def score_fidelity(musicxml: str, read: Iterable[int]) -> tuple[float, float]:
    """(share of the notes read that are in the score, share of the score's notes that were read)."""
    written = score_pitches(musicxml)
    seen = Counter(read)
    common = sum((written & seen).values())
    total_seen, total_written = sum(seen.values()), sum(written.values())
    return (
        common / total_seen if total_seen else 0.0,
        common / total_written if total_written else 0.0,
    )

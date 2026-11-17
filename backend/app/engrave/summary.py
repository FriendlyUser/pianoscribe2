"""Facts about a MusicXML score for display: key, meter, tempo, size."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass

_MAJOR = ["Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#"]
_MINOR = ["Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#"]


@dataclass(frozen=True)
class ScoreSummary:
    key: str  # of the opening, e.g. "F major"
    key_changes: int
    time_signature: str  # the prevailing one
    tempo_bpm: float  # first metronome mark, in the unit the score prints
    tempo_unit: str  # "quarter", "dotted quarter", "eighth", ...
    measures: int
    notes: int

    @property
    def text(self) -> str:
        return (
            f"{self.key} · {self.time_signature} · {self.tempo_unit} = {round(self.tempo_bpm)} · {self.measures} bars"
        )


def key_name(fifths: int, mode: str | None) -> str:
    fifths = max(-7, min(7, fifths))
    if mode == "minor":
        return f"{_MINOR[fifths + 7]} minor"
    return f"{_MAJOR[fifths + 7]} major"


def summarize(musicxml: str) -> ScoreSummary:
    root = ET.fromstring(musicxml.encode("utf-8"))  # noqa: S314 - our own score writer's output
    part = root.find("part")
    measures = part.findall("measure") if part is not None else []
    meters: Counter[str] = Counter()
    keys: list[int] = []
    mode: str | None = None
    current_meter = "4/4"
    tempo, unit = 0.0, "quarter"
    notes = 0
    for measure in measures:
        for attributes in measure.findall("attributes"):
            if attributes.find("time") is not None:
                current_meter = f"{attributes.findtext('time/beats')}/{attributes.findtext('time/beat-type')}"
            fifths = attributes.findtext("key/fifths")
            if fifths is not None and (not keys or keys[-1] != int(fifths)):
                keys.append(int(fifths))
                mode = mode or attributes.findtext("key/mode")
        meters[current_meter] += 1
        if tempo == 0.0:
            metronome = measure.find(".//metronome")
            if metronome is not None and metronome.findtext("per-minute"):
                try:
                    tempo = float(metronome.findtext("per-minute") or 0)
                except ValueError:
                    tempo = 0.0
                unit = (metronome.findtext("beat-unit") or "quarter").replace("-", " ")
                if metronome.find("beat-unit-dot") is not None:
                    unit = f"dotted {unit}"
        for note in measure.findall("note"):
            sounded = note.find("pitch") is not None and note.find("grace") is None
            if sounded and not any(t.get("type") == "stop" for t in note.findall("tie")):
                notes += 1
    if tempo == 0.0:
        sound = root.find(".//sound[@tempo]")
        tempo = float(sound.get("tempo") or 0.0) if sound is not None else 0.0
    return ScoreSummary(
        key=key_name(keys[0] if keys else 0, mode),
        key_changes=max(0, len(keys) - 1),
        time_signature=meters.most_common(1)[0][0] if meters else "4/4",
        tempo_bpm=tempo,
        tempo_unit=unit,
        measures=len(measures),
        notes=notes,
    )

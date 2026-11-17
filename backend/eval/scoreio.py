"""Minimal MusicXML reader: notes with their bar and position, plus bar metadata.

Deliberately independent of the code that writes scores, so the evaluation
cannot share a bug with the system under test.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

_STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


@dataclass(frozen=True)
class Note:
    bar: int  # 0-based index of the measure in reading order
    pos: Fraction  # quarter lengths from the bar line
    start: Fraction  # quarter lengths from the beginning of the score
    duration: Fraction  # quarter lengths, ties merged
    pitch: int
    staff: int


@dataclass(frozen=True)
class Bar:
    index: int
    start: Fraction  # quarter lengths from the beginning
    length: Fraction  # quarter lengths actually notated
    meter: tuple[int, int]
    fifths: int


@dataclass(frozen=True)
class Score:
    notes: list[Note]
    bars: list[Bar]
    tempo_bpm: float | None  # quarter notes per minute, first marking

    @property
    def length(self) -> Fraction:
        return self.bars[-1].start + self.bars[-1].length if self.bars else Fraction(0)


def _read_xml(path: Path) -> ET.Element:
    if path.suffix.lower() == ".mxl":
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if n.endswith(".xml") and not n.startswith("META-INF"))
            return ET.fromstring(archive.read(name))  # noqa: S314 - scores under evaluation, local files
    return ET.parse(path).getroot()  # noqa: S314


def read_musicxml(path: str | Path) -> Score:
    """Parse a partwise MusicXML file (all parts merged; ties joined; grace notes ignored)."""
    root = _read_xml(Path(path))
    notes: list[Note] = []
    bars: list[Bar] = []
    tempo: float | None = None

    for part_index, part in enumerate(root.findall("part")):
        divisions = 1
        meter = (4, 4)
        fifths = 0
        bar_start = Fraction(0)
        open_ties: dict[tuple[int, int], int] = {}  # (pitch, staff) -> index into notes
        for bar_index, measure in enumerate(part.findall("measure")):
            cursor = Fraction(0)
            furthest = Fraction(0)
            previous_onset = Fraction(0)
            for element in measure:
                tag = element.tag
                if tag == "attributes":
                    if element.findtext("divisions"):
                        divisions = int(element.findtext("divisions"))
                    if element.find("time") is not None:
                        meter = (int(element.findtext("time/beats")), int(element.findtext("time/beat-type")))
                    if element.find("key/fifths") is not None:
                        fifths = int(element.findtext("key/fifths"))
                elif tag == "direction" and tempo is None:
                    sound = element.find("sound")
                    if sound is not None and sound.get("tempo"):
                        tempo = float(sound.get("tempo"))
                elif tag == "sound" and tempo is None and element.get("tempo"):
                    tempo = float(element.get("tempo"))
                elif tag == "backup":
                    cursor -= Fraction(int(element.findtext("duration")), divisions)
                elif tag == "forward":
                    cursor += Fraction(int(element.findtext("duration")), divisions)
                    furthest = max(furthest, cursor)
                elif tag == "note":
                    if element.find("grace") is not None:
                        continue
                    duration = Fraction(int(element.findtext("duration") or 0), divisions)
                    is_chord = element.find("chord") is not None
                    onset = previous_onset if is_chord else cursor
                    pitch_el = element.find("pitch")
                    if pitch_el is not None:
                        pitch = (
                            (int(pitch_el.findtext("octave")) + 1) * 12
                            + _STEP[pitch_el.findtext("step")]
                            + int(float(pitch_el.findtext("alter") or 0))
                        )
                        staff = int(element.findtext("staff") or (part_index + 1))
                        ties = {t.get("type") for t in element.findall("tie")}
                        key = (pitch, staff)
                        if "stop" in ties and key in open_ties:
                            i = open_ties[key]
                            old = notes[i]
                            notes[i] = Note(old.bar, old.pos, old.start, old.duration + duration, old.pitch, old.staff)
                            if "start" not in ties:
                                del open_ties[key]
                        else:
                            notes.append(Note(bar_index, onset, bar_start + onset, duration, pitch, staff))
                            if "start" in ties:
                                open_ties[key] = len(notes) - 1
                    if not is_chord:
                        previous_onset = cursor
                        cursor += duration
                        furthest = max(furthest, cursor)
            nominal = Fraction(4 * meter[0], meter[1])
            length = furthest if furthest > 0 else nominal
            if part_index == 0:
                bars.append(Bar(bar_index, bar_start, length, meter, fifths))
            bar_start += length
    notes.sort(key=lambda n: (n.start, n.pitch))
    return Score(notes=notes, bars=bars, tempo_bpm=tempo)

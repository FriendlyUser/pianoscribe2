"""Verovio engraving: MusicXML in, paginated SVG + playback timing out.

One engine instance loads the canonical MusicXML once; every export (print
pages for the PDF, pages for the HTML view, MIDI, timing map) is rendered
from that single loaded document, which is what keeps the outputs consistent.
"""

from __future__ import annotations

import base64
import json
import threading
from dataclasses import dataclass
from typing import Any, Literal

from app.errors import JobError
from app.logging import get_logger

log = get_logger(__name__)

# Verovio page units are tenths of a millimetre at scale 100.
PAGE_SIZES: dict[str, tuple[int, int]] = {"A4": (2100, 2970), "Letter": (2159, 2794)}
PAGE_POINTS: dict[str, tuple[float, float]] = {"A4": (595.276, 841.890), "Letter": (612.0, 792.0)}

_BASE_OPTIONS: dict[str, Any] = {
    "scale": 100,
    "breaks": "auto",
    "header": "auto",
    "footer": "none",
    "svgViewBox": True,
    "pageMarginTop": 150,
    "pageMarginBottom": 150,
    "spacingSystem": 12,
    "spacingStaff": 10,
    "minLastJustification": 0.6,
    "breaksNoWidow": True,
    "xmlIdSeed": 1,
    "smuflTextFont": "none",
}

# Verovio keeps process-global state while loading; serialise access.
_lock = threading.Lock()


@dataclass(frozen=True)
class PlaybackNote:
    id: str  # SVG element id of the note
    pitch: int
    start_ms: float
    duration_ms: float


@dataclass(frozen=True)
class Engraving:
    print_pages: list[str]  # SVG, one per printed page
    compact_pages: list[str]  # SVG, narrow layout for small screens
    playback: list[PlaybackNote]
    midi: bytes


def _options(layout: Literal["print", "compact"], page_size: str) -> dict[str, Any]:
    width, height = PAGE_SIZES[page_size]
    if layout == "print":
        return {
            **_BASE_OPTIONS,
            "pageWidth": width,
            "pageHeight": height,
            "pageMarginLeft": 150,
            "pageMarginRight": 150,
        }
    # Narrow column, one tall page: reflows to fewer bars per system on phones.
    return {
        **_BASE_OPTIONS,
        "pageWidth": 1000,
        "pageHeight": 60000,
        "adjustPageHeight": True,
        "pageMarginLeft": 40,
        "pageMarginRight": 40,
        "pageMarginTop": 60,
        "pageMarginBottom": 60,
        "header": "none",
    }


def ottavas_in_reading_order(musicxml: str) -> str:
    """``musicxml`` with every octave line (8va, 8vb) written where Verovio looks for it.

    MusicXML lets a bar's directions stand anywhere in the bar, each with its own place in
    time, and a writer may put them all after the bar's notes. Verovio starts an octave line
    at the next note it reads, whichever staff that note is on: written after the notes, a
    line for the left hand is drawn a bar late and over the right hand's staff. Here a line's
    start is moved to just before the first note of its own staff in its bar, and its end to
    just after the last; nothing else is changed. Text with no octave lines is returned as it is.
    """
    if "octave-shift" not in musicxml:
        return musicxml
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(musicxml)  # noqa: S314 - our own score writer's output
    except ET.ParseError:
        return musicxml
    for part in root.iter("part"):
        for measure in part.findall("measure"):
            lines = [
                (child, shift.get("type"), child.findtext("staff") or "1")
                for child in measure
                if child.tag == "direction"
                for shift in [next(child.iter("octave-shift"), None)]
                if shift is not None
            ]
            for direction, kind, staff in lines:
                notes = [c for c in measure if c.tag == "note" and (c.findtext("staff") or "1") == staff]
                if not notes:
                    continue
                measure.remove(direction)
                at = list(measure).index(notes[-1]) + 1 if kind == "stop" else list(measure).index(notes[0])
                measure.insert(at, direction)
    declaration = musicxml[: musicxml.index("<score")] if "<score" in musicxml else ""
    return declaration + ET.tostring(root, encoding="unicode")


def engrave(musicxml: str, *, page_size: str = "A4") -> Engraving:
    """Render ``musicxml`` to SVG pages (two layouts), a timing map and MIDI."""
    import verovio

    musicxml = ottavas_in_reading_order(musicxml)
    with _lock:
        toolkit = verovio.toolkit()
        toolkit.setOptions(_options("print", page_size))
        if not toolkit.loadData(musicxml):
            raise JobError("engraving_failed", "The notation could not be engraved.", retryable=False)
        print_pages = [toolkit.renderToSVG(page) for page in range(1, toolkit.getPageCount() + 1)]

        timemap = toolkit.renderToTimemap({"includeRests": False, "includeMeasures": False})
        if isinstance(timemap, str):
            timemap = json.loads(timemap)
        playback: list[PlaybackNote] = []
        for entry in timemap:
            for note_id in entry.get("on", []):
                values = toolkit.getMIDIValuesForElement(note_id)
                if isinstance(values, str):
                    values = json.loads(values)
                if not values or "pitch" not in values:
                    continue
                playback.append(
                    PlaybackNote(
                        id=note_id,
                        pitch=int(values["pitch"]),
                        start_ms=float(values.get("time", entry["tstamp"])),
                        duration_ms=float(values.get("duration", 0.0)),
                    )
                )
        midi = base64.b64decode(toolkit.renderToMIDI())

        toolkit.setOptions(_options("compact", page_size))
        toolkit.redoLayout()
        compact_pages = [toolkit.renderToSVG(page) for page in range(1, toolkit.getPageCount() + 1)]

    log.info("engrave.done", pages=len(print_pages), playback_notes=len(playback))
    return Engraving(print_pages=print_pages, compact_pages=compact_pages, playback=playback, midi=midi)

"""Engraving layer: canonical MusicXML -> PDF, standalone HTML and MIDI."""

from __future__ import annotations

from dataclasses import dataclass

from app.engrave.html import render_html
from app.engrave.pdf import svg_pages_to_pdf
from app.engrave.verovio_engine import engrave


@dataclass(frozen=True)
class ScoreExports:
    pdf: bytes
    html: str
    midi: bytes
    pages: int


def export_musicxml(musicxml: str, *, title: str, summary: str, page_size: str = "A4") -> ScoreExports:
    """Produce every export from one loaded rendering of ``musicxml``."""
    engraving = engrave(musicxml, page_size=page_size)
    return ScoreExports(
        pdf=svg_pages_to_pdf(engraving.print_pages, page_size=page_size, title=title),
        html=render_html(engraving, title=title, summary=summary),
        midi=engraving.midi,
        pages=len(engraving.print_pages),
    )

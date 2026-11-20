"""Engraving layer: canonical MusicXML -> PDF, standalone HTML and MIDI."""

from __future__ import annotations

from dataclasses import dataclass

from app.engrave.html import render_html
from app.engrave.verovio_engine import engrave


@dataclass(frozen=True)
class ScoreExports:
    pdf: bytes | None  # None when PDF output is left to a desktop shell
    html: str
    midi: bytes
    pages: int


def svg_pages_to_pdf(pages: list[str], *, page_size: str, title: str) -> bytes:
    from app.engrave.pdf import svg_pages_to_pdf as render  # needs the native cairo library

    return render(pages, page_size=page_size, title=title)


def export_musicxml(
    musicxml: str, *, title: str, summary: str, page_size: str = "A4", pdf: bool = True
) -> ScoreExports:
    """Produce every export from one loaded rendering of ``musicxml``."""
    engraving = engrave(musicxml, page_size=page_size)
    return ScoreExports(
        pdf=svg_pages_to_pdf(engraving.print_pages, page_size=page_size, title=title) if pdf else None,
        html=render_html(engraving, title=title, summary=summary),
        midi=engraving.midi,
        pages=len(engraving.print_pages),
    )

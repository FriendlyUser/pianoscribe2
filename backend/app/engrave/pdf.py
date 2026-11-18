"""Print-ready PDF assembled from the engraved SVG pages."""

from __future__ import annotations

import base64
import io
import re
import subprocess
from functools import lru_cache
from pathlib import Path

from app.engrave.verovio_engine import PAGE_POINTS
from app.logging import get_logger

log = get_logger(__name__)
_font_ready = False


def install_music_font() -> None:
    """Expose Verovio's SMuFL text font (dynamics, tempo glyphs) to fontconfig.

    Verovio embeds the font in each SVG as a data URI, which browsers honour
    but the Cairo rasteriser does not, so the same font file is registered
    with the system font cache. Idempotent and cheap after the first call.
    """
    global _font_ready
    if _font_ready:
        return
    listed = subprocess.run(["fc-list", ":family=Leipzig"], check=False, capture_output=True, timeout=60)
    if listed.stdout.strip():  # already registered (the worker image installs it system-wide)
        _font_ready = True
        return
    import verovio

    font_dir = Path.home() / ".fonts"
    target = font_dir / "Leipzig.woff2"
    if not target.exists():
        css = (Path(verovio.__file__).parent / "data" / "Leipzig.css").read_text()
        match = re.search(r"base64,([A-Za-z0-9+/=]+)", css)
        if match is None:
            raise RuntimeError("Verovio music font not found in package data")
        font_dir.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(match.group(1)))
        subprocess.run(["fc-cache", "-f", str(font_dir)], check=False, capture_output=True, timeout=120)
    _font_ready = True


# Characters the score's text font (a Times-like serif) has no glyphs for: CJK punctuation, kana,
# ideographs, Hangul, and the full-width forms common in Japanese titles.
_WIDE = re.compile("[\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]")
_LEAF_TSPAN = re.compile(r"<tspan([^<>]*)>([^<>]+)</tspan>")
_CJK_FAMILIES = ("Noto Serif CJK JP", "Noto Sans CJK JP", "IPAexMincho", "IPAexGothic", "WenQuanYi Micro Hei")


@lru_cache(maxsize=1)
def cjk_font_family() -> str | None:
    """A font installed here that has both Latin and CJK glyphs, or None."""
    for family in _CJK_FAMILIES:
        try:
            found = subprocess.run(
                ["fc-list", f":family={family}", "family"], capture_output=True, text=True, timeout=30, check=False
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        if family.lower() in found.lower():
            return family
    return None


def with_cjk_font(svg: str, family: str | None) -> str:
    """Give text runs that contain CJK characters a font that has them.

    The rasteriser uses exactly the font it is told to, with no fallback for missing
    glyphs, so a Japanese title set in the score's serif font comes out blank.
    """
    if not family or not _WIDE.search(svg):
        return svg

    def fix(match: re.Match[str]) -> str:
        attributes, text = match.group(1), match.group(2)
        if not _WIDE.search(text):
            return match.group(0)
        attributes = re.sub(r'\s+font-family="[^"]*"', "", attributes)
        return f'<tspan{attributes} font-family="{family}">{text}</tspan>'

    return _LEAF_TSPAN.sub(fix, svg)


def svg_pages_to_pdf(pages: list[str], *, page_size: str, title: str) -> bytes:
    """Convert SVG pages to a single vector PDF at the physical page size."""
    import cairosvg
    from pypdf import PdfReader, PdfWriter

    install_music_font()
    # CairoSVG sizes its output in CSS pixels (96 per inch); PDF uses points (72 per inch).
    width, height = (points * 96.0 / 72.0 for points in PAGE_POINTS[page_size])
    writer = PdfWriter()
    family = cjk_font_family()
    for svg in pages:
        svg = with_cjk_font(svg, family)
        page_pdf = cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), output_width=width, output_height=height)
        writer.append(PdfReader(io.BytesIO(page_pdf)))
    writer.add_metadata({"/Title": title, "/Creator": "PianoScribe", "/Producer": "PianoScribe (Verovio, CairoSVG)"})
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()

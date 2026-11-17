"""Standalone, responsive HTML export of the engraved score.

The file is fully self-contained (inline SVG, CSS, JS and the music font), so
it can be opened offline or embedded by the web app. It carries two layouts
rendered from the same loaded score: the print pages (identical to the PDF)
for wide screens and printing, and a narrow reflowed column for phones.
"""

from __future__ import annotations

import html
import json
import re
from functools import lru_cache
from importlib import resources
from pathlib import Path

from app.engrave.verovio_engine import Engraving

_ID_ATTR = re.compile(r'\b(id)="([^"]+)"')
_HREF_ATTR = re.compile(r'\b((?:xlink:)?href)="#([^"]+)"')
_FONT_STYLE = re.compile(r"<style[^>]*>\s*@font-face.*?</style>", re.S)
_ROOT_ID = re.compile(r'<svg\b[^>]*\bid="([^"]+)"')
_STYLE_BLOCK = re.compile(r"<style[^>]*>.*?</style>", re.S)


@lru_cache
def _asset(name: str) -> str:
    return resources.files("app.engrave").joinpath("assets", name).read_text(encoding="utf-8")


@lru_cache
def _music_font_css() -> str:
    import verovio

    return (Path(verovio.__file__).parent / "data" / "Leipzig.css").read_text(encoding="utf-8")


def _strip_font(svg: str) -> str:
    """Drop the per-page embedded font; the document embeds it once."""
    return _FONT_STYLE.sub("", svg)


def _prefix_ids(svg: str, prefix: str) -> str:
    """Namespace element ids so two layouts can share one HTML document.

    Covers ``id`` attributes, ``href`` references and the id selectors in
    Verovio's per-document stylesheet (which carries, among others, the rule
    that gives staff lines and stems their stroke).
    """
    root = _ROOT_ID.search(svg)
    svg = _ID_ATTR.sub(lambda m: f'{m.group(1)}="{prefix}{m.group(2)}"', svg)
    svg = _HREF_ATTR.sub(lambda m: f'{m.group(1)}="#{prefix}{m.group(2)}"', svg)
    if root is not None:
        selector = re.compile(rf"#{re.escape(root.group(1))}(?![\w-])")
        svg = _STYLE_BLOCK.sub(lambda m: selector.sub(f"#{prefix}{root.group(1)}", m.group(0)), svg)
    return svg


def _pages(pages: list[str], prefix: str = "") -> str:
    out = []
    for number, svg in enumerate(pages, start=1):
        body = _strip_font(svg)
        if prefix:
            body = _prefix_ids(body, prefix)
        out.append(f'<section class="page" aria-label="Page {number}">{body}</section>')
    return "\n".join(out)


def render_html(engraving: Engraving, *, title: str, summary: str) -> str:
    """Build the standalone HTML document for an engraved score."""
    playback = [[n.id, n.pitch, round(n.start_ms, 1), round(n.duration_ms, 1)] for n in engraving.playback]
    data = json.dumps(playback, separators=(",", ":")).replace("</", "<\\/")
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="PianoScribe">
<title>{safe_title} - piano score</title>
<style>
{_music_font_css()}
{_asset("score.css")}
</style>
</head>
<body>
<header class="bar">
  <div class="meta"><h1>{safe_title}</h1><p>{html.escape(summary)}</p></div>
  <div class="controls">
    <button id="play" class="primary" type="button" aria-pressed="false">Play</button>
    <button id="stop" type="button">Stop</button>
    <label>Speed <input id="speed" type="range" min="40" max="150" step="5" value="100"><output id="speed-out">100%</output></label>
    <span id="clock" class="clock" aria-live="off">0:00 / 0:00</span>
    <button id="print" type="button">Print</button>
  </div>
</header>
<main>
<div class="layout layout-print">
{_pages(engraving.print_pages)}
</div>
<div class="layout layout-compact">
{_pages(engraving.compact_pages, prefix="c-")}
</div>
</main>
<script type="application/json" id="playback-data">{data}</script>
<script>
{_asset("score.js")}
</script>
</body>
</html>
"""

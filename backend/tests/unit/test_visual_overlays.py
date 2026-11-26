"""Bars that are one thing laid across the keyboard (a title banner), not notes."""

from __future__ import annotations

from app.visual.keyboard import Key, Keyboard
from app.visual.reader import OVERLAY_BARS, drop_overlays
from app.visual.roll import Bar

BLACK = {1, 3, 6, 8, 10}
KEYS = tuple(Key(p, p % 12 in BLACK, 10.0 * (p - 33), 10.0 * (p - 32)) for p in range(33, 97))  # A1 to C7
KEYBOARD = Keyboard(top=600, black_bottom=680, bottom=720, width=640, white_width=20.0, keys=KEYS, octave_certain=False)
WHITE = [i for i, key in enumerate(KEYS) if not key.black]
PINK = (250.0, 180.0, 250.0)


def bar(key: int, start: float, end: float) -> Bar:
    return Bar(key=key, start=start, end=end, colour=PINK)


def test_a_banner_over_every_white_key_is_dropped() -> None:
    # As read from a title banner scrolling down a screen recording: every white key from A1 to
    # B5, struck within two frames of each other (about 8 rows) and released together.
    banner = [bar(k, 100.0 + (3 * n) % 9, 600.0) for n, k in enumerate(WHITE[:30])]
    music = [bar(WHITE[5], 700.0, 760.0), bar(WHITE[9], 700.0, 760.0), bar(WHITE[12], 700.0, 790.0)]
    kept, dropped = drop_overlays(banner + music, KEYBOARD, tolerance=12.0)
    assert dropped == 30 and kept == music


def test_a_chord_of_ten_neighbouring_keys_is_kept() -> None:
    # Ten fingers on ten neighbouring white keys: unusual, but playable.
    cluster = [bar(k, 100.0, 300.0) for k in WHITE[10 : 10 + OVERLAY_BARS - 1]]
    assert drop_overlays(cluster, KEYBOARD, tolerance=12.0) == (cluster, 0)


def test_many_keys_struck_together_but_not_neighbours_are_kept() -> None:
    # Octaves and fifths over the whole keyboard, both hands and the sustain: keys left out between them.
    wide = [bar(k, 100.0, 300.0) for k in WHITE[::3][:14]]
    assert drop_overlays(wide, KEYBOARD, tolerance=12.0) == (wide, 0)


def test_a_glissando_is_kept() -> None:
    # Fifteen neighbouring white keys in a tenth of a second, one after the other.
    glissando = [bar(k, 100.0 + 0.8 * n, 108.0 + 0.8 * n) for n, k in enumerate(WHITE[14:29])]
    assert drop_overlays(glissando, KEYBOARD, tolerance=12.0) == (glissando, 0)


def test_bars_struck_together_but_released_apart_are_kept() -> None:
    # The same keys struck at once and let go one by one: a pedalled arpeggio as written, not a banner.
    released = [bar(k, 100.0, 300.0 + 40.0 * n) for n, k in enumerate(WHITE[3:17])]
    assert drop_overlays(released, KEYBOARD, tolerance=12.0) == (released, 0)

"""Bars that are one thing laid across the keyboard (a title banner), not notes."""

from __future__ import annotations

from app.visual.keyboard import Key, Keyboard
from app.visual.reader import OVERLAY_BARS, drop_overlays, merge_slivers
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


def test_a_sliver_cut_from_a_bar_is_joined_back() -> None:
    # As read from a 720p screen recording: one bar cut in two by a false seam, 3 rows from its end
    # (and another 3 rows from its start), next to two real repeated notes, which stay two.
    cut_at_the_end = [bar(WHITE[4], 100.0, 127.0), bar(WHITE[4], 127.0, 130.0)]
    cut_at_the_start = [bar(WHITE[9], 200.0, 203.0), bar(WHITE[9], 203.5, 260.0)]
    repeated = [bar(WHITE[12], 100.0, 120.0), bar(WHITE[12], 120.0, 140.0)]
    apart = [bar(WHITE[20], 100.0, 103.0), bar(WHITE[20], 110.0, 140.0)]  # short, but not touching: its own note
    kept, joined = merge_slivers(cut_at_the_end + cut_at_the_start + repeated + apart, shortest=8.0)
    assert joined == 2 and len(kept) == 6
    spans = sorted((b.key, b.start, b.end) for b in kept)
    assert (WHITE[4], 100.0, 130.0) in spans and (WHITE[9], 200.0, 260.0) in spans
    assert (WHITE[12], 100.0, 120.0) in spans and (WHITE[12], 120.0, 140.0) in spans
    assert (WHITE[20], 100.0, 103.0) in spans


def _strips(depth_at: dict[int, tuple[float, ...]], rows: int = 120):
    """A roll, zones and keyboard with one white key whose bar covers rows 10 to 110: its two corner
    strips fully covered but for the bar's rounded ends and the dips given (row -> cover of the rows from it)."""
    from types import SimpleNamespace

    import numpy as np

    cover = np.ones(rows, dtype=np.float32)
    cover[:10] = cover[110:] = 0.0
    cover[10:14] = (0.2, 0.5, 0.8, 0.9)
    cover[106:110] = (0.9, 0.8, 0.5, 0.2)
    for row, values in depth_at.items():
        cover[row : row + len(values)] = values
    present = cover > 0.1
    roll = SimpleNamespace(solid=np.stack([cover, cover, cover], axis=1))
    zones = SimpleNamespace(
        edge_left=np.array([0]), edge_right=np.array([1]), black_left=np.array([-1]), black_right=np.array([-1])
    )
    keyboard = SimpleNamespace(keys=[Key(60, False, 0.0, 40.0)])
    return roll, zones, keyboard, present


def test_a_faint_notch_has_whole_bar_on_both_sides(monkeypatch) -> None:
    from app.visual import roll as rolls

    # Forty rows in, a notch too faint to cut a bar on outright (the strips still half covered).
    roll, zones, keyboard, present = _strips({48: (0.8, 0.55, 0.5, 0.6, 0.85)})
    monkeypatch.setattr(rolls, "key_presence", lambda *_: (present[:, None], None))
    whole = Bar(key=0, start=10.0, end=110.0, colour=PINK)
    (found,) = rolls.faint_notches(roll, zones, keyboard, [whole], least=12, wide=8)
    assert found[0] == 0 and found[1] == 50.0 and abs(found[2] - 0.5) < 1e-6
    # The bar's own rounded ends are no notch, however low the strips fall there.
    assert rolls.faint_notches(roll, zones, keyboard, [Bar(0, 10.0, 46.0, PINK)], least=12, wide=8) == []
    # Nor is a dip where a neighbouring key's bar lies over the corner strips.
    monkeypatch.setattr(rolls, "key_presence", lambda *_: (np_ones_like(present), None))
    zones.black_left[0] = zones.black_right[0] = 0
    assert rolls.faint_notches(roll, zones, keyboard, [whole], least=12, wide=8) == []


def np_ones_like(present):
    import numpy as np

    return np.ones((len(present), 1), dtype=bool)


def test_a_faint_seam_needs_a_second_sign() -> None:
    from types import SimpleNamespace

    import numpy as np

    from app.visual.reader import split_at_faint_seams

    roll = SimpleNamespace(clock_of=lambda row: np.asarray(row, dtype=np.float64))
    long = Bar(key=WHITE[4], start=100.0, end=160.0, colour=PINK)
    other = Bar(key=WHITE[9], start=100.0, end=160.0, colour=PINK)
    short = Bar(key=WHITE[12], start=100.0, end=120.0, colour=PINK)
    double = Bar(key=WHITE[12], start=120.0, end=160.0, colour=PINK)  # touches ``short``, twice its length
    bars = [long, other, short, double]

    def cut(notches: list[tuple[int, float, float]]) -> tuple[int, dict[str, int]]:
        out, signs = split_at_faint_seams(bars, notches, roll, 30.0, None, together=2.0)
        return len(out) - len(bars), signs

    assert cut([(0, 130.0, 0.35)]) == (0, {"deep": 0, "together": 0, "run": 0, "heard": 0})  # faint, and alone
    assert cut([(0, 130.0, 0.6)])[1]["deep"] == 1  # deep enough by itself
    assert cut([(0, 130.0, 0.35), (1, 131.0, 0.3)]) == (
        2,
        {"deep": 0, "together": 2, "run": 0, "heard": 0},
    )  # a chord again
    assert cut([(3, 140.0, 0.35)])[1]["run"] == 1  # one more of a run of repeated notes
    assert cut([(3, 150.0, 0.35)])[0] == 0  # not where the run would put it


def test_an_attack_is_a_jump_in_the_sound_not_its_ripple() -> None:
    import numpy as np

    from app.visual.audio import attack_times

    rate = 22050
    time = np.arange(int(4.0 * rate)) / rate
    sound = np.zeros_like(time)
    for struck in (0.5, 1.5, 2.5):  # three notes, each ringing on with a slow tremolo for a second
        on = (time >= struck) & (time < struck + 0.9)
        ringing = np.exp(-2.0 * (time[on] - struck)) * (1.0 + 0.3 * np.sin(2 * np.pi * 9.0 * (time[on] - struck)))
        release = np.clip((struck + 0.9 - time[on]) / 0.1, 0.0, 1.0)  # let go gently: a key released is no attack
        sound[on] += ringing * release * np.sin(2 * np.pi * 440.0 * time[on])
    found = attack_times(sound.astype(np.float32), rate)
    assert len(found) == 3 and all(abs(a - b) < 0.03 for a, b in zip(found, (0.5, 1.5, 2.5), strict=True))

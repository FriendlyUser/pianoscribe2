"""What a roll made from a score shows of it, and what the writer makes of that (model_scripts).

Its bar lines (``steady_beats.drawn_beats``), the chords it spreads (``exact_score.gather_spread``),
the note values of a bar counted in threes, and the key signatures the notes themselves call for
(``notate_exact.signatures_by_the_notes``).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model_scripts"))

from fractions import Fraction  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from exact_score import TICKS, Bar, gather_spread, plan_score, split_value  # noqa: E402
from notate_exact import mark_tempos, signatures_by_the_notes  # noqa: E402
from steady_beats import drawn_beats  # noqa: E402

EIGHTH = 0.4  # seconds: a bar of 6/8 lasts 2.4
BAR = 6 * EIGHTH
LATE = 0.021  # how far after its bar line a note struck on it is read (three rows of the roll)
# One bar in eighths: the left hand on the two beats, a melody that divides the first into a dotted figure.
FIGURE = [("left", 41, 0.0, 3.0), ("left", 48, 3.0, 3.0)] + [
    ("right", pitch, at, length)
    for pitch, at, length in (
        (65, 0.0, 1.5),
        (67, 1.5, 0.5),
        (69, 2.0, 1.0),
        (72, 3.0, 1.0),
        (70, 4.0, 1.0),
        (69, 5.0, 1.0),
    )
]


def six_eight(bars: int, *, start: float = 4.0, seed: int = 5) -> tuple[list[dict], list[float], dict]:
    """A piece in 6/8 as a video at thirty frames a second shows it: (notes, bar lines drawn, what the tracker heard).

    The tracker hears the bar lines to a few hundredths of a second, most bars in six eighths, the
    first five in three quarters, and here and there half a bar as a bar of its own.
    """
    rng = random.Random(seed)
    lines = [start + BAR * k for k in range(bars + 1)]
    notes = []
    for line in lines[:-1]:
        for hand, pitch, at, length in FIGURE:
            struck = line + LATE + at * EIGHTH + rng.uniform(-0.008, 0.008)
            notes.append({"start": struck, "end": struck + length * EIGHTH - 0.05, "pitch": pitch, "hand": hand})
    tracked, beats = [], []
    for k, line in enumerate(lines[:-1]):
        heard = line + LATE + rng.uniform(-0.03, 0.03)
        if k < 5:
            tracked.append({"start": heard, "beats": 3, "unit": 4, "meter": "3/4"})
            beats += [heard + 2 * EIGHTH * n for n in range(3)]
        elif k % 9 == 7:
            tracked += [{"start": heard + 3 * EIGHTH * half, "beats": 3, "unit": 8, "meter": "3/8"} for half in (0, 1)]
            beats += [heard + EIGHTH * n for n in range(6)]
        else:
            tracked.append({"start": heard, "beats": 6, "unit": 8, "meter": "6/8"})
            beats += [heard + EIGHTH * n for n in range(6)]
    return notes, lines, {"beats": beats, "bars": tracked}


def test_the_bars_a_roll_draws_are_the_bars_of_the_score() -> None:
    notes, lines, heard = six_eight(24)
    steady = drawn_beats(heard, [n["start"] for n in notes], lines)
    assert steady is not None and steady.drawn is not None
    assert steady.drawn["meter"] == "6/8" and steady.drawn["bar_lines_put_in"] == 0
    assert abs(steady.drawn["lines_from_the_notes_ms"] - 1000 * LATE) < 3.0
    # Every bar is a bar of six eighths, each beat a sixth of what is drawn, where the notes are.
    assert [(bar["beats"], bar["unit"]) for bar in steady.bars] == [(6, 8)] * len(lines)
    assert len(steady.beats) == 6 * (len(lines) - 1) + 1 and all(steady.fitted)
    for k, line in enumerate(lines[:-1]):
        assert all(abs(steady.beats[6 * k + n] - (line + LATE + n * EIGHTH)) < 0.004 for n in range(6))
    # Written on them the figure is the same in every bar, whatever the tracker made of it.
    plan = plan_score(notes, heard, lines=lines)
    assert [(bar.beats, bar.unit) for bar in plan.bars] == [(6, 8)] * 24
    one_bar = [(w.staff, w.onset, w.kind, w.dots, w.tied) for w in plan.written if w.bar == 0]
    assert all(
        [(w.staff, w.onset, w.kind, w.dots, w.tied) for w in plan.written if w.bar == k] == one_bar for k in range(24)
    )
    upper = [(onset, kind, dots) for staff, onset, kind, dots, _ in one_bar if staff == 1]
    assert upper[:3] == [(Fraction(0), "eighth", 1), (Fraction(3, 4), "16th", 0), (Fraction(1), "eighth", 0)]
    # Without the lines the same notes come out in the tracker's bars: three quarters, then six eighths.
    assert len({(bar.beats, bar.unit) for bar in plan_score(notes, heard).bars}) > 1


def test_lines_that_are_not_heard_as_bars_are_not_the_bars() -> None:
    # A roll made from a file that counts four quarters to the bar whatever the music does: the
    # music's bars (six eighths, as heard) are three quarters long, and every other line falls mid-bar.
    notes, lines, heard = six_eight(24)
    every_four_quarters = [lines[0] + 8 * EIGHTH * k for k in range(18)]
    assert drawn_beats(heard, [n["start"] for n in notes], every_four_quarters) is None
    # No lines at all, or too few to say anything: the tracker's bars as before.
    assert drawn_beats(heard, [n["start"] for n in notes], lines[:3]) is None
    assert plan_score(notes, heard, lines=[]).steady.drawn is None


def test_a_line_that_was_not_read_is_put_in_and_a_roll_missing_many_is_not_relied_on() -> None:
    notes, lines, heard = six_eight(30)
    onsets = [n["start"] for n in notes]
    whole = drawn_beats(heard, onsets, lines)
    # Two lines the video's compression took, and one read twice.
    read = [t for k, t in enumerate(lines) if k not in (9, 17)] + [lines[4] + 0.04]
    mended = drawn_beats(heard, onsets, read)
    assert whole is not None and mended is not None and mended.drawn is not None
    assert mended.drawn["bar_lines_put_in"] == 2 and len(mended.bars) == len(whole.bars)
    assert max(abs(a - b) for a, b in zip(mended.beats, whole.beats, strict=True)) < 0.03
    # A third of them missing: such a roll's lines are not relied on.
    assert drawn_beats(heard, onsets, [t for k, t in enumerate(lines) if k % 3]) is None


def test_a_chord_the_roll_spreads_is_one_chord() -> None:
    beats = [EIGHTH * k for k in range(13)]

    def note(start: float, pitch: int, lasts: float, hand: str = "left") -> dict:
        return {"start": start, "end": start + lasts, "pitch": pitch, "hand": hand}

    notes = [
        note(0.0, 46, 2.0), note(0.029, 53, 1.97), note(0.058, 57, 1.94), note(0.160, 62, 1.84),  # spread, slowly
        note(0.8, 60, 0.09, "right"), note(0.9, 62, 0.09, "right"), note(1.0, 64, 0.3, "right"),  # a run: each gives way
        note(1.6, 48, 0.7), note(1.614, 49, 0.7),  # a black key read two rows late: one moment as it is
        note(2.4, 60, 0.7, "right"), note(2.45, 48, 0.7),  # the two hands are two hands
    ]  # fmt: skip
    onsets = [n["start"] for n in notes]
    assert gather_spread(onsets, notes, beats, 8, 0.04) == [[0, 1, 2, 3]]
    assert onsets[:4] == [0.0] * 4 and onsets[4:] == [n["start"] for n in notes[4:]]
    # In a score with its bar lines drawn, the chord is written on the beat with the sign for it.
    piece, lines, heard = six_eight(12)
    for n in piece:
        if n["hand"] == "left" and abs((n["start"] - lines[0] - LATE) % BAR) < 0.02:
            n["end"] = n["start"] + 1.0
    rolled = [dict(n) for n in piece] + [
        {"start": line + LATE + 0.045 * (k + 1), "end": line + LATE + 1.0, "pitch": 48 + 4 * k, "hand": "left"}
        for line in lines[:-1]
        for k in range(2)
    ]
    plan = plan_score(rolled, heard, lines=lines)
    assert plan.spread == 12
    first = [w for w in plan.written if w.staff == 2 and w.onset == 0]
    assert all(w.spread and w.pitches == (41, 48, 52) for w in first) and len(first) == 12
    assert not any(w.spread for w in plan.written if w.onset != 0)


def test_two_dotted_eighths_share_a_beat_of_six_eight() -> None:
    bar = Bar(0, 6 * TICKS, 6, 8, 0.0)
    # The second half of a beat of three eighths: written whole, not as a sixteenth tied to an eighth.
    assert [(v.start, v.kind, v.dots) for v in split_value(18, 36, [bar])] == [(18, "eighth", 1)]
    assert [(v.start, v.kind, v.dots) for v in split_value(36 + 18, 72, [bar])] == [(54, "eighth", 1)]
    # So is the dotted eighth after a sixteenth. A note over the middle of the bar is still tied there.
    assert [(v.start, v.kind, v.dots) for v in split_value(6, 24, [bar])] == [(6, "eighth", 1)]
    assert [(v.start, v.ticks) for v in split_value(24, 48, [bar])] == [(24, 12), (36, 12)]
    # In four quarters nothing changes: a note that hides a beat is tied over it.
    assert len(split_value(18, 36, [Bar(0, 4 * TICKS, 4, 4, 0.0)])) == 2


def test_a_bar_of_six_eighths_is_counted_in_dotted_quarters() -> None:
    measures = [SimpleNamespace(time_signature=(6, 8), length=Fraction(3), directions=[], tempos=[]) for _ in range(20)]
    # A recording that drops a frame now and then: every fifth bar a frame short, the others a little long.
    lengths = [2.36 if k % 5 == 4 else 2.41 for k in range(20)]
    starts = [4.0 + sum(lengths[:k]) for k in range(20)]
    steady = SimpleNamespace(stretches=[], followed=None, drawn={"scatter_ms": 8.0})
    assert mark_tempos(measures, starts, 4.0 + sum(lengths), steady) == [(0, 50)]
    assert measures[0].directions == [(Fraction(0), "metronome_quarter._50")]
    assert not any(measure.directions for measure in measures[1:])
    # Played, the bars take the time they took: 75 quarters a minute. The usual bar's tempo (74.7)
    # would end the piece a fifth of a second after the video does.
    assert len(measures[0].tempos) == 1 and abs(measures[0].tempos[0][1] - 60.0 * 60 / sum(lengths)) < 1e-9
    assert not any(measure.tempos for measure in measures[1:])


def test_key_signatures_are_the_ones_that_save_most_accidentals() -> None:
    f_major, d_major, c_major = [53, 55, 57, 58, 60, 62, 64], [50, 52, 54, 55, 57, 59, 61], [48, 50, 52, 53, 55, 57, 59]
    # Eight bars the tracker never heard as another key, every one with F sharp and C sharp in it:
    # in one flat they take six accidentals a bar. And where it heard the piece return to F, eight bars
    # early, the notes still have B natural.
    notes = [*[f_major * 2] * 12, *[d_major * 2] * 8, *[c_major * 2] * 16, *[f_major * 2] * 10]  # two octaves a bar
    heard = [*[-1] * 28, *[0] * 8, *[-1] * 10]
    assert signatures_by_the_notes(notes, heard) == [*[-1] * 12, *[2] * 8, *[0] * 16, *[-1] * 10]
    # A bridge the tracker hears in B major between two stretches of E: its notes have A natural three
    # times and A sharp never. The signature stays.
    e_major = [64, 66, 68, 69, 71, 73, 75]
    bridge = [[64, 68, 71, 75], [66, 69, 73], [64, 69, 71], [68, 69, 75]]
    assert (
        signatures_by_the_notes([*[e_major] * 8, *bridge, *[e_major] * 4], [*[4] * 8, *[5] * 4, *[4] * 4]) == [4] * 16
    )
    # A change the notes go along with stands; one they say nothing about is not made on hearsay.
    a_flat, b_flat = [56, 58, 60, 61, 63, 65, 67], [58, 60, 62, 63, 65, 67, 69]
    up_a_tone = [*[-4] * 6, *[-2] * 6]
    assert signatures_by_the_notes([*[a_flat * 2] * 6, *[b_flat * 2] * 6], up_a_tone) == up_a_tone
    assert signatures_by_the_notes([[64, 71]] * 16, [*[4] * 8, *[5] * 8]) == [4] * 16
    # Two bars of another harmony, however many accidentals, are written with accidentals.
    chromatic = [54, 56, 58, 59, 61, 63, 65] * 2
    assert signatures_by_the_notes([*[f_major] * 8, chromatic, chromatic, *[f_major] * 8], [-1] * 18) == [-1] * 18
    # With nothing between them to choose by, the one heard; with nothing heard either, the plainer.
    assert signatures_by_the_notes([[60, 64, 67]] * 8, [1] * 8) == [1] * 8
    assert signatures_by_the_notes([], []) == []

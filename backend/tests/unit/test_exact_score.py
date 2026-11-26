"""Writing exactly timed notes down by rule (model_scripts/exact_score.py)."""

from __future__ import annotations

import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model_scripts"))

from exact_score import (  # noqa: E402
    TICKS,
    Bar,
    Chord,
    group_chords,
    layout_bars,
    plan_score,
    plan_voices,
    quantize_onsets,
    settle_lengths,
    snap_beats,
    split_value,
)

from tests import synth  # noqa: E402

FOUR_FOUR = [Bar(start=48 * i, ticks=48, beats=4, unit=4, seconds=2.0 * i) for i in range(3)]


def kinds(start: int, end: int, bars: list[Bar] = FOUR_FOUR) -> list[tuple[int, str, int, bool]]:
    return [(v.start, v.kind, v.dots, v.triplet) for v in split_value(start, end, bars)]


def test_a_beat_moves_onto_the_note_struck_on_it() -> None:
    beats = [0.0, 0.5, 1.0, 1.5, 2.0]
    onsets = [0.012, 0.26, 0.488, 1.51]  # on beats 0, 1 and 3; one between beats; none on beats 2 and 4
    snapped = snap_beats(beats, onsets)
    assert snapped[0] == 0.012 and snapped[1] == 0.488 and snapped[3] == 1.51
    assert snapped[1] < snapped[2] < snapped[3]  # a beat with no note keeps its place between its neighbours
    assert abs(snapped[2] - 0.999) < 0.01


def test_notes_take_the_simplest_grid_that_fits_their_beat() -> None:
    assert quantize_onsets([0.0, 0.5, 1.0, 1.25, 1.75]) == [0, 6, 12, 15, 21]  # halves, then quarters
    assert quantize_onsets([2.0, 2.33, 2.67]) == [24, 28, 32]  # thirds: a triplet
    assert quantize_onsets([0.98, 1.01]) == [12, 12]  # a hair before or after the beat is on it
    assert quantize_onsets([3.0, 3.17, 3.5]) == [36, 38, 42]  # sixths


def test_note_values_show_the_beats() -> None:
    assert kinds(0, 12) == [(0, "quarter", 0, False)]
    assert kinds(0, 18) == [(0, "quarter", 1, False)]  # dotted quarter from the bar line
    assert kinds(6, 18) == [(6, "quarter", 0, False)]  # eighth, quarter, eighth: a syncopation written whole
    assert kinds(18, 30) == [(18, "eighth", 0, False), (24, "eighth", 0, False)]  # but not over the middle of the bar
    assert kinds(3, 9) == [(3, "eighth", 0, False)]  # sixteenth, eighth, sixteenth inside one beat
    assert kinds(3, 12) == [(3, "eighth", 1, False)]  # sixteenth, dotted eighth
    assert kinds(9, 15) == [(9, "16th", 0, False), (12, "16th", 0, False)]  # off the beat and across it: tied
    assert kinds(12, 48) == [(12, "quarter", 0, False), (24, "half", 0, False)]  # shows the middle of the bar
    assert kinds(0, 48) == [(0, "whole", 0, False)]
    assert kinds(36, 60) == [(36, "quarter", 0, False), (48, "quarter", 0, False)]  # tied over the bar line
    assert kinds(0, 15) == [(0, "quarter", 0, False), (12, "16th", 0, False)]


def test_triplets_are_written_inside_their_beat() -> None:
    assert kinds(0, 8) == [(0, "quarter", 0, True)]  # two thirds of a beat
    assert kinds(8, 12) == [(8, "eighth", 0, True)]
    # From the last third of a beat for a beat and a half: a triplet eighth tied to a dotted quarter.
    assert kinds(20, 42) == [(20, "eighth", 0, True), (24, "quarter", 1, False)]


def test_a_three_four_bar_and_a_compound_bar() -> None:
    waltz = [Bar(0, 36, 3, 4, 0.0)]
    assert kinds(0, 36, waltz) == [(0, "half", 1, False)]
    assert kinds(12, 36, waltz) == [(12, "half", 0, False)]
    six_eight = [Bar(0, 72, 6, 8, 0.0)]  # the beat is the eighth note
    assert kinds(0, 36, six_eight) == [(0, "quarter", 1, False)]
    assert kinds(24, 48, six_eight) == [(24, "eighth", 0, False), (36, "eighth", 0, False)]  # shows the second half


def test_a_held_note_and_a_moving_line_are_two_voices() -> None:
    held = [(0, 46, 48)]
    line = [(0, 11, 60), (12, 23, 62), (24, 35, 64), (36, 47, 65)]
    voices = plan_voices(group_chords(held + line))
    assert len(voices) == 2
    assert [c.pitches for c in voices[0]] == [[60], [62], [64], [65]]  # the upper line is the first voice
    for voice in voices:
        settle_lengths(voice, set())
    assert [(c.start, c.end) for c in voices[0]] == [(0, 12), (12, 24), (24, 36), (36, 48)]
    assert [(c.start, c.end) for c in voices[1]] == [(0, 48)]


def test_notes_struck_and_held_together_are_one_chord() -> None:
    chords = group_chords([(0, 11, 60), (0, 10, 64), (0, 11, 67), (12, 20, 62)])
    assert [(c.start, c.pitches) for c in chords] == [(0, [60, 64, 67]), (12, [62])]


def test_a_note_lasts_to_the_next_unless_a_rest_follows() -> None:
    voice = [Chord(0, 4, [60]), Chord(6, 11, [62]), Chord(12, 17, [64]), Chord(48, 60, [65])]
    settle_lengths(voice, set())
    # Eighths played short are eighths; the third note is a short quarter-beat note before a long rest.
    assert [(c.start, c.end) for c in voice[:3]] == [(0, 6), (6, 12), (12, 18)]


def test_a_pickup_bar_before_the_first_bar_line() -> None:
    beats = [0.5 * i for i in range(12)]
    annotation = {"bars": [{"start": 1.0, "beats": 4, "unit": 4}, {"start": 3.0, "beats": 4, "unit": 4}]}
    bars = layout_bars(annotation, beats, first_tick=12, last_tick=110)
    assert [(b.start, b.ticks, b.beats) for b in bars] == [(12, 12, 4), (24, 48, 4), (72, 48, 4)]
    # Bars the tracker counts on after the last note are not written.
    annotation["bars"] += [{"start": 5.0, "beats": 4, "unit": 4}, {"start": 7.0, "beats": 4, "unit": 4}]
    assert len(layout_bars(annotation, [0.5 * i for i in range(20)], first_tick=12, last_tick=110)) == 3


def test_ode_to_joy_is_written_as_it_is() -> None:
    beat = 60.0 / synth.ODE_BPM
    notes = [
        {
            "start": 1.0 + n.start_beat * beat + 0.004,  # a few milliseconds off, as a reading is
            "end": 1.0 + (n.start_beat + 0.9 * n.duration_beats) * beat,
            "pitch": n.pitch,
            "hand": "right" if n.velocity == 92 else "left",
        }
        for n in synth.ode_to_joy()
    ]
    tracked = [1.0 + beat * i + (0.011 if i % 2 else -0.009) for i in range(34)]  # a beat tracker's jitter
    annotation = {"beats": tracked, "bars": [{"start": tracked[4 * i], "beats": 4, "unit": 4} for i in range(8)]}
    plan = plan_score(notes, annotation)
    assert len(plan.bars) == 8 and plan.off_grid == 0 and plan.cut_short == 0
    assert plan.voices == {1: 1, 2: 1}
    written = sorted(plan.written, key=lambda w: (w.bar, w.onset, w.staff))
    assert sum(len(w.pitches) for w in written) == len(notes)  # no ties needed, so one value per note
    melody = [(w.bar, w.onset, w.kind, w.dots, w.pitches) for w in written if w.staff == 1]
    assert melody[:4] == [(0, F(0), "quarter", 0, (71,)), (0, F(1), "quarter", 0, (71,)),
                          (0, F(2), "quarter", 0, (72,)), (0, F(3), "quarter", 0, (74,))]  # fmt: skip
    # Bar 4: dotted quarter, eighth, half.
    assert [(m[1], m[2], m[3]) for m in melody if m[0] == 3] == [
        (F(0), "quarter", 1),
        (F(3, 2), "eighth", 0),
        (F(2), "half", 0),
    ]
    bass = [w for w in written if w.staff == 2]
    assert {w.pitches for w in bass if w.bar == 3} == {(50, 57), (50, 54)}  # two-note chords in the left hand
    assert all(not w.triplet for w in written) and TICKS == 12

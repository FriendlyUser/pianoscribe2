"""The fixes applied around the score model (model_scripts/score_fixes.py)."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model_scripts"))

from score_fixes import complete_meters, drop_warmup, plan_warmup, warmed_annotation  # noqa: E402

BEAT = 0.36
DOWNBEATS = [round(0.54 + i * 6 * BEAT, 4) for i in range(8)]
BEATS = [0.18] + [round(0.54 + i * BEAT, 4) for i in range(48)]
ANNOTATION = {
    "duration": 20.0,
    "beats": BEATS,
    "downbeats": DOWNBEATS,
    "bars": [{"start": d, "beats": 6, "unit": 8, "meter": "6/8"} for d in DOWNBEATS],
    "melody": [
        {"start": round(0.54 + i * BEAT, 4), "end": round(0.54 + i * BEAT + 0.3, 4), "pitch": 60 + i % 12}
        for i in range(40)
    ],
}


def test_bars_without_a_meter_take_their_neighbours_unit_and_their_own_length() -> None:
    # An anacrusis of two beats before five labelled 6/8 bars (0.48 "beat seconds" per beat).
    starts = [-0.96, 0.0, 2.88, 5.76, 8.64, 11.52]
    meters = [None, (6, 8), (6, 8), (6, 8), None, (6, 8)]
    assert complete_meters(meters, starts, 0.48) == [(2, 8), (6, 8), (6, 8), (6, 8), (6, 8), (6, 8)]
    assert complete_meters([(4, 4), (4, 4)], [0.0, 1.92], 0.48) == [(4, 4), (4, 4)]
    assert complete_meters([None, None], [0.0, 1.92], 0.48) == [None, None]  # nothing known: leave the model free


def test_warmup_spans_the_first_two_full_bars() -> None:
    assert plan_warmup(ANNOTATION, first_note=0.54) == (DOWNBEATS[0], DOWNBEATS[2])
    # A piece that starts with silence: the stretch begins at the bar of the first note.
    assert plan_warmup(ANNOTATION, first_note=DOWNBEATS[3] + 0.4) == (DOWNBEATS[3], DOWNBEATS[5])


def test_warmup_with_an_anacrusis_starts_at_its_beat() -> None:
    start, end = plan_warmup(ANNOTATION, first_note=0.25)
    assert start == 0.18 and end == DOWNBEATS[1]  # the pickup and one full bar


def test_warmup_is_skipped_for_very_short_pieces() -> None:
    short = {**ANNOTATION, "downbeats": DOWNBEATS[:3]}
    assert plan_warmup(short, first_note=0.54) is None


def test_warmed_annotation_repeats_the_opening_then_shifts_everything() -> None:
    start, end = DOWNBEATS[0], DOWNBEATS[2]
    shift = end - start
    warmed = warmed_annotation(ANNOTATION, start, end)
    assert warmed["duration"] == ANNOTATION["duration"] + shift
    # Two copies of the opening's downbeats, then every original downbeat, later by the shift.
    expected = DOWNBEATS[:2] + [round(d + shift, 4) for d in DOWNBEATS]
    assert [round(d, 4) for d in warmed["downbeats"]] == expected
    assert len(warmed["bars"]) == len(ANNOTATION["bars"]) + 2
    assert all(
        b - a > 0.1 for a, b in zip(warmed["beats"], warmed["beats"][1:], strict=False)
    )  # strictly increasing grid
    periods = [b - a for a, b in zip(warmed["beats"][1:], warmed["beats"][2:], strict=False)]
    assert max(periods) - min(periods) < 1e-6  # the seam keeps the pulse
    opening_notes = [n for n in ANNOTATION["melody"] if n["start"] < end]
    assert len(warmed["melody"]) == len(ANNOTATION["melody"]) + len(opening_notes)
    assert ANNOTATION["downbeats"] == DOWNBEATS  # the input is not modified


def test_warmed_annotation_marks_the_seam_before_an_anacrusis() -> None:
    start, end = plan_warmup(ANNOTATION, first_note=0.25)
    warmed = warmed_annotation(ANNOTATION, start, end)
    # The repeated pickup must begin its own (short) bar instead of extending the bar before it.
    assert any(abs(d - end) < 1e-6 for d in warmed["downbeats"])


@dataclass
class Group:
    onset: F
    staff: int = 1
    voice: int = 1
    slur_stop: int = 0
    slur_start: int = 0


@dataclass
class Measure:
    length: F = F(3)
    groups: list[Group] = field(default_factory=list)
    directions: list[tuple[F, str]] = field(default_factory=list)
    tempos: list[tuple[F, float]] = field(default_factory=list)


def test_dropping_warmup_measures_carries_tempo_and_dynamic_forward() -> None:
    measures = [
        Measure(directions=[(F(0), "metronome_quarter_84"), (F(0), "dyn_mp")], tempos=[(F(0), 84.0)]),
        Measure(),
        Measure(groups=[Group(F(0))]),
        Measure(),
    ]
    kept = drop_warmup(measures, 2)
    assert len(kept) == 2
    assert (F(0), "metronome_quarter_84") in kept[0].directions and (F(0), "dyn_mp") in kept[0].directions
    assert kept[0].tempos == [(F(0), 84.0)]
    assert drop_warmup(measures, 0) is measures


def test_dropping_keeps_an_existing_marking_on_the_first_kept_measure() -> None:
    measures = [Measure(directions=[(F(0), "dyn_p")]), Measure(directions=[(F(0), "dyn_f")], tempos=[(F(0), 60.0)])]
    kept = drop_warmup(measures, 1)
    assert [name for _, name in kept[0].directions if name.startswith("dyn_")] == ["dyn_f"]
    assert kept[0].tempos == [(F(0), 60.0)]


def test_lines_and_slurs_that_started_in_dropped_measures_are_closed_cleanly() -> None:
    measures = [
        Measure(groups=[Group(F(0), slur_start=1)], directions=[(F(0), "pedal_start"), (F(1), "wedge_crescendo")]),
        Measure(
            groups=[Group(F(0), slur_stop=1), Group(F(1), slur_start=1), Group(F(2), slur_stop=1)],
            directions=[(F(0), "pedal_stop"), (F(1), "wedge_stop"), (F(2), "pedal_start")],
        ),
        Measure(directions=[(F(1), "pedal_stop")]),
    ]
    kept = drop_warmup(measures, 1)
    first = kept[0]
    assert [g.slur_stop for g in first.groups] == [0, 0, 1]  # the orphan stop is gone, the complete slur stays
    assert [name for _, name in first.directions] == ["pedal_start"]
    assert [name for _, name in kept[1].directions] == ["pedal_stop"]


def _annotation(bar_beats: list[int], unit: int = 4, period: float = 0.4, first: float = 1.0) -> dict:
    beats, bars, t = [], [], first
    for n in bar_beats:
        bars.append({"start": round(t, 4), "beats": n, "unit": unit, "meter": f"{n}/{unit}"})
        for _ in range(n):
            beats.append(round(t, 4))
            t += period
    return {
        "duration": t + 1,
        "beats": [round(first - period, 4), *beats],
        "downbeats": [b["start"] for b in bars],
        "bars": bars,
        "melody": [],
    }


def test_double_length_bars_are_split_where_the_short_bar_is_common() -> None:
    from score_fixes import regularize_bars

    annotation = _annotation([3, 6, 6, 3, 3, 6, 3])
    fixed = regularize_bars(annotation)
    assert [b["beats"] for b in fixed["bars"]] == [3] * 10
    assert {b["meter"] for b in fixed["bars"]} == {"3/4"}
    assert fixed["downbeats"] == [b["start"] for b in fixed["bars"]]
    assert fixed["downbeats"] == sorted(set(fixed["downbeats"])) and set(fixed["downbeats"]) <= set(annotation["beats"])
    assert [b["beats"] for b in annotation["bars"]] == [3, 6, 6, 3, 3, 6, 3]  # the input is untouched


def test_an_occasional_short_bar_does_not_split_the_usual_bar() -> None:
    from score_fixes import regularize_bars

    annotation = _annotation([4] * 12 + [2] + [4] * 8)  # one 2/4 bar in a 4/4 piece
    assert regularize_bars(annotation) is annotation
    waltz = _annotation([3] * 20)
    assert regularize_bars(waltz) is waltz


def test_a_long_first_bar_gives_up_its_anacrusis() -> None:
    from score_fixes import regularize_bars

    fixed = regularize_bars(_annotation([4, 6, 6, 3, 3, 6, 3, 3]))
    assert [b["beats"] for b in fixed["bars"]][:4] == [1, 3, 3, 3]
    assert fixed["bars"][0]["meter"] == "1/4" and fixed["bars"][1]["start"] == 1.4


def test_bars_whose_beats_are_missing_are_left_alone() -> None:
    from score_fixes import regularize_bars

    annotation = _annotation([3, 3, 3, 6, 3, 3])
    annotation["beats"] = [b for b in annotation["beats"] if not 4.5 < b < 5.5]  # a gap inside the 6-beat bar
    assert [b["beats"] for b in regularize_bars(annotation)["bars"]] == [3, 3, 3, 6, 3, 3]


@dataclass
class TimedMeasure:
    time_signature: tuple[int, int]
    length: F


def test_a_short_first_bar_is_written_as_a_pickup_in_the_main_meter() -> None:
    from score_fixes import mark_anacrusis

    measures = [TimedMeasure((1, 4), F(1)), TimedMeasure((3, 4), F(3)), TimedMeasure((3, 4), F(3))]
    assert mark_anacrusis(measures)
    assert measures[0].time_signature == (3, 4) and measures[0].length == F(1)  # incomplete bar, same signature


def test_a_real_change_of_meter_at_the_start_is_kept() -> None:
    from score_fixes import mark_anacrusis

    longer_first = [TimedMeasure((4, 4), F(4)), TimedMeasure((3, 4), F(3))]
    other_unit = [TimedMeasure((3, 8), F(3, 2)), TimedMeasure((2, 4), F(2))]
    same = [TimedMeasure((3, 4), F(3)), TimedMeasure((3, 4), F(3))]
    for measures in (longer_first, other_unit, same):
        before = [m.time_signature for m in measures]
        assert not mark_anacrusis(measures)
        assert [m.time_signature for m in measures] == before


def test_warm_events_repeat_the_opening_notes_and_keep_every_other_event() -> None:
    import numpy as np
    from score_fixes import warm_events

    note, pedal = 0, 1
    events = np.array(
        [
            [5, pedal, 127, 0],  # pedal pressed before the first note ...
            [10, note, 60, 30],
            [40, note, 64, 100],  # rings past the end of the stretch
            [58, note, 67, 20],  # struck just ahead of the next downbeat
            [70, pedal, 0, 0],  # ... and released after the stretch
            [90, note, 72, 10],
        ]
    )
    warmed = warm_events(events, start=10, end=60, guard=5, note_kind=note)
    copy, rest = warmed[:2], warmed[2:]
    assert copy.tolist() == [[10, note, 60, 30], [40, note, 64, 20]]  # notes only, cut at the end, no early downbeat
    assert rest[:, 0].tolist() == [5, 60, 90, 108, 120, 140]  # everything from the start moves by 50
    assert (warmed[:, 1] == pedal).sum() == (events[:, 1] == pedal).sum() == 2  # no orphaned pedal event
    assert events[1, 0] == 10  # the input is untouched


def test_key_names_map_to_key_signatures() -> None:
    from score_fixes import signature_of

    assert [signature_of(k) for k in ("C:major", "G:major", "D:major", "F:major", "A#:major", "F#:major")] == [
        0,
        1,
        2,
        -1,
        -2,
        6,
    ]
    assert [signature_of(k) for k in ("A:minor", "E:minor", "D:minor", "C#:minor", "A:harmonic_minor")] == [
        0,
        1,
        -1,
        4,
        0,
    ]
    # A mode takes the signature of the major scale with the same notes.
    assert [signature_of(k) for k in ("D:dorian", "E:phrygian", "F:lydian", "G:mixolydian", "B:locrian")] == [0] * 5
    assert signature_of("E:phrygian_dominant") == 0  # the notes of A harmonic minor
    assert signature_of("Bb:major") == -2 and signature_of("") is None and signature_of("C:unknown") is None


def _keys(*segments: tuple[float, str]) -> list[dict]:
    return [{"start": start, "end": 0.0, "key": key} for start, key in segments]


def test_a_brief_excursion_keeps_the_signature_in_force() -> None:
    from score_fixes import bar_signatures

    bars = [float(i) for i in range(40)]
    # C for 12 bars, four bars each of G and D, C again, then F to the end.
    tracked = _keys((0, "C:major"), (12, "G:major"), (16, "D:major"), (20, "C:major"), (26, "F:major"))
    assert bar_signatures(tracked, bars) == [0] * 26 + [-1] * 14


def test_a_key_that_lasts_gets_its_signature_and_a_short_opening_joins_what_follows() -> None:
    from score_fixes import bar_signatures

    bars = [float(i) for i in range(30)]
    assert (
        bar_signatures(_keys((0, "F:major"), (10, "D:major"), (18, "C:major")), bars) == [-1] * 10 + [2] * 8 + [0] * 12
    )
    assert bar_signatures(_keys((0, "E:major"), (3, "A:minor")), bars) == [0] * 30
    assert bar_signatures(_keys((0, "A:minor"), (20, "C:major")), bars) == [0] * 30  # relative keys: one signature
    assert bar_signatures([], bars) == [None] * 30


def test_songs_follow_the_tracker_and_solo_piano_is_only_corrected() -> None:
    from score_fixes import correct_regions, follow_tracker

    model = [1] * 12 + [0] * 12
    assert follow_tracker(model, [0] * 24) == [0] * 24
    assert follow_tracker(model, [None] * 24) == model
    # Solo: the first region is heard as one signature throughout and corrected; the second
    # spans a change the score model chose not to print, and is left as the model wrote it.
    assert correct_regions(model, [0] * 12 + [0] * 4 + [1] * 8) == [0] * 24
    assert correct_regions([0] * 24, [0] * 8 + [1] * 12 + [0] * 4) == [0] * 24
    assert correct_regions([1] * 24, [0] * 24) == [0] * 24


def test_only_an_accompaniment_lead_in_is_settled() -> None:
    from score_fixes import lead_in_to_settle

    beats = [round(0.168 + 0.37 * i, 3) for i in range(20)]
    annotation = {"beats": beats, "downbeats": [0.538, 2.758], "melody": [{"start": 0.53, "end": 0.9, "pitch": 69}]}
    assert lead_in_to_settle([0.17, 0.51, 0.54, 0.91], annotation) == 0.538  # a bass note one beat early
    assert lead_in_to_settle([0.48, 0.51, 0.91], annotation) == 0.538  # a slightly early chord
    assert lead_in_to_settle([0.54, 0.91], annotation) is None  # nothing before the downbeat
    assert lead_in_to_settle([0.05, 0.54], annotation) is None  # more than a beat ahead: an introduction
    pickup = {**annotation, "melody": [{"start": 0.17, "end": 0.5, "pitch": 76}]}
    assert lead_in_to_settle([0.17, 0.54], pickup) is None  # the melody itself has an anacrusis
    no_melody = {**annotation, "melody": []}
    assert lead_in_to_settle([0.17, 0.54], no_melody) is None and lead_in_to_settle([0.45, 0.54], no_melody) == 0.538


def test_a_long_piece_is_cut_into_even_sections_at_bar_lines() -> None:
    import math

    from score_fixes import plan_sections

    bars = [float(i * 2) for i in range(203)]
    spans = plan_sections(bars, 64)
    assert len(spans) == 4 and spans[0][0] == -math.inf and spans[-1][1] == math.inf
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:], strict=False))  # no gap, no overlap
    sizes = [sum(1 for t in bars if lo <= t < hi) for lo, hi in spans]
    assert sizes == [51, 51, 51, 50] and all(lo in bars for lo, _ in spans[1:])
    assert plan_sections(bars[:64], 64) == [(-math.inf, math.inf)]  # short enough: one pass
    assert plan_sections(bars, 0) == [(-math.inf, math.inf)]  # disabled


def test_a_section_of_the_annotation_is_cut_out_and_moved_to_the_start() -> None:
    from score_fixes import cut_annotation

    cut = cut_annotation(ANNOTATION, DOWNBEATS[2], DOWNBEATS[5], shift=DOWNBEATS[2] - 1.0)
    assert [round(d, 4) for d in cut["downbeats"]] == [round(1.0 + i * 6 * BEAT, 4) for i in range(3)]
    assert len(cut["bars"]) == 3 and cut["bars"][0]["start"] == 1.0 and len(cut["beats"]) == 18
    assert cut["melody"] and min(n["start"] for n in cut["melody"]) >= 1.0 - 1e-6
    assert cut["duration"] == DOWNBEATS[5] - DOWNBEATS[2] + 1.0
    assert ANNOTATION["downbeats"] == DOWNBEATS  # the input is not modified


def test_joined_sections_do_not_restate_tempo_and_dynamic() -> None:
    from score_fixes import join_sections

    first = [
        Measure(directions=[(F(0), "metronome_quarter_144"), (F(0), "dyn_p")]),
        Measure(directions=[(F(1), "dyn_f")]),
    ]
    second = [Measure(directions=[(F(0), "metronome_quarter_144"), (F(0), "dyn_f"), (F(2), "dyn_p")]), Measure()]
    third = [Measure(directions=[(F(0), "metronome_quarter_120"), (F(0), "dyn_p")])]
    joined = join_sections([first, second, third])
    assert len(joined) == 5
    assert joined[2].directions == [(F(2), "dyn_p")]  # same tempo and dynamic as before the join: removed
    assert joined[4].directions == [(F(0), "metronome_quarter_120")]  # a new tempo is kept, the dynamic is not new
    assert joined[0].directions == [(F(0), "metronome_quarter_144"), (F(0), "dyn_p")]

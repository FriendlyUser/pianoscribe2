"""The beat the notes themselves keep (model_scripts/steady_beats.py), and what the writer makes of it."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model_scripts"))

import json  # noqa: E402
from fractions import Fraction  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from exact_score import (  # noqa: E402
    TICKS,
    Chord,
    Plan,
    gather_rolled,
    keep_counting,
    plan_score,
    plan_voices,
    quantize_onsets,
    settle_lengths,
    share_wide_chords,
)
from notate_exact import mark_played  # noqa: E402
from steady_beats import PRECISE, steady_beats  # noqa: E402

# One bar's rhythm in sixteenths, a melody over a bass note on every beat.
MELODY = (0, 2, 3, 4, 6, 8, 10, 11, 12, 14)


def piece(sections: list[tuple[float, int]], *, start: float = 2.0) -> tuple[list[float], list[float], list[float]]:
    """(onsets, beat times, bar-line times) of bars of 4/4 played to a click: ``sections`` are (bpm, bars)."""
    onsets, beats, lines = [], [], []
    now = start
    for bpm, bars in sections:
        beat = 60.0 / bpm
        for _ in range(bars):
            lines.append(now)
            beats += [now + beat * k for k in range(4)]
            onsets += [now + beat * k / 4.0 for k in MELODY] + [now + beat * k for k in range(4)]
            now += 4 * beat
    return onsets, beats, lines


def tracked(beats: list[float], lines: list[float], jitter: float = 0.03, seed: int = 1) -> dict:
    """What a beat tracker makes of it: every beat some hundredths of a second off."""
    rng = random.Random(seed)
    heard = [b + rng.uniform(-jitter, jitter) for b in beats]
    return {"beats": heard, "bars": [{"start": heard[beats.index(t)], "beats": 4, "unit": 4} for t in lines]}


def furthest(found: list[float], truth: list[float]) -> float:
    return max(min(abs(f - t) for f in found) for t in truth)


def counted(found: list, truth: list) -> bool:
    """As many as the music has; the line that closes the last bar may be there too."""
    return len(found) in (len(truth), len(truth) + 1)


def test_a_steady_beat_is_taken_from_the_notes_not_the_tracker() -> None:
    onsets, beats, lines = piece([(106.0, 12)])
    read = [t + random.Random(i).uniform(-0.002, 0.002) for i, t in enumerate(onsets)]  # read to 2 ms
    steady = steady_beats(tracked(beats, lines), read)
    assert len(steady.stretches) == 1 and abs(steady.stretches[0]["bpm"] - 106.0) < 0.02
    assert furthest(steady.beats, beats) < 0.002  # the tracker was 30 ms off; the notes are not
    assert [round(b["start"], 2) for b in steady.bars][: len(lines)] == [round(t, 2) for t in lines]
    assert counted(steady.bars, lines)
    assert steady.stretches[0]["scatter_ms"] < 1000 * PRECISE


def test_each_change_of_tempo_is_found_where_it_is() -> None:
    onsets, beats, lines = piece([(106.0, 8), (118.0, 8), (120.0, 8), (124.0, 8)])
    steady = steady_beats(tracked(beats, lines), onsets)
    assert [round(s["bpm"]) for s in steady.stretches] == [106, 118, 120, 124]
    assert [s["then"] for s in steady.stretches] == ["line", "line", "line", "end"]
    assert furthest(steady.beats, beats) < 0.002 and counted(steady.beats, beats)
    assert counted(steady.bars, lines) and furthest([b["start"] for b in steady.bars], lines) < 0.002


def test_a_beat_the_tracker_lost_is_put_back() -> None:
    onsets, beats, lines = piece([(120.0, 16)])
    heard = tracked(beats, lines)
    lost = 4 * 7 + 2  # the third beat of the eighth bar
    heard["beats"] = heard["beats"][:lost] + heard["beats"][lost + 1 :]
    steady = steady_beats(heard, onsets)
    assert counted(steady.beats, beats) and furthest(steady.beats, beats) < 0.002
    assert steady.regular_bars in (16, 17)  # every bar four beats, counted


def test_a_slowing_bar_between_two_tempos_keeps_its_beats() -> None:
    # Eight bars at 106, one bar at 100 (a ritardando written as a tempo), then 118: the bar between is
    # too short to find by itself, but its beats run from the last line of the one tempo to the first of the next.
    onsets, beats, lines = piece([(106.0, 8), (100.0, 1), (118.0, 8)])
    steady = steady_beats(tracked(beats, lines), onsets)
    assert [s["then"] for s in steady.stretches] == ["tempo", "end"]
    assert counted(steady.beats, beats) and furthest(steady.beats, beats) < 0.003
    assert counted(steady.bars, lines)


def test_a_ritardando_puts_its_beats_on_the_notes() -> None:
    # 124, then one bar whose four beats slow (0.6 s, 0.6 s, 0.75 s, a pause), then 65: no steady beat in that bar.
    onsets, beats, lines = piece([(124.0, 8)])
    at = lines[-1] + 4 * 60.0 / 124.0
    slowing = [at, at + 0.6, at + 1.2, at + 1.95]
    after = at + 4.55
    tail_onsets, tail_beats, tail_lines = piece([(65.0, 4)], start=after)
    eighths = [at + 0.3, at + 0.9]  # notes between the beats of the slowing bar
    steady = steady_beats(
        tracked([*beats, *slowing, *tail_beats], [*lines, at, *tail_lines], jitter=0.05),
        [*onsets, *slowing, *eighths, *tail_onsets],
    )
    assert furthest(steady.beats, [*beats, *slowing, *tail_beats]) < 0.003
    every = [*lines, at, *tail_lines]
    assert counted(steady.bars, every) and furthest([b["start"] for b in steady.bars], every) < 0.003


def test_free_playing_keeps_the_tracked_beats() -> None:
    rng = random.Random(3)
    beats = [1.0]
    for _ in range(40):
        beats.append(beats[-1] + rng.uniform(0.45, 0.75))  # rubato: no two beats alike
    onsets = [b + rng.uniform(-0.01, 0.01) for b in beats] + [b + rng.uniform(0.1, 0.4) for b in beats[::3]]
    annotation = {"beats": beats, "bars": [{"start": beats[4 * i], "beats": 4, "unit": 4} for i in range(10)]}
    steady = steady_beats(annotation, onsets)
    assert steady.stretches == [] and steady.beats == beats and steady.bars == annotation["bars"]


def test_a_coarse_reading_follows_the_notes_and_invents_no_triplets() -> None:
    # A screen recording at 30 frames a second: each note a frame early or late, the notes of a chord a
    # frame apart, and the whole reading wandering by some hundredths of a second from bar to bar.
    onsets, beats, lines = piece([(100.0, 24)])
    rng = random.Random(5)
    wander, read = 0.0, []
    for t in sorted(onsets):
        wander = max(-0.05, min(0.05, wander + rng.uniform(-0.006, 0.006)))
        read.append(t + wander + rng.choice((-0.017, 0.0, 0.017)))
    notes = [{"start": t, "end": t + 0.12, "pitch": 60 + i % 5, "hand": "right"} for i, t in enumerate(read)]
    plan = plan_score(notes, tracked(beats, lines))
    assert plan.steady is not None and all(s["scatter_ms"] > 1000 * PRECISE for s in plan.steady.stretches)
    assert not any(w.triplet for w in plan.written)  # all on sixteenths, as played
    assert len(plan.bars) == 24 and all(bar.ticks == 48 for bar in plan.bars)
    # Recorded steadily (each note a frame early or late, no more), the lines drawn through the notes stand.
    steadily = [t + rng.choice((-0.017, 0.0, 0.017)) for t in sorted(onsets)]
    notes = [{"start": t, "end": t + 0.12, "pitch": 60 + i % 5, "hand": "right"} for i, t in enumerate(steadily)]
    plan = plan_score(notes, tracked(beats, lines))
    assert plan.steady is not None and plan.steady.followed is None and len(plan.steady.stretches) >= 1
    assert sum(plan.steady.fitted) >= 90 and not any(w.triplet for w in plan.written)


def test_read_coarsely_one_late_note_is_not_a_triplet() -> None:
    loose = {0: 0.06, 1: 0.06}
    assert quantize_onsets([0.0, 0.31, 0.5], loose) == [0, 3, 6]  # 0.31 is nearer a third; it is a late sixteenth
    assert quantize_onsets([1.0, 1.34, 1.66], loose) == [12, 16, 20]  # two notes on thirds: a triplet
    assert quantize_onsets([0.0, 0.31, 0.5]) == [0, 4, 6]  # read finely, 0.31 is where it says


def test_a_drawn_length_is_the_note_value_and_a_played_one_is_not() -> None:
    def written(start: int, end: int, following: int, fine: bool) -> int:
        voice = [Chord(start, end, [60]), Chord(following, following + 6, [62])]
        settle_lengths(voice, set(), set(range(8)) if fine else None)
        return voice[0].end - voice[0].start

    assert written(0, 6, 9, fine=True) == 6  # drawn as an eighth, a sixteenth rest after it: written so
    assert written(0, 6, 9, fine=False) == 9  # played so, the rest is a breath: the note runs on
    assert written(0, 5, 6, fine=True) == 6  # the sliver cut off a note struck again
    assert written(0, 43, 48, fine=True) == 48  # nine tenths of a whole note is a whole note
    assert written(9, 13, 18, fine=False) == 3  # played short on the last sixteenth: a sixteenth, not a tied eighth
    assert written(0, 9, 18, fine=False) == 12  # played for three quarters of a beat: a quarter


def test_a_single_line_stays_one_voice() -> None:
    # A melody played legato (each note held a little into the next), then two real voices, then one again.
    melody = [Chord(12 * i, 12 * i + 15, [60 + i]) for i in range(4)]
    held = Chord(48, 96, [48])  # held under...
    moving = [Chord(48 + 12 * i, 60 + 12 * i, [72 + i]) for i in range(4)]  # ...four moving notes
    after = [Chord(96 + 12 * i, 108 + 12 * i, [65 + i]) for i in range(2)]
    voices = plan_voices([*melody, held, *moving, *after], played=True)
    assert [c.voice for c in melody] == [1, 1, 1, 1] and [c.end for c in melody[:3]] == [12, 24, 36]
    assert held.voice == 2 and [c.voice for c in moving] == [1, 1, 1, 1]
    assert [c.voice for c in after] == [1, 1] and len(voices) == 2


def test_a_chord_too_wide_for_one_hand_is_shared_between_the_staves() -> None:
    wide = [(0, 12, p) for p in (43, 50, 55, 58, 62, 67)]  # G2 to G4 in the "left hand" colour
    close = [(24, 36, p) for p in (50, 55, 58)]
    melody = [(24, 36, 74)]
    ticked = {1: list(melody), 2: [*wide, *close]}
    assert share_wide_chords(ticked) == 3
    assert sorted(p for _, _, p in ticked[2]) == [43, 50, 50, 55, 55, 58]
    assert sorted(p for _, _, p in ticked[1]) == [58, 62, 67, 74]
    # With a melody note far above, the upper part would not fit the right hand either: left as drawn.
    ticked = {1: [(0, 12, 86)], 2: list(wide)}
    assert share_wide_chords(ticked) == 0 and len(ticked[2]) == 6


def test_notes_that_run_on_keep_one_beat_where_the_tracker_changed_its_mind() -> None:
    # Twenty bars of even eighths. For the first twelve the tracker takes the off-beat for the
    # beat (the figure's lowest notes fall there), then rights itself: half a beat would go
    # missing from the score where it does. The notes never break step, so the beat does not
    # either, and of the tracker's two answers the one that starts the piece on a beat stands.
    beat, start = 60.0 / 115.0, 3.0
    rng = random.Random(4)
    onsets = [start + 0.5 * beat * k + rng.uniform(-0.009, 0.009) for k in range(160)]
    beats = [start + beat * k for k in range(80)]
    heard = [b + (0.5 * beat if k < 48 else 0.0) + rng.uniform(-0.02, 0.02) for k, b in enumerate(beats)]
    bars = [{"start": heard[k], "beats": 4, "unit": 4} for k in range(0, 80, 4)]
    steady = steady_beats({"beats": heard, "bars": bars}, onsets)
    assert steady.rephased >= 1
    assert furthest(steady.beats, beats) < 0.02  # every beat of the music, none half a beat off
    assert len([b for b in steady.beats if start - 0.1 < b < beats[-1] + 0.1]) == 80
    lines = [b["start"] for b in steady.bars]
    assert furthest(lines, beats[::4]) < 0.02 and all(b["beats"] == 4 for b in steady.bars)
    # Where the tracker is of one mind there is nothing to settle: a piece that begins before the beat stays so.
    pickup = [t + 0.5 * beat for t in onsets]
    agreed = [b + 0.5 * beat + rng.uniform(-0.02, 0.02) for b in beats]
    after = steady_beats(
        {"beats": agreed, "bars": [{"start": agreed[k], "beats": 4, "unit": 4} for k in range(0, 80, 4)]}, pickup
    )
    assert after.rephased == 0 and furthest(after.beats, [b + 0.5 * beat for b in beats]) < 0.02


def test_a_few_chords_far_apart_keep_no_tempo() -> None:
    # After the last steady bar, a ritardando and five slow chords, each struck by a hand: its
    # notes a hundredth of a second apart. Counted note by note they would sit on many a grid.
    onsets, beats, lines = piece([(120.0, 8)])
    end = lines[-1] + 2.0
    coda = [end + 0.31, end + 0.9, end + 1.8, end + 4.7, end + 7.6, end + 8.7, end + 11.6]
    spread = [t + d for t in coda for d in (0.0, 0.009, 0.017)]
    heard = tracked(beats, lines)
    heard["beats"] += [end + 0.5 * k for k in range(26)]
    heard["bars"] += [{"start": end + 2.0 * k, "beats": 4, "unit": 4} for k in range(7)]
    steady = steady_beats(heard, onsets + spread)
    assert len(steady.stretches) == 1 and steady.stretches[0]["end"] < end
    assert not any(fitted for b, fitted in zip(steady.beats, steady.fitted, strict=True) if b > end + 0.6)


def test_with_the_pedal_down_a_note_lasts_to_the_next_or_to_a_beat() -> None:
    # Keys touched for a sixteenth and let go, the pedal holding the sound: a beat apart they are
    # quarters, not eighths and rests; the last, before bars of nothing, lasts its beat.
    def touched() -> list[Chord]:
        return [Chord(12 * k, 12 * k + 3, [60 + k]) for k in range(4)] + [Chord(51, 53, [70]), Chord(120, 123, [72])]

    plain = touched()
    settle_lengths(plain, set(), None)
    assert [c.end - c.start for c in plain[:3]] == [6, 6, 6]  # as a player's, without the pedal: eighths and rests
    rung = touched()
    count = settle_lengths(rung, set(), None, let_ring=True)
    assert [c.end for c in rung] == [12, 24, 36, 51, 60, 132] and count == 6
    # Read finely or not: under the pedal a key held a little into the next note is legato, not a second voice.
    line = [Chord(0, 9, [60]), Chord(6, 15, [64]), Chord(12, 24, [67])]
    assert len(plan_voices([Chord(c.start, c.end, c.pitches) for c in line])) == 2
    assert len(plan_voices(line, played=True)) == 1


def test_a_score_of_pedalled_notes_has_no_slivers_of_rest() -> None:
    beat = 0.5
    notes = [
        {"start": 2.0 + beat * 0.5 * k, "end": 2.0 + beat * 0.5 * k + 0.07, "pitch": 60 + k % 5, "hand": "right"}
        for k in range(32)
    ]
    notes += [
        {"start": 2.0 + beat * 2 * k, "end": 2.0 + beat * 2 * k + 0.09, "pitch": 36, "hand": "left"} for k in range(8)
    ]
    annotation = tracked([2.0 + beat * k for k in range(18)], [2.0 + 2.0 * k for k in range(5)], jitter=0.0)
    plain = plan_score(notes, annotation)
    rung = plan_score(notes, annotation, let_ring=True)
    assert {w.kind for w in plain.written if w.staff == 1} == {"16th"} and plain.rung == 0
    assert {w.kind for w in rung.written if w.staff == 1} == {"eighth"}
    assert [w.kind for w in rung.written if w.staff == 2] == ["half"] * 7 + ["quarter"] and rung.rung >= 8


# ----------------------------------------------------------------------------------------------
# played by hand, to no click
# ----------------------------------------------------------------------------------------------
# What the hands do in a bar: (hand, pitches, beats from the bar line it is struck on).
VERSE = [("left", (34, 46), 0.0), ("left", (53, 58), 1.0), ("left", (53, 58), 3.0)] + [
    ("right", (70, 82), 0.5 * k) for k in (0, 2, 4, 5, 6, 7)
]
SPARSE = [("left", (31, 43), 0.0), ("left", (55, 58), 1.0), ("left", (55, 58), 3.0)]
PICKUP = [("left", (27, 39), 0.0), ("right", (68,), 2.0), ("right", (68,), 2.5), ("right", (68,), 3.5)]
HELD = [("left", (58, 62, 65), 0.0)] + [("right", (72 + k % 3,), 0.5 * k) for k in range(1, 8)]
Bars = list[tuple[float | list[float], list[tuple[str, tuple[int, ...], float]]]]


def drifting(rng: random.Random, bpm: float, bars: int) -> list[float]:
    """Seconds per beat of ``bars`` bars about ``bpm``, as a player keeps time alone: every bar a little
    off the last, and the whole drifting by a few per cent over a page."""
    drift, out = 0.0, []
    for _ in range(bars):
        drift = max(-0.03, min(0.03, drift + rng.gauss(0.0, 0.006)))
        out.append(60.0 / (bpm * (1.0 + drift + rng.gauss(0.0, 0.008))))
    return out


def played(bars: Bars, *, seed: int = 7, start: float = 3.0) -> tuple[list[dict], list[float], list[float], dict]:
    """(notes, beats, bar lines, what the tracker heard) of ``bars``: (seconds per beat, or of each of its four
    beats; what is played in it). Every note is struck a few milliseconds off its place, as a hand
    does, and the tracker hears each beat within two hundredths of a second of where it is."""
    rng = random.Random(seed)
    notes: list[dict] = []
    beats: list[float] = []
    lines: list[float] = []
    now = start
    for lengths, pattern in bars:
        lengths = [lengths] * 4 if isinstance(lengths, float) else lengths
        edges = [now]
        for length in lengths:
            edges.append(edges[-1] + length)
        lines.append(now)
        beats += edges[:4]
        for hand, pitches, at in pattern:
            k = min(int(at), 3)
            time = edges[k] + (at - k) * lengths[k] + rng.gauss(0.0, 0.008)
            notes += [
                {"start": time + rng.uniform(-0.004, 0.004), "end": time + 0.1, "pitch": pitch, "hand": hand}
                for pitch in pitches
            ]
        now = edges[4]
    heard = [b + rng.uniform(-0.02, 0.02) for b in beats]
    annotation = {"beats": heard, "bars": [{"start": heard[4 * k], "beats": 4, "unit": 4} for k in range(len(lines))]}
    return notes, beats, lines, annotation


def written_as_played(plan: Plan, notes: list[dict], bars: Bars, lines: list[float]) -> None:
    """Every bar is four beats and begins where its bar line is; every note is written on the beat it was played on."""
    assert plan.steady is not None and plan.steady.followed is not None
    json.dumps(plan.steady.followed)  # goes into the job's facts
    assert len(plan.bars) == len(bars) and all(bar.ticks == 4 * TICKS for bar in plan.bars)
    assert max(abs(bar.seconds - line) for bar, line in zip(plan.bars, lines, strict=True)) < 0.05
    number_of = {tick: n for n, bar in enumerate(plan.bars) for tick in range(bar.start, bar.start + bar.ticks)}
    found: dict[tuple[int, str], list[tuple[int, int]]] = {}
    for i, note in enumerate(notes):
        number = number_of[plan.starts[i]]
        found.setdefault((number, note["hand"]), []).append((plan.starts[i] - plan.bars[number].start, note["pitch"]))
    wanted: dict[tuple[int, str], list[tuple[int, int]]] = {}
    for number, (_, pattern) in enumerate(bars):
        for hand, pitches, at in pattern:
            wanted.setdefault((number, hand), []).extend((round(at * TICKS), pitch) for pitch in pitches)
    wrong = {
        key: (sorted(found.get(key, [])), sorted(wanted[key]))
        for key in wanted
        if sorted(found.get(key, [])) != sorted(wanted[key])
    }
    assert not wrong, wrong


def verse_broadened_verse(seed: int) -> Bars:
    rng = random.Random(seed)
    return [
        *[(length, VERSE) for length in drifting(rng, 116.0, 12)],
        (0.58, PICKUP),
        *[(length, VERSE) for length in drifting(rng, 117.0, 12)],
    ]


def test_a_bar_the_player_broadens_is_written_on_the_beat_heard() -> None:
    # Twelve bars about 116 to the minute, then a bar a ninth slower in which only a bass note and
    # three notes of the melody are struck (on the third beat, after it, and before the next
    # bar), then on at 117: as before a new verse. Lines drawn evenly through the verse and on
    # through that bar put its notes between everything, and all that follows half a beat out.
    for seed in (11, 12, 13):
        bars = verse_broadened_verse(seed)
        notes, _, lines, heard = played(bars, seed=seed)
        written_as_played(plan_score(notes, heard, let_ring=True), notes, bars, lines)


def test_a_hurried_beat_does_not_move_the_bar_line() -> None:
    # Chords on the first, second and fourth beat, nothing on the third: twice the player comes in
    # early after the silent beat, by a twentieth of a second and more. The bar lines stay with the bass.
    for seed in (21, 22, 23):
        rng = random.Random(seed)
        bars: Bars = [
            (length, VERSE if k < 8 or k >= 18 else SPARSE) for k, length in enumerate(drifting(rng, 116.0, 26))
        ]
        for k in (11, 14):
            length = bars[k][0]
            assert isinstance(length, float)
            bars[k] = ([length, length, 0.9 * length, 0.9 * length], SPARSE)
        notes, _, lines, heard = played(bars, seed=seed)
        written_as_played(plan_score(notes, heard, let_ring=True), notes, bars, lines)


def test_eighths_over_a_held_chord_keep_their_beat_through_a_change_of_pace() -> None:
    # A melody in even eighths over one held chord to the bar, slower than the verse before it and
    # quickening into the next: no two bars of it at one tempo.
    for seed in (31, 32, 33):
        rng = random.Random(seed)
        bars: Bars = [(length, VERSE) for length in drifting(rng, 114.0, 10)]
        bars += [(60.0 / bpm, HELD) for bpm in (112.0, 111.0, 115.0, 111.0, 111.0, 119.0)]
        bars += [([0.49] * 4, [("left", (53, 57, 60), 0.0), ("right", (62, 74), 3.0)])]
        bars += [(length, VERSE) for length in drifting(rng, 117.0, 10)]
        notes, _, lines, heard = played(bars, seed=seed)
        written_as_played(plan_score(notes, heard, let_ring=True), notes, bars, lines)


def test_played_music_keeps_one_beat_where_the_tracker_changed_its_mind() -> None:
    # Through the first eight bars the tracker has the off-beat for the beat, then rights itself with
    # one beat half as long again. The notes run on: the eight bars are the fewer, and are put right.
    for seed in (41, 42, 43):
        bars = verse_broadened_verse(seed)
        notes, beats, lines, _ = played(bars, seed=seed)
        rng = random.Random(seed)
        heard = [(a + b) / 2.0 + rng.uniform(-0.02, 0.02) for a, b in zip(beats[:32], beats[1:33], strict=True)]
        heard += [b + rng.uniform(-0.02, 0.02) for b in beats[33:]]
        starts = [heard[4 * k] for k in range(8)] + [heard[4 * k - 1] for k in range(9, len(lines))]
        annotation = {"beats": heard, "bars": [{"start": t, "beats": 4, "unit": 4} for t in starts]}
        plan = plan_score(notes, annotation, let_ring=True)
        written_as_played(plan, notes, bars, lines)
        assert plan.steady is not None and plan.steady.followed["off_the_off_beat"] >= 30


def test_a_beat_the_tracker_lost_in_played_music_is_put_back() -> None:
    # The tracker misses the second beat of the broadened bar, where nothing is struck; its bar lines
    # stay where they were, so that bar counts a beat too few.
    for seed in (51, 52, 53):
        bars = verse_broadened_verse(seed)
        notes, _, lines, heard = played(bars, seed=seed)
        lost = 4 * 12 + 1
        annotation = {"beats": heard["beats"][:lost] + heard["beats"][lost + 1 :], "bars": heard["bars"]}
        plan = plan_score(notes, annotation, let_ring=True)
        written_as_played(plan, notes, bars, lines)
        assert plan.steady is not None and plan.steady.followed["put_in"] >= 1


def test_beats_are_put_back_where_the_tracker_counts_every_other_one() -> None:
    # Fitted beats half a second apart to 10.0; then the tracker's: a ritardando (no beat of it twice
    # the one before), and from 11.8 on, with notes still to come, beats twice as long.
    beats = [0.5 * k for k in range(21)] + [10.55, 11.15, 11.8, 13.1, 14.3, 15.5, 16.7, 17.9]
    exact = [True] * 21 + [False] * 8
    bars = [{"start": 2.0 * k, "beats": 4, "unit": 4} for k in range(6)] + [
        {"start": t, "beats": 4, "unit": 4} for t in (13.1, 17.9)
    ]
    out, added, lines = keep_counting(beats, exact, bars, last_onset=15.5)
    assert out[:24] == beats[:24] and len(added) == 5  # the ritardando stays as it was heard
    assert [round(t, 2) for t in out[24:]] == [12.45, 13.1, 13.7, 14.3, 14.9, 15.5, 16.1, 16.7, 17.3, 17.9]
    assert [round(float(bar["start"]), 2) for bar in lines][-3:] == [13.1, 15.5, 17.9]  # a bar of eight beats is two
    # Beats twice as long only after the last note is struck measure nothing but how long it is held.
    assert keep_counting(beats, exact, bars, last_onset=11.0) == (beats, [], bars)


def test_a_rolled_octave_is_one_chord() -> None:
    beats = [0.5 * k for k in range(9)]

    def note(start: float, pitch: int, hand: str = "right") -> dict:
        return {"start": start, "end": start + 0.2, "pitch": pitch, "hand": hand}

    notes = [
        note(0.93, 60), note(1.0, 72),  # an octave rolled upwards on to the beat
        note(1.5, 64), note(1.56, 67),  # a third struck unevenly: not an octave, left as played
        note(2.0, 60), note(2.125, 72),  # a sixteenth apart: two notes
        note(2.5, 48, "left"), note(2.56, 72),  # an octave between the hands is not rolled by one of them
        note(3.0, 60), note(3.06, 72), note(3.12, 60), note(3.18, 72),  # a tremolo
    ]  # fmt: skip
    struck = [n["start"] for n in notes]
    assert gather_rolled(struck, notes, beats) == 1
    assert struck[:2] == [1.0, 1.0] and struck[2:] == [n["start"] for n in notes[2:]]


def test_played_music_has_one_tempo_mark_and_its_ritardandos() -> None:
    def marks(tempos: list[float]) -> list[tuple[int, str]]:
        measures = [SimpleNamespace(directions=[(Fraction(0), "metronome_quarter_99")], tempos=[]) for _ in tempos]
        assert mark_played(measures, tempos) == [(0, 115)]
        assert all(measure.tempos == [(Fraction(0), tempo)] for measure, tempo in zip(measures, tempos, strict=True))
        return [(k, name) for k, measure in enumerate(measures) for _, name in measure.directions]

    breathing = [115.0 + 2.5 * ((7 * k) % 5 - 2) / 2.0 for k in range(40)]  # every bar within 2.5 of 115
    assert marks(breathing) == [(0, "metronome_quarter_115")]
    broadened = list(breathing)
    broadened[20] = 103.0
    assert marks(broadened) == [(0, "metronome_quarter_115"), (20, "tempo_rit"), (21, "tempo_a_tempo")]
    closing = list(breathing)
    closing[37:] = [104.0, 96.0, 115.0]
    assert marks(closing) == [(0, "metronome_quarter_115"), (37, "tempo_rit")]  # nothing returns to tempo at the close

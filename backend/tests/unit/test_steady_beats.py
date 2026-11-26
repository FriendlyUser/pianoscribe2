"""The beat the notes themselves keep (model_scripts/steady_beats.py), and what the writer makes of it."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model_scripts"))

from exact_score import Chord, plan_score, plan_voices, quantize_onsets, settle_lengths, share_wide_chords  # noqa: E402
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

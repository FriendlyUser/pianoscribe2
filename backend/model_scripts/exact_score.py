"""Write notes whose timing is known exactly as notation, by rule.

A piano-tutorial video gives every note's key, the moment it is struck, how long it is
held and which hand plays it, to a few milliseconds (``app.visual``). Notation is then no
longer a matter of judgement about what was played, so it is not left to the score model,
which writes fluent notation but may respell a rhythm or add a note. This module places
each note on the beat grid and writes it down as it is.

What is taken from the audio models, as on the other paths: where the beats are, how they
group into bars, and the key. What is done here, in order:

1. **Beats on the notes** (``snap_beats``). A tracked beat is some milliseconds off; the
   notes are not. Each beat that has a note on it moves onto that note.
2. **Positions** (``quantize_onsets``). Within each beat the notes of a hand lie on the
   simplest grid that fits them: halves, quarters, thirds or sixths of the beat.
3. **Voices** (``plan_voices``). Notes of one hand struck together are a chord; a note
   held while others move is a second voice.
4. **Note values** (``settle_lengths``). A bar in the video is a little shorter than the
   note it stands for (keys are released before the next note); a note runs to the next
   note of its voice unless a real rest follows.
5. **Ties** (``split_value``). A note is cut at bar lines, and inside a bar wherever it
   would hide a beat stronger than the one it starts on, and the parts are tied.

Positions are counted in ticks, twelve to the beat, so that quarters (3 ticks) and thirds
(4 ticks) of a beat are both whole numbers. Nothing here imports the score model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

TICKS = 12  # per beat
# Grids a hand's notes may lie on within one beat, simplest first (in ticks).
GRIDS: tuple[tuple[int, ...], ...] = ((0, 6), (0, 3, 6, 9), (0, 4, 8), (0, 2, 4, 6, 8, 10))
WRITABLE_TICKS = (0, 2, 3, 4, 6, 8, 9, 10)  # the union of the grids
FIT = 0.045  # beats a note may lie from a grid point and still be on it
MAX_VOICES = 2  # per hand
NOTE_TYPES = {
    "whole": Fraction(4),
    "half": Fraction(2),
    "quarter": Fraction(1),
    "eighth": Fraction(1, 2),
    "16th": Fraction(1, 4),
    "32nd": Fraction(1, 8),
    "64th": Fraction(1, 16),
}


# ----------------------------------------------------------------------------------------------
# beats
# ----------------------------------------------------------------------------------------------
def snap_beats(beats: list[float], onsets: list[float], reach: float = 0.17) -> list[float]:
    """Move each beat onto the note struck on it, if there is one within ``reach`` of a beat.

    Beats with no note on them keep their place between their neighbours.
    """
    if len(beats) < 2 or not onsets:
        return list(beats)
    notes = sorted(onsets)
    moved = list(beats)
    fixed = [False] * len(beats)
    cursor = 0
    for i, beat in enumerate(beats):
        before = beat - beats[i - 1] if i > 0 else beats[1] - beats[0]
        after = beats[i + 1] - beat if i + 1 < len(beats) else beats[-1] - beats[-2]
        limit = reach * min(before, after)
        while cursor < len(notes) and notes[cursor] < beat - limit:
            cursor += 1
        best = None
        k = cursor
        while k < len(notes) and notes[k] <= beat + limit:
            if best is None or abs(notes[k] - beat) < abs(best - beat):
                best = notes[k]
            k += 1
        if best is not None:
            moved[i], fixed[i] = best, True
    anchors = [i for i, ok in enumerate(fixed) if ok]
    for left, right in zip(anchors, anchors[1:], strict=False):
        span = beats[right] - beats[left]
        for i in range(left + 1, right):
            moved[i] = moved[left] + (beats[i] - beats[left]) / span * (moved[right] - moved[left])
    if anchors:
        for i in range(anchors[0]):
            moved[i] = beats[i] + (moved[anchors[0]] - beats[anchors[0]])
        for i in range(anchors[-1] + 1, len(beats)):
            moved[i] = beats[i] + (moved[anchors[-1]] - beats[anchors[-1]])
    # Keep the beats in order whatever was moved.
    for i in range(1, len(moved)):
        if moved[i] <= moved[i - 1]:
            return list(beats)
    return moved


def beat_position(time: float, beats: list[float]) -> float:
    """``time`` in beats (0 at the first beat), continuing at the edge tempo beyond either end."""
    if time <= beats[0]:
        return (time - beats[0]) / (beats[1] - beats[0])
    if time >= beats[-1]:
        return len(beats) - 1 + (time - beats[-1]) / (beats[-1] - beats[-2])
    low, high = 0, len(beats) - 1
    while high - low > 1:
        middle = (low + high) // 2
        if beats[middle] <= time:
            low = middle
        else:
            high = middle
    return low + (time - beats[low]) / (beats[low + 1] - beats[low])


def quantize_onsets(positions: list[float]) -> list[int]:
    """Tick of each onset (positions in beats), on the simplest grid that fits its beat."""
    by_beat: dict[int, list[int]] = {}
    for index, position in enumerate(positions):
        by_beat.setdefault(math.floor(position + FIT), []).append(index)
    ticks = [0] * len(positions)
    for beat, members in by_beat.items():
        inside = [positions[i] - beat for i in members]
        chosen: tuple[int, ...] = WRITABLE_TICKS
        for grid in GRIDS:
            points = (*grid, TICKS)
            if all(min(abs(value * TICKS - p) for p in points) <= FIT * TICKS for value in inside):
                chosen = grid
                break
        points = (*chosen, TICKS)
        for i, value in zip(members, inside, strict=True):
            ticks[i] = beat * TICKS + min(points, key=lambda p: abs(value * TICKS - p))
    return ticks


# ----------------------------------------------------------------------------------------------
# bars
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Bar:
    start: int  # tick of the bar line
    ticks: int  # length
    beats: int  # of the time signature
    unit: int  # the note value of one beat: 4 or 8
    seconds: float  # when the bar begins in the recording

    @property
    def nominal(self) -> int:
        return self.beats * TICKS

    @property
    def quarter(self) -> Fraction:
        """Quarter notes per tick."""
        return Fraction(4, self.unit * TICKS)


def layout_bars(annotation: dict, beats: list[float], first_tick: int, last_tick: int) -> list[Bar]:
    """Bars covering the ticks ``first_tick`` up to ``last_tick``, from the tracker's bar lines.

    A bar line is the beat nearest the tracker's bar start, and a bar is as long as the
    distance to the next bar line. Notes before the first bar line make a pickup bar;
    after the last one, bars of the same kind are added.
    """
    lines: list[tuple[int, int, int]] = []  # (beat index, beats, unit)
    for bar in annotation.get("bars") or []:
        index = round(beat_position(float(bar["start"]), beats))
        if not lines or index > lines[-1][0]:
            lines.append((index, int(bar.get("beats") or 4), int(bar.get("unit") or 4)))
    if not lines:
        lines = [(math.floor(first_tick / TICKS), 4, 4)]

    def seconds(beat_index: int) -> float:
        if beat_index <= 0:
            return beats[0] + beat_index * (beats[1] - beats[0])
        if beat_index >= len(beats) - 1:
            return beats[-1] + (beat_index - (len(beats) - 1)) * (beats[-1] - beats[-2])
        return beats[beat_index]

    def bar(index: int, length: int, unit: int, nominal: int | None = None) -> Bar:
        return Bar(index * TICKS, length * TICKS, nominal or length, unit, seconds(index))

    bars = [
        bar(index, (following[0] - index) if following else count, unit)
        for (index, count, unit), following in zip(lines, [*lines[1:], None], strict=True)
    ]
    count, unit = lines[-1][1], lines[-1][2]
    while bars[-1].start + bars[-1].ticks < last_tick:
        bars.append(bar((bars[-1].start + bars[-1].ticks) // TICKS, count, unit))
    bars = [b for b in bars if b.start + b.ticks > first_tick and b.start < last_tick]  # only bars with music in them
    if not bars:
        raise ValueError("no bars cover the notes")
    first = bars[0]
    if first.start > first_tick:
        # Notes before the first bar line: whole bars of the same kind, led by a pickup as long as it needs.
        early = math.ceil((first.start - first_tick) / TICKS)
        whole, pickup = divmod(early, first.beats)
        start = first.start // TICKS - early
        lead = [bar(start, pickup, first.unit, nominal=first.beats)] if pickup else []
        lead += [bar(start + pickup + k * first.beats, first.beats, first.unit) for k in range(whole)]
        bars = lead + bars
    return bars


# ----------------------------------------------------------------------------------------------
# voices and note values
# ----------------------------------------------------------------------------------------------
@dataclass
class Chord:
    start: int  # tick
    end: int  # tick; first as held, then as written
    pitches: list[int]
    voice: int = 1
    held: int = 0  # ticks the keys were really down

    def __post_init__(self) -> None:
        self.held = self.end - self.start


def group_chords(notes: list[tuple[int, int, int]]) -> list[Chord]:
    """Notes (start tick, end tick, pitch) of one hand as chords: struck together and held alike."""
    by_start: dict[int, list[tuple[int, int]]] = {}
    for start, end, pitch in notes:
        by_start.setdefault(start, []).append((max(end, start + 1), pitch))
    chords: list[Chord] = []
    for start in sorted(by_start):
        members = sorted(by_start[start])
        current: list[tuple[int, int]] = []
        for end, pitch in members:
            shortest = current[0][0] - start if current else 0
            if current and end - current[0][0] > max(3, 0.34 * shortest):
                chords.append(Chord(start, current[-1][0], sorted(p for _, p in current)))
                current = []
            current.append((end, pitch))
        chords.append(Chord(start, current[-1][0], sorted(p for _, p in current)))
    return chords


def plan_voices(chords: list[Chord]) -> list[list[Chord]]:
    """Share the chords of one hand among at most ``MAX_VOICES`` voices, each a line with no overlaps.

    A chord continues the voice that ended most recently. A chord that begins while every
    voice still sounds takes the key-release of the nearest one as a slur into it: that
    chord is cut short where the new one starts.
    """
    voices: list[list[Chord]] = []
    for chord in sorted(chords, key=lambda c: (c.start, -(c.end - c.start), c.pitches[0])):
        free = [v for v in voices if v[-1].end <= chord.start]
        slurred = [
            v
            for v in voices
            if 0 < v[-1].end - chord.start <= max(2, 0.12 * (v[-1].end - v[-1].start)) and v[-1].start < chord.start
        ]
        if free:
            voice = max(free, key=lambda v: v[-1].end)
        elif slurred:
            voice = min(slurred, key=lambda v: v[-1].end)
        elif len(voices) < MAX_VOICES:
            voices.append([])
            voice = voices[-1]
        else:
            started_before = [v for v in voices if v[-1].start < chord.start]
            if not started_before:
                # nowhere to put it without losing a note: sound it with the chord struck at the same time
                target = min(voices, key=lambda v: abs(v[-1].end - chord.end))
                target[-1].pitches = sorted(set(target[-1].pitches) | set(chord.pitches))
                continue
            voice = min(started_before, key=lambda v: v[-1].end)
        if voice and voice[-1].end > chord.start:
            voice[-1].end = chord.start
        voice.append(chord)
    voices.sort(key=lambda v: -sum(sum(c.pitches) / len(c.pitches) for c in v) / len(v))
    for number, voice in enumerate(voices, start=1):
        for chord in voice:
            chord.voice = number
    return voices


def _may_end(tick: int, triple_beats: set[int]) -> bool:
    """Whether a note can end at ``tick``: on the beat, or on the grid its beat is counted in."""
    if tick % TICKS == 0:
        return True
    return tick % 2 == 0 if tick // TICKS in triple_beats else tick % 3 == 0


def settle_lengths(voice: list[Chord], triple_beats: set[int]) -> None:
    """Give each chord of a voice its written length.

    A key is released a little before the next note, so a bar in the video is shorter than
    the note value it stands for. A chord therefore lasts until the next chord of its
    voice when the gap between them is short; otherwise it gets the simplest note value
    it could be a shortened form of, and a rest follows.
    """
    for chord, following in zip(voice, [*voice[1:], None], strict=True):
        limit = following.start if following is not None else None
        held = max(1, chord.end - chord.start)
        if limit is not None and (limit - chord.end <= 3 or held >= 0.6 * (limit - chord.start)):
            chord.end = limit
            continue
        best, best_cost = None, 0.0
        for length in range(max(1, held - 1), 2 * held + 13):
            end = chord.start + length
            if limit is not None and end > limit:
                break
            if not _may_end(end, triple_beats) or not 0.6 <= held / length <= 1.1:
                continue
            plain = (
                0.0
                if length % TICKS == 0
                else 0.02
                if length % 6 == 0
                else 0.06
                if length % 3 == 0 or length % 4 == 0
                else 0.15
            )
            cost = abs(held / length - 0.95) + plain
            if best is None or cost < best_cost:
                best, best_cost = end, cost
        if best is None:
            best = chord.end
            while not _may_end(best, triple_beats) and (limit is None or best < limit):
                best += 1
        chord.end = max(best, chord.start + 1)


# ----------------------------------------------------------------------------------------------
# note values and ties
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Value:
    start: int  # tick
    ticks: int
    kind: str  # "quarter", "eighth", ...
    dots: int = 0
    triplet: bool = False


def _plain(ticks: int, unit: int) -> tuple[str, int] | None:
    """(note type, dots) that lasts ``ticks``, or None if no single plain value does."""
    quarters = Fraction(ticks * 4, unit * TICKS)
    for kind, length in NOTE_TYPES.items():
        if quarters == length:
            return kind, 0
        if quarters == length * Fraction(3, 2):
            return kind, 1
    return None


def _strength(offset: int, bar: Bar) -> int:
    """How strong the place ``offset`` ticks into ``bar`` is: 5 for the bar line, down to 0."""
    if offset == 0:
        return 5
    compound = bar.unit >= 8 and bar.beats % 3 == 0 and bar.beats > 3
    if compound and offset % (3 * TICKS) == 0:
        return 4
    if not compound and bar.beats % 2 == 0 and bar.beats >= 4 and offset % (bar.beats // 2 * TICKS) == 0:
        return 4
    if offset % TICKS == 0:
        return 3
    if offset % 6 == 0:
        return 2
    return 1 if offset % 3 == 0 else 0


def _split_plain(start: int, end: int, bar: Bar) -> list[Value]:
    """[start, end) inside ``bar``, on quarters of the beat, as plain note values to be tied."""
    length = end - start
    inside = [(_strength(t - bar.start, bar), t) for t in range(start + 3, end, 3)]
    strongest = max((s for s, _ in inside), default=-1)
    plain = _plain(length, bar.unit)
    if plain is not None and _strength(start - bar.start, bar) >= strongest:
        return [Value(start, length, plain[0], plain[1])]
    if not inside:
        return [Value(start, length, "16th" if bar.unit == 4 else "32nd")]
    # Cut at the strongest place inside; of several equally strong, the one that leaves the longest plain start.
    places = [t for s, t in inside if s == strongest]
    cut = places[0] if _strength(start - bar.start, bar) < strongest else places[-1]
    if _strength(start - bar.start, bar) >= strongest:
        for place in reversed(places):
            if _plain(place - start, bar.unit) is not None:
                cut = place
                break
    return _split_plain(start, cut, bar) + _split_plain(cut, end, bar)


def _split_triplet(start: int, end: int, bar: Bar) -> list[Value]:
    """[start, end) inside one beat, on thirds or sixths of it, as triplet values to be tied."""
    names = {8: "quarter", 4: "eighth", 2: "16th"} if bar.unit == 4 else {8: "eighth", 4: "16th", 2: "32nd"}
    out: list[Value] = []
    position = start
    while position < end:
        for size in (8, 4, 2):
            if (position % TICKS) % min(size, 4) == 0 and position + size <= end:
                out.append(Value(position, size, names[size], 0, True))
                position += size
                break
        else:
            # an odd tick: nothing writable ends here; give the remainder to a sixth of the beat
            out.append(Value(position, end - position, names[2], 0, True))
            break
    return out


def split_value(start: int, end: int, bars: list[Bar]) -> list[Value]:
    """Note values, to be tied in order, that together last from ``start`` to ``end`` (ticks)."""
    values: list[Value] = []
    for bar in bars:
        a, b = max(start, bar.start), min(end, bar.start + bar.ticks)
        if a >= b:
            continue
        cuts = [a, *[t for t in range(bar.start, b, TICKS) if a < t < b], b]
        pending: int | None = None
        for left, right in zip(cuts, cuts[1:], strict=False):
            whole = left % TICKS == 0 and right % TICKS == 0
            if whole or (left % 3 == 0 and right % 3 == 0):
                pending = left if pending is None else pending
                continue
            if pending is not None:
                values += _split_plain(pending, left, bar)
                pending = None
            values += _split_triplet(left, right, bar)
        if pending is not None:
            values += _split_plain(pending, b, bar)
    return values


# ----------------------------------------------------------------------------------------------
# the whole plan
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Written:
    """One note value on the page."""

    bar: int  # index into the bars
    onset: Fraction  # quarter notes from the bar line
    staff: int  # 1: right hand, upper staff; 2: left hand
    voice: int
    kind: str
    dots: int
    triplet: bool
    pitches: tuple[int, ...]
    tied: bool  # tied to the next value of the same note


@dataclass
class Plan:
    bars: list[Bar]
    written: list[Written]
    beats: list[float]  # the beats the notes were placed on
    notes: int = 0
    moved_beats: int = 0
    off_grid: int = 0  # notes further than FIT from every grid point
    cut_short: int = 0  # chords cut where another began, for want of a third voice
    voices: dict[int, int] = field(default_factory=dict)  # staff -> voices used


def plan_score(notes: list[dict], annotation: dict) -> Plan:
    """Lay ``notes`` (``start``, ``end`` in seconds, ``pitch``, ``hand``) out on the tracker's beats and bars."""
    tracked = [float(b) for b in annotation["beats"]]
    beats = snap_beats(tracked, [float(n["start"]) for n in notes])
    moved = sum(abs(a - b) > 1e-6 for a, b in zip(tracked, beats, strict=True))
    staves = {1: [n for n in notes if n["hand"] != "left"], 2: [n for n in notes if n["hand"] == "left"]}
    ticked: dict[int, list[tuple[int, int, int]]] = {}
    off_grid = 0
    triple: dict[int, set[int]] = {}
    for staff, members in staves.items():
        positions = [beat_position(float(n["start"]), beats) for n in members]
        starts = quantize_onsets(positions)
        off_grid += sum(abs(p * TICKS - t) > FIT * TICKS for p, t in zip(positions, starts, strict=True))
        triple[staff] = {t // TICKS for t in starts if t % 3 != 0}
        ends = [
            max(s + 1, round(beat_position(float(n["end"]), beats) * TICKS))
            for n, s in zip(members, starts, strict=True)
        ]
        ticked[staff] = [(s, e, int(n["pitch"])) for s, e, n in zip(starts, ends, members, strict=True)]
    every = [item for members in ticked.values() for item in members]
    if not every:
        raise ValueError("no notes to write")
    plan = Plan(bars=[], written=[], beats=beats, notes=len(every), moved_beats=moved, off_grid=off_grid)
    chords_by_staff: dict[int, list[list[Chord]]] = {}
    last = max(e for _, e, _ in every)
    for staff, members in ticked.items():
        chords = group_chords(members)
        held = {id(c): c.end for c in chords}
        voices = plan_voices(chords)
        plan.cut_short += sum(c.end < held[id(c)] - 2 for c in chords)
        for voice in voices:
            settle_lengths(voice, triple[staff])
        chords_by_staff[staff] = voices
        plan.voices[staff] = len(voices)
        last = max([last, *[c.end for v in voices for c in v]])
    plan.bars = layout_bars(annotation, beats, min(s for s, _, _ in every), last)
    index_of = {}
    for number, bar in enumerate(plan.bars):
        for tick in range(bar.start, bar.start + bar.ticks):
            index_of[tick] = number
    for staff, voices in chords_by_staff.items():
        for voice in voices:
            for chord in voice:
                values = split_value(chord.start, chord.end, plan.bars)
                for position, value in enumerate(values):
                    bar = plan.bars[index_of[value.start]]
                    plan.written.append(
                        Written(
                            bar=index_of[value.start],
                            onset=(value.start - bar.start) * bar.quarter,
                            staff=staff,
                            voice=chord.voice,
                            kind=value.kind,
                            dots=value.dots,
                            triplet=value.triplet,
                            pitches=tuple(chord.pitches),
                            tied=position < len(values) - 1,
                        )
                    )
    plan.written.sort(key=lambda w: (w.bar, w.onset, w.staff, w.voice))
    return plan

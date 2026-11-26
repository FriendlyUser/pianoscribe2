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

from steady_beats import PRECISE, Steady, steady_beats

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
def snap_beats(
    beats: list[float], onsets: list[float], reach: float = 0.17, exact: list[bool] | None = None
) -> list[float]:
    """Move each beat onto the note struck on it, if there is one within ``reach`` of a beat.

    Beats with no note on them keep their place between their neighbours. Beats marked
    ``exact`` (fitted to the notes already, see ``steady_beats``) stay where they are.
    """
    if len(beats) < 2 or not onsets:
        return list(beats)
    notes = sorted(onsets)
    moved = list(beats)
    fixed = list(exact) if exact is not None else [False] * len(beats)
    cursor = 0
    for i, beat in enumerate(beats):
        if fixed[i]:
            continue
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


STRAIGHT = (0, 3, 6, 9)  # quarters of the beat, in ticks


def quantize_onsets(positions: list[float], loose: dict[int, float] | None = None) -> list[int]:
    """Tick of each onset (positions in beats), on the simplest grid that fits its beat.

    ``loose`` gives, for beats read coarsely, how far (in beats) a note may lie from a half or a
    quarter of the beat and still be on it: more than ``FIT``, since a note read a frame late is
    nearer a third of the beat than the quarter it is on. Thirds and sixths must always fit closely.
    """
    by_beat: dict[int, list[int]] = {}
    for index, position in enumerate(positions):
        by_beat.setdefault(math.floor(position + FIT), []).append(index)
    ticks = [0] * len(positions)
    for beat, members in by_beat.items():
        inside = [positions[i] - beat for i in members]
        reach = max(FIT, (loose or {}).get(beat, FIT))

        def fits(grid: tuple[int, ...], inside: list[float] = inside, reach: float = reach) -> bool:
            allowed = (reach if all(p % 3 == 0 for p in grid) else FIT) * TICKS
            return all(min(abs(value * TICKS - p) for p in (*grid, TICKS)) <= allowed for value in inside)

        coarse = loose is not None and beat in loose
        chosen = next((grid for grid in GRIDS if fits(grid)), None)
        if coarse and chosen is not None and len(chosen) > 2:
            # Sixteenths that fit only loosely are placed one by one, below (the stronger place wins a near tie).
            tight = all(min(abs(value * TICKS - p) for p in (*chosen, TICKS)) <= FIT * TICKS for value in inside)
            chosen = chosen if tight else None
        if coarse and chosen is not None and any(p % 3 for p in chosen):
            # Read coarsely, one note late by a frame sits on a third of the beat. A triplet is at
            # least two notes on thirds or sixths of the beat that are not also quarters of it.
            thirds = {min(chosen, key=lambda p, value=value: abs(value * TICKS - p)) for value in inside}
            if len({p for p in thirds if p % 3}) < 2:
                chosen = None
        for i, value in zip(members, inside, strict=True):
            if chosen is None:
                # No one grid holds every note of the beat. Each goes to the nearest quarter of the
                # beat, unless (read finely) it sits squarely on a third or a sixth and on no quarter.
                near = min(WRITABLE_TICKS, key=lambda p, value=value: abs(value * TICKS - p))
                straight = min((*STRAIGHT, TICKS), key=lambda p, value=value: abs(value * TICKS - p))
                exact = not coarse and abs(value * TICKS - near) <= FIT * TICKS < abs(value * TICKS - straight)
                if coarse:
                    # Read coarsely, a note halfway between two quarters of the beat is on the stronger.
                    close = [
                        p for p in (*STRAIGHT, TICKS) if abs(value * TICKS - p) <= abs(value * TICKS - straight) + 0.5
                    ]
                    straight = min(close, key=lambda p: (p % TICKS != 0, p % 6 != 0))
                ticks[i] = beat * TICKS + (near if exact else straight)
            else:
                ticks[i] = beat * TICKS + min((*chosen, TICKS), key=lambda p, value=value: abs(value * TICKS - p))
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
    cap: int | None = None  # tick by which it must end: where the next passage of its hand begins

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


def plan_voices(chords: list[Chord], played: bool = False) -> list[list[Chord]]:
    """Share the chords of one hand among at most ``MAX_VOICES`` voices, each a line with no overlaps.

    A chord continues the voice that ended nearest its start, a moment before it or (the
    key-release of a slur) a moment after: that chord is then cut short where the new one
    starts. A second voice is opened only for a chord that begins while another is really
    held, and only for as long as something is: once every voice has ended, the passage is
    over and the next chord starts again in one voice. So a single line stays one voice.

    ``played``: the lengths are a player's (see ``settle_lengths``), who holds a note into the
    next far longer than a score drawn by machine does.
    """

    def slur(chord: Chord) -> int:
        length = chord.end - chord.start
        return min(8, max(4, round(0.3 * length))) if played else max(2, round(0.12 * length))

    passages: list[list[list[Chord]]] = []
    voices: list[list[Chord]] = []
    for chord in sorted(chords, key=lambda c: (c.start, -(c.end - c.start), c.pitches[0])):
        if voices and all(v[-1].end - chord.start <= slur(v[-1]) and v[-1].start < chord.start for v in voices):
            for voice in voices:
                voice[-1].end = min(voice[-1].end, chord.start)
                for member in voice:
                    member.cap = chord.start
            passages.append(voices)
            voices = []
        if not voices:
            voices.append([chord])
            continue
        near = [v for v in voices if v[-1].end - chord.start <= slur(v[-1]) and v[-1].start < chord.start]
        if near:
            voice = min(near, key=lambda v: (abs(v[-1].end - chord.start), v[-1].end > chord.start))
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
    passages.append(voices)
    lines: list[list[Chord]] = [[] for _ in range(MAX_VOICES)]
    for passage in passages:
        passage.sort(key=lambda v: -sum(sum(c.pitches) / len(c.pitches) for c in v) / len(v))
        for number, voice in enumerate(passage, start=1):
            for chord in voice:
                chord.voice = number
            lines[number - 1] += voice
    return [line for line in lines if line]


def _may_end(tick: int, triple_beats: set[int]) -> bool:
    """Whether a note can end at ``tick``: on the beat, or on the grid its beat is counted in."""
    if tick % TICKS == 0:
        return True
    return tick % 2 == 0 if tick // TICKS in triple_beats else tick % 3 == 0


RING = 2 * TICKS  # ticks to the next note of its voice within which a note the pedal holds is written out to it


def settle_lengths(
    voice: list[Chord], triple_beats: set[int], fine: set[int] | None = None, *, let_ring: bool = False
) -> int:
    """Give each chord of a voice its written length; returns how many were let ring (see ``let_ring``).

    ``fine``: the beats read finely (a video drawn from a score, read to a few milliseconds).
    There a bar's length is the note value itself, but for the sliver cut off a note that is
    struck again (or off every note, a tenth of it or so): a chord runs to the next of its
    voice when the gap is that sliver, and otherwise ends where its bar ends, on the grid,
    with the rest that follows written out.

    Elsewhere (a recording of a performance, read coarsely) lengths are a player's: a key is
    released some way before the next note. A chord runs to the next of its voice when the gap
    is no more than a sixteenth or it fills most of the space; otherwise it gets the plainest
    note value it could be a shortened form of, and a rest follows.

    ``let_ring``: the notes of this performance go on sounding after their keys are released
    (the pedal holds them), so how long a key was down is not how long its note lasts, however
    finely it is read. A chord then runs to the next of its voice when that comes within two
    beats (or as a player's would above), and otherwise ends on a beat: the one nearest the
    release of its keys, the first after it began at the least. Written out, that is, not as
    a short note and a rest, and not tied over by a sliver.
    """
    rung = 0
    for chord, following in zip(voice, [*voice[1:], None], strict=True):
        limits = [t for t in (following.start if following is not None else None, chord.cap) if t is not None]
        limit = min(limits) if limits else None
        held = max(1, chord.end - chord.start)
        # Lengths are exact where the video is read finely, unless the pedal holds the notes: then how
        # long a key was down is a player's doing here too, however exactly it is read.
        exact = fine is not None and chord.start // TICKS in fine and not let_ring
        ring = let_ring
        if limit is not None:
            gap = limit - chord.end
            space = limit - chord.start
            if (gap <= 2 or held >= 0.88 * space) if exact else (gap <= 4 or held >= 0.55 * space):
                chord.end = limit
                continue
            if ring and space <= RING:
                chord.end = limit
                rung += 1
                continue
        if ring:
            beat = (chord.start // TICKS + 1) * TICKS  # the first beat after it began
            end = max(beat, round(chord.end / TICKS) * TICKS)
            end = end if limit is None else min(end, limit)
            rung += end > chord.end + 2
            chord.end = end
            continue
        whole = round(held / TICKS) * TICKS
        if not limits and whole >= TICKS and abs(held - whole) <= 3:
            # The last note of its hand, let ring a little past the beat or released a little before it.
            chord.end = chord.start + whole
            continue
        best, best_cost = None, 0.0
        span = range(max(1, held - 2), math.ceil(held / 0.88) + 3) if exact else range(max(1, held - 2), 2 * held + 13)
        for length in span:
            end = chord.start + length
            if limit is not None and end > limit:
                break
            if not _may_end(end, triple_beats):
                continue
            if exact:
                # As long as its bar, or the value its bar is the gated form of (nine tenths of it or so).
                # Where a bar ends is read less surely than where it begins (a tick or so): of the
                # places near its end, the note ends on the plainest, a beat before a half of one,
                # a half before a quarter, any of them before a third.
                as_drawn = 0.88 * length <= held <= length
                place = 0.0 if end % TICKS == 0 else 0.15 if end % 6 == 0 else 0.3 if end % 3 == 0 else 0.45
                cost = (0.0 if as_drawn else 0.25 * abs(length - held)) + place
            else:
                if not 0.5 <= held / length <= 1.4:
                    continue
                plain = (
                    0.0
                    if length % TICKS == 0
                    else 0.05
                    if length % 6 == 0
                    else 0.35
                    if length % 3 == 0 or length % 4 == 0
                    else 0.5
                )
                # A value that would have to be tied over a beat it does not start on is not plain.
                tied = chord.start % 6 != 0 and (end - 1) // TICKS != chord.start // TICKS
                cost = abs(held / length - 0.9) + plain + (0.55 if tied else 0.0)
            if best is None or cost < best_cost - 1e-9 or (not exact and abs(cost - best_cost) <= 1e-9):
                best, best_cost = end, cost  # of equals, the longer: fewer rests
        if best is None:
            best = chord.end
            while not _may_end(best, triple_beats) and (limit is None or best < limit):
                best += 1
        chord.end = max(best, chord.start + 1)
    return rung


HAND_SPAN = 16  # semitones one hand can strike together at the very most (a major tenth)


def share_wide_chords(ticked: dict[int, list[tuple[int, int, int]]]) -> int:
    """Move to the other staff the part of a chord no one hand can strike; the number of notes moved.

    A video colours notes by track, and a track is not always a hand: an accompaniment may be
    drawn as one "hand" two octaves wide. Notes (start, end, pitch) struck together on one
    staff and further apart than a hand reaches are divided where the staves meet (at the gap
    nearest middle C that leaves each part within a hand's reach), and the part on the far
    side goes to the other staff, if what is struck there at that moment still fits one hand
    with it.
    """
    moved = 0
    struck: dict[int, dict[int, list[tuple[int, int, int]]]] = {1: {}, 2: {}}
    for staff in (1, 2):
        for item in ticked.get(staff, []):
            struck[staff].setdefault(item[0], []).append(item)
    for staff, other in ((2, 1), (1, 2)):
        for start, items in sorted(struck[staff].items()):
            pitches = sorted(item[2] for item in items)
            if pitches[-1] - pitches[0] <= HAND_SPAN:
                continue
            cuts = [
                (b - a, b)
                for a, b in zip(pitches, pitches[1:], strict=False)
                if a - pitches[0] <= HAND_SPAN and pitches[-1] - b <= HAND_SPAN and b > a
            ]
            if not cuts:
                continue
            cut = min(cuts, key=lambda item: (abs(item[1] - 59.5), -item[0]))[1]  # the gap nearest middle C
            goes = [item for item in items if (item[2] >= cut) == (staff == 2)]
            there = [item[2] for item in struck[other].get(start, [])] + [item[2] for item in goes]
            if max(there) - min(there) > HAND_SPAN:
                continue
            struck[staff][start] = [item for item in items if item not in goes]
            struck[other].setdefault(start, []).extend(goes)
            moved += len(goes)
    for staff in (1, 2):
        ticked[staff] = [item for start in sorted(struck[staff]) for item in struck[staff][start]]
    return moved


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
    # Syncopations written whole by custom, though they hide a stronger place than they start on:
    # inside one beat, the note between two sixteenths (16th, 8th, 16th) and the one after a
    # sixteenth (16th, dotted 8th); across one beat that is not the middle of the bar, the note
    # between two eighths (8th, quarter, 8th).
    offset = (start - bar.start) % TICKS
    whole = (offset == 3 and length in (6, 9)) or (offset == 6 and length == TICKS and strongest <= 3)
    if plain is not None and bar.unit == 4 and whole:
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
    steady: Steady | None = None  # the beat the notes keep, where they keep one
    shared: int = 0  # notes written on the other hand's staff, their chord being too wide for one hand
    rung: int = 0  # chords written out beyond the release of their keys, the pedal holding them
    voices: dict[int, int] = field(default_factory=dict)  # staff -> voices used


def plan_score(notes: list[dict], annotation: dict, *, let_ring: bool = False) -> Plan:
    """Lay ``notes`` (``start``, ``end`` in seconds, ``pitch``, ``hand``) out on the tracker's beats and bars.

    ``let_ring``: the notes ring on after their keys are released (see ``settle_lengths``).
    """
    onsets = [float(n["start"]) for n in notes]
    steady = steady_beats(annotation, onsets)
    annotation = {**annotation, "beats": steady.beats, "bars": steady.bars}
    beats = snap_beats(steady.beats, onsets, exact=steady.fitted)
    moved = sum(abs(a - b) > 1e-6 for a, b in zip(steady.beats, beats, strict=True))
    # Notes struck together are written together: each takes the time of its moment.
    struck = list(onsets)
    group: list[int] = []
    for index in [*sorted(range(len(notes)), key=lambda i: onsets[i]), -1]:
        if index >= 0 and group and onsets[index] - onsets[group[0]] <= steady.within:
            group.append(index)
            continue
        for member in group:
            struck[member] = sum(onsets[i] for i in group) / len(group)
        group = [index]
    # Read finely or coarsely is a matter of the whole video (its frame rate, its picture), not of
    # one stretch whose few notes happen to sit well.
    fitted = sorted(scatter for scatter in steady.scatter if scatter < float("inf"))
    finely = bool(fitted) and fitted[len(fitted) // 2] <= PRECISE
    fine = {beat for beat, scatter in enumerate(steady.scatter) if finely and scatter <= PRECISE}
    loose = {
        beat: min(0.075, 2.5 * max(scatter, PRECISE) / (beats[beat + 1] - beats[beat]))
        for beat, scatter in enumerate(steady.scatter[:-1])
        if scatter < float("inf") and beat not in fine
    }
    hands = {
        1: [i for i, n in enumerate(notes) if n["hand"] != "left"],
        2: [i for i, n in enumerate(notes) if n["hand"] == "left"],
    }
    ticked: dict[int, list[tuple[int, int, int]]] = {}
    off_grid = 0
    for staff, members in hands.items():
        positions = [beat_position(struck[i], beats) for i in members]
        starts = quantize_onsets(positions, loose)
        off_grid += sum(
            abs(p * TICKS - t) > max(FIT, loose.get(t // TICKS, FIT)) * TICKS
            for p, t in zip(positions, starts, strict=True)
        )
        ends = [
            max(s + 1, round(beat_position(float(notes[i]["end"]), beats) * TICKS))
            for i, s in zip(members, starts, strict=True)
        ]
        ticked[staff] = [(s, e, int(notes[i]["pitch"])) for s, e, i in zip(starts, ends, members, strict=True)]
    shared = share_wide_chords(ticked)
    triple = {staff: {s // TICKS for s, _, _ in members if s % 3 != 0} for staff, members in ticked.items()}
    every = [item for members in ticked.values() for item in members]
    if not every:
        raise ValueError("no notes to write")
    plan = Plan(bars=[], written=[], beats=beats, notes=len(every), moved_beats=moved, off_grid=off_grid)
    plan.steady = steady
    plan.shared = shared
    chords_by_staff: dict[int, list[list[Chord]]] = {}
    last = min(s for s, _, _ in every) + 1  # the last tick written, once the lengths are settled
    for staff, members in ticked.items():
        chords = group_chords(members)
        held = {id(c): c.end for c in chords}
        # Under the pedal a key held into the next note is a player's legato, however finely it is read.
        voices = plan_voices(chords, played=not finely or let_ring)
        plan.cut_short += sum(c.end < held[id(c)] - 2 for c in chords)
        for voice in voices:
            plan.rung += settle_lengths(voice, triple[staff], fine, let_ring=let_ring)
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

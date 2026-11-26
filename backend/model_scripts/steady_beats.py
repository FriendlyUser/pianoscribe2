"""The beat the notes themselves keep.

A tutorial video is drawn from a score played by a machine: its notes sit on a beat that is
steady to a few milliseconds, for stretches as long as the tempo holds. The beat *heard* in
the recording (the beat tracker's) is right about where the beats are, to a few hundredths of
a second, and about which of them begin a bar; it is not steady, and it rounds what it hears.
Written on it, the same figure comes out in different rhythms from bar to bar.

So the tracked beats are used only to say roughly where the beat is. Here the notes are asked:

1. **Stretches** (``find_stretches``). In a window of a few bars, the tempo and phase are
   searched for that put the onsets on quarters of the beat most exactly. Windows that agree
   are one stretch of steady tempo, whose beat is then fitted to every onset in it by least
   squares, and which reaches as far to either side as the onsets keep to it.
2. **Joins** (``steady_beats``). Where the tempo changes, one stretch's beat line falls on the
   next one's; that line is where the change is. Where the notes keep no steady beat (a free
   introduction, a ritardando), the tracked beats stay.

Numpy only; nothing here knows about notation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STEPS = 4  # the grid the tempo is fitted on: quarters of the beat
FINE = 12  # the grid a fitted beat is checked on: quarters and thirds of the beat
WINDOW_BEATS = 12.0
HOP_BEATS = 2.0
LEAST_ONSETS = 6
TEMPO_RANGE = 0.08  # how far a stretch's tempo may be from the tracked tempo around it
SHARP = 0.12  # scatter (RMS, as a share of a quarter of the beat) under which a window keeps a steady beat
SAME_TEMPO = 0.004  # relative difference under which two windows are in one tempo
SAME_PHASE = 0.2  # of a quarter of the beat
LEAST_STRETCH_BEATS = 6.0
# How plain each twelfth of the beat is as a place for a note: the beat, its half, its quarters and thirds, the rest.
_PLAIN = (0, 3, 3, 2, 2, 3, 1, 3, 2, 2, 3, 3)
_WRITABLE = (0, 2, 3, 4, 6, 8, 9, 10)  # twelfths of the beat a note can be written on
COARSE_MOMENT = 0.04  # seconds within which the notes of a coarsely read video are struck together


@dataclass
class Stretch:
    """A stretch of steady tempo: beat lines at ``origin + k * period``."""

    origin: float  # seconds of one beat line
    period: float  # seconds per beat
    start: float  # first and last onset that keep to it (seconds)
    end: float
    scatter: float  # RMS distance of its onsets from the grid, seconds
    onsets: int

    def line(self, k: int) -> float:
        return self.origin + k * self.period

    def index(self, time: float) -> float:
        return (time - self.origin) / self.period


def moments(onsets: list[float], within: float = 0.012) -> np.ndarray:
    """The moments at which notes are struck: onsets no further than ``within`` from the first of
    their group (the notes of one chord, read a few milliseconds apart) are one moment, at their mean."""
    return np.array([float(np.mean(group)) for group in group_moments(onsets, within)])


def group_moments(onsets: list[float], within: float) -> list[list[float]]:
    """``onsets`` in order, grouped into moments (see ``moments``)."""
    groups: list[list[float]] = []
    for time in sorted(float(t) for t in onsets):
        if groups and time - groups[-1][0] <= within:
            groups[-1].append(time)
        else:
            groups.append([time])
    return groups


def _sharpest(onsets: np.ndarray, periods: np.ndarray) -> tuple[float, float, float]:
    """(period, a grid line, scatter as a share of the step) of the grid of ``STEPS`` to the beat that fits best."""
    steps = periods / STEPS
    angles = 2.0 * np.pi * onsets[None, :] / steps[:, None]
    mean = np.exp(1j * angles).mean(axis=1)
    best = int(np.argmax(np.abs(mean)))
    step = float(steps[best])
    line = float(np.angle(mean[best]) * step / (2.0 * np.pi))
    off = (onsets - line + step / 2.0) % step - step / 2.0
    return float(periods[best]), line, float(np.sqrt((off**2).mean()) / step)


def _fit(onsets: np.ndarray, origin: float, period: float) -> tuple[float, float, float, np.ndarray]:
    """Least-squares beat (origin, period, scatter, which onsets are on it) through the onsets on quarters of it."""
    step = period / STEPS
    inside = np.ones(len(onsets), dtype=bool)
    scatter = 0.0
    for _ in range(4):
        index = np.round((onsets - origin) / step)
        off = onsets - (origin + index * step)
        inside = np.abs(off) <= 0.25 * step
        if inside.sum() < 4 or np.ptp(index[inside]) == 0:
            break
        slope, intercept = np.polyfit(index[inside], onsets[inside], 1)
        origin, step = float(intercept), float(slope)
        off = onsets[inside] - (origin + np.round((onsets[inside] - origin) / step) * step)
        scatter = float(np.sqrt((off**2).mean()))
    return origin, step * STEPS, scatter, inside


def _on_grid(onsets: np.ndarray, origin: float, period: float, scatter: float) -> np.ndarray:
    """Which onsets lie on the beat's grid: on the quarters, thirds and sixths of the beat when the fit is
    fine enough to tell them apart, else on its quarters."""
    reach = max(3.0 * scatter, 0.006)
    if reach <= period / FINE / 4.0:
        tick = (onsets - origin) / (period / FINE)
        nearest = np.round(tick)
        on: np.ndarray = (np.abs(tick - nearest) * (period / FINE) <= reach) & np.isin(nearest % FINE, _WRITABLE)
        return on
    step = period / STEPS
    off = (onsets - origin + step / 2.0) % step - step / 2.0
    return np.abs(off) <= min(reach, 0.3 * step)


def _extent(on: np.ndarray, first: int, last: int, low: int, high: int) -> tuple[int, int]:
    """Grow [first, last] (indices into the onsets), within [low, high], for as long as the onsets beyond
    keep to the grid: until two in a row are off it. A single onset off the grid (an ornament, a
    note read a frame late) does not end a stretch; another tempo does at once."""
    edge, misses = last, 0
    for k in range(last + 1, high + 1):
        misses = 0 if on[k] else misses + 1
        if misses >= 2:
            break
        if on[k]:
            edge = k
    last = edge
    edge, misses = first, 0
    for k in range(first - 1, low - 1, -1):
        misses = 0 if on[k] else misses + 1
        if misses >= 2:
            break
        if on[k]:
            edge = k
    first = edge
    # An onset on the grid just past one off it may be there by chance (one in four is): a stretch
    # ends on two onsets in a row that keep to it.
    while last - 1 > first and not (on[last] and on[last - 1]):
        last -= 1
    while first + 1 < last and not (on[first] and on[first + 1]):
        first += 1
    return first, last


def _windows(tracked: np.ndarray, times: np.ndarray) -> list[tuple[float, float, float, float]]:
    """(centre, period, a grid line, scatter as a share of the step) of the grid that fits the onsets best in each
    window of ``WINDOW_BEATS``, ``HOP_BEATS`` apart; the share is infinite where a window holds too few onsets."""
    usual = float(np.median(np.diff(tracked)))
    found: list[tuple[float, float, float, float]] = []
    centre = float(times[0])
    while centre <= times[-1]:
        local = np.diff(tracked[(tracked >= centre - 8 * usual) & (tracked <= centre + 8 * usual)])
        about = float(np.median(local)) if len(local) >= 4 else usual
        inside = times[np.abs(times - centre) <= WINDOW_BEATS / 2.0 * about]
        if len(inside) >= LEAST_ONSETS and np.ptp(inside) >= 4 * about:
            period, line, share = _sharpest(inside, about * np.linspace(1.0 - TEMPO_RANGE, 1.0 + TEMPO_RANGE, 801))
            found.append((centre, period, line, share))
        else:
            found.append((centre, about, centre, float("inf")))
        centre += HOP_BEATS * about
    return found


WANDERS = 0.003  # change of tempo between windows a bar and a half apart above which no click keeps the player's time
APART = 6  # windows: three bars' worth of hops, so that two windows compared share no more than half their notes


def wander(tracked: np.ndarray, times: np.ndarray) -> float | None:
    """How far the tempo the notes keep changes from one window of three bars to one a bar and a half on: the
    median change, as a share of the tempo. None where too few windows keep any tempo to say.

    A video drawn from a score, or played to a click, keeps its tempo to a fifth of a per cent
    from window to window (which is what reading it coarsely leaves uncertain), but for the few
    places where the tempo is changed. A player keeping time alone is a per cent off from one
    window to the next, all the way through.
    """
    if len(tracked) < 4 or len(times) < LEAST_ONSETS:
        return None
    periods = np.array([period if share <= SHARP else np.nan for _, period, _, share in _windows(tracked, times)])
    if len(periods) <= APART:
        return None
    change = np.abs(np.log(periods[APART:] / periods[:-APART]))
    change = change[np.isfinite(change)]
    return float(np.median(change)) if len(change) >= 8 else None


def find_stretches(beats: list[float], times: np.ndarray) -> list[Stretch]:
    """The stretches of steady tempo in the moments ``times`` (see ``moments``), in order, none overlapping another."""
    tracked = np.asarray(beats, dtype=np.float64)
    if len(tracked) < 4 or len(times) < LEAST_ONSETS:
        return []
    # scatter share, centre, period, a grid line
    seeds = [
        (share, centre, period, line) for centre, period, line, share in _windows(tracked, times) if share <= SHARP
    ]
    # The sharpest window first: each grows into the stretch around it, as far as the onsets keep its beat.
    found: list[tuple[int, int, Stretch]] = []
    for _, centre, period, line in sorted(seeds):
        if any(times[a] <= centre <= times[b] for a, b, _ in found):
            continue
        low = max([b + 1 for a, b, _ in found if times[b] < centre], default=0)
        high = min([a - 1 for a, b, _ in found if times[a] > centre], default=len(times) - 1)
        members = np.flatnonzero(np.abs(times - centre) <= WINDOW_BEATS / 2.0 * period)
        members = members[(members >= low) & (members <= high)]
        if len(members) < LEAST_ONSETS:
            continue
        first, last = int(members[0]), int(members[-1])
        origin, scatter = line, 0.0
        on = np.zeros(len(times), dtype=bool)
        for _ in range(3):  # fit, see how far it reaches, fit again over all of that: the tempo sharpens each time
            origin, period, scatter, _ = _fit(times[first : last + 1], origin, period)
            on = _on_grid(times, origin, period, scatter)
            first, last = _extent(on, first, last, low, high)
        # The notes of one chord, struck a hundredth of a second apart by a player's hand, count once:
        # a few chords far apart sit on many a grid, each note of them at once, and keep no tempo.
        struck = np.concatenate(([True], np.diff(times[first : last + 1]) > COARSE_MOMENT))
        chord = np.cumsum(struck) - 1
        on_chord = np.zeros(int(chord[-1]) + 1, dtype=bool)
        np.logical_or.at(on_chord, chord, on[first : last + 1])
        kept = int(on_chord.sum())
        span = times[last] - times[first]
        if kept < LEAST_ONSETS or kept < 0.8 * len(on_chord) or span < LEAST_STRETCH_BEATS * period:
            continue
        # Notes few and far between (fewer than one to the beat) keep a beat only by falling on it:
        # on the beat and its half, four in five of them, and on two beats in five at the least.
        if len(on_chord) < span / period:
            quarters = np.round((times[first : last + 1][struck] - origin) / (period / STEPS)).astype(np.int64) % STEPS
            halves = np.bincount(quarters % (STEPS // 2), minlength=STEPS // 2)
            if halves.max() < 0.8 * len(on_chord) or len(on_chord) < 0.4 * span / period:
                continue
        kept = int(on[first : last + 1].sum())
        found.append((first, last, Stretch(origin, period, float(times[first]), float(times[last]), scatter, kept)))
    return [stretch for _, _, stretch in sorted(found, key=lambda item: item[0])]


def _phase(stretch: Stretch, tracked: np.ndarray) -> None:
    """Move the stretch's origin onto the quarter of its beat that the tracked beats mark as the beat."""
    near = tracked[(tracked >= stretch.start - stretch.period) & (tracked <= stretch.end + stretch.period)]
    if len(near) == 0:
        return
    angle = np.angle(np.exp(2j * np.pi * (near - stretch.origin) / stretch.period).mean())
    quarter = int(np.round(angle / (2.0 * np.pi) * STEPS)) % STEPS
    stretch.origin += quarter * stretch.period / STEPS


RUNS_ON = 2.0  # beats between two stretches under which the notes run on from one into the other
SPLIT = 0.25  # share of the tracked beats on another quarter above which the tracker has changed its mind


def _in_step(a: Stretch, b: Stretch) -> int | None:
    """How many quarters of the beat the grid of ``b`` is numbered ahead of the grid of ``a``, where the
    notes run on from ``a`` into ``b`` on one grid; None where they do not (a pause, a free bar)."""
    if b.start - a.end > RUNS_ON * max(a.period, b.period):
        return None
    step = min(a.period, b.period) / STEPS
    reach = min(max(3.0 * (a.scatter + b.scatter), 0.008), 0.3 * step)
    if min(a.scatter, b.scatter) > PRECISE and abs(a.period / b.period - 1.0) <= 0.015:
        reach = 0.45 * step  # one tempo, read coarsely (see ``_join``)
    middle = (a.end + b.start) / 2.0
    in_a = int(np.round((middle - a.origin) / (a.period / STEPS)))
    line = a.origin + in_a * a.period / STEPS
    in_b = int(np.round((line - b.origin) / (b.period / STEPS)))
    if abs(line - (b.origin + in_b * b.period / STEPS)) > reach:
        return None
    return (in_b - in_a) % STEPS


def _votes(stretch: Stretch, tracked: np.ndarray) -> np.ndarray:
    """How many of the tracked beats within the stretch fall on each quarter of its grid."""
    near = tracked[
        (tracked >= stretch.start - 0.25 * stretch.period) & (tracked <= stretch.end + 0.25 * stretch.period)
    ]
    quarters = np.round((near - stretch.origin) / (stretch.period / STEPS)).astype(np.int64) % STEPS
    return np.bincount(quarters, minlength=STEPS).astype(np.float64)


# Share of the tracked beats within a stretch that must agree on the beat, or on it and its off-beat.
DECIDED = 0.8


def set_phases(stretches: list[Stretch], tracked: np.ndarray) -> int:
    """Move every stretch's origin onto the quarter of its grid that is the beat; returns how many stretches
    were given another beat than most of the tracked beats within them mark.

    The tracked beats say which quarter that is, stretch by stretch. But the tracker can take
    the off-beat for the beat through a whole passage (a figure whose lowest notes fall between
    the beats) and right itself later, while the notes run on evenly: in the score, half a beat
    would go missing where it does. Notes that run on from one stretch into the next keep one
    beat between them. So stretches that run on, each sure of its beat but for the choice
    between it and its off-beat, decide together, by all the tracked beats within them; and
    where the tracker has plainly changed its mind along the way, the passage is taken to
    begin on a beat, as music does far more often than not, if one of its two answers says so.
    """
    own = [_votes(stretch, tracked) for stretch in stretches]
    said = [int(np.argmax(votes)) for votes in own]
    groups: list[list[tuple[int, int]]] = []  # (stretch, how far its numbering is ahead of its group's first)
    for index, stretch in enumerate(stretches):
        pair = own[index][said[index]] + own[index][(said[index] + STEPS // 2) % STEPS]
        if own[index].sum() == 0 or pair < DECIDED * own[index].sum():
            _phase(stretch, tracked)  # the tracked beats within it are all over the grid: on its own, as they lean
            continue
        ahead = None
        if groups and groups[-1][-1][0] == index - 1:
            before, so_far = groups[-1][-1]
            step = _in_step(stretches[before], stretch)
            first = groups[-1][0][0]
            if step is not None and (said[index] - (so_far + step) - said[first]) % STEPS in (0, STEPS // 2):
                ahead = (so_far + step) % STEPS
        if ahead is None:
            groups.append([(index, 0)])
        else:
            groups[-1].append((index, ahead))
    changed = 0
    for group in groups:
        total = np.zeros(STEPS)
        for index, ahead in group:
            total += np.roll(own[index], -ahead)
        ranked = np.argsort(-total, kind="stable")
        beat = int(ranked[0])
        first = stretches[group[0][0]]
        begins_on = int(np.round((first.start - first.origin) / (first.period / STEPS))) % STEPS
        if total[ranked[1]] >= SPLIT * total.sum() and int(ranked[1]) == begins_on:
            beat = begins_on
        for index, ahead in group:
            quarter = (beat + ahead) % STEPS
            changed += quarter != said[index]
            stretches[index].origin += quarter * stretches[index].period / STEPS
    return changed


PRECISE = (
    0.005  # scatter (seconds) under which a stretch is read finely enough to tell a third from a quarter of the beat
)


def _follow(lines: list[float], times: np.ndarray, period: float) -> list[float]:
    """Beat lines of a coarsely read stretch, each moved with the notes around it.

    A screen recording repeats and skips frames, and the reading of it wanders by a few
    hundredths of a second from one bar to the next: every note of a moment is late or early
    together. The steady beat is right about the tempo; where exactly the beat is at each moment
    is followed from the notes, which lie on quarters of the beat far more often than not.
    """
    out: list[float] = []
    offset = 0.0
    step = period / STEPS
    for line in lines:
        at = line + offset
        near = times[(times >= at - 0.5 * period) & (times < at + 0.5 * period)]
        off = (near - at + step / 2.0) % step - step / 2.0
        off = off[np.abs(off) <= 0.07 * period]  # a third of the beat is 0.083 of it from a quarter
        if len(off):
            offset = float(np.clip(offset + 0.7 * np.median(off), -0.12 * period, 0.12 * period))
        out.append(line + offset)
    return out


@dataclass
class Steady:
    """The beats and bar lines to write the notes on."""

    beats: list[float]
    fitted: list[bool]  # per beat: taken from the notes (True) or from the tracker (False)
    scatter: list[float]  # per beat: how exactly the notes around it keep the beat, seconds (inf: tracker's beat)
    bars: list[dict]  # as the tracker's: start (seconds), beats, unit
    stretches: list[dict]  # each steady stretch: start, end (seconds), bpm, scatter_ms, onsets, beats, then
    regular_bars: int = 0  # bar lines placed by counting beats
    tracked_bars: int = 0  # bar lines left where the tracker heard them
    within: float = 0.012  # seconds within which onsets are one moment (see ``moments``)
    rephased: int = 0  # stretches whose beat is another quarter of their grid than the tracked beats within them mark
    followed: dict | None = None  # the notes keep no click's time: the beat heard was followed (see ``follow``)


def _join(a: Stretch, b: Stretch, times: np.ndarray) -> tuple[str, tuple[int, ...]]:
    """How stretch ``a`` gives way to ``b``.

    ("line", (k, j)): beat line k of ``a`` is beat line j of ``b``: the tempo changes there.
    ("tempo", (k, j, count)): between line k of ``a`` and line j of ``b`` lie ``count`` beats of a
        third tempo, which the onsets between keep (a slowing bar before a new section, say).
    ("loose", ()): neither: a pause or a gradual change.
    """
    reach = min(max(3.0 * (a.scatter + b.scatter), 0.008), 0.3 * min(a.period, b.period) / STEPS)
    if min(a.scatter, b.scatter) > PRECISE and abs(a.period / b.period - 1.0) <= 0.015:
        # One tempo, read coarsely: the two fits are of the same beat, and differ by what the
        # reading wandered between them. Short of half a quarter of the beat, they still agree.
        reach = 0.45 * min(a.period, b.period) / STEPS
    # The change lies between the last onset that keeps to a and the first that keeps to b.
    # (An onset or two just past the change may keep to both tempos, and be held by the wrong one.)
    best: tuple[float, int, int] | None = None
    for k in range(int(np.ceil(a.index(a.end) - 2.0 - 1e-3)), int(np.floor(a.index(b.start) + 2.0 + 1e-3)) + 1):
        line = a.line(k)
        j = int(round(b.index(line)))
        apart = abs(line - b.line(j))
        if apart > reach or (best is not None and apart >= best[0]):
            continue
        late = times[(times > line + reach) & (times <= a.end + 1e-4)]
        early = times[(times >= b.start - 1e-4) & (times < line - reach)]
        if (len(late) and not _on_grid(late, b.origin, b.period, b.scatter).all()) or (
            len(early) and not _on_grid(early, a.origin, a.period, a.scatter).all()
        ):
            continue
        best = (apart, k, j)
    if best is not None:
        return "line", (best[1], best[2])
    gap = times[(times > a.end + 1e-4) & (times < b.start - 1e-4)]
    if len(gap) >= 3:
        low, high = 0.7 * min(a.period, b.period), 1.45 * max(a.period, b.period)
        found: tuple[float, float, int, int, int] | None = None
        for k in range(int(np.floor(a.index(a.end) - 1.0 + 1e-3)), int(np.floor(a.index(gap[0]) + 1e-3)) + 1):
            for j in range(int(np.ceil(b.index(gap[-1]) - 1e-3)), int(np.ceil(b.index(b.start) + 1.0 - 1e-3)) + 1):
                span = b.line(j) - a.line(k)
                for count in range(max(1, int(np.ceil(span / high))), int(np.floor(span / low)) + 1):
                    period = span / count
                    inside = gap[(gap > a.line(k) + reach) & (gap < b.line(j) - reach)]
                    if len(inside) < 3:
                        continue
                    step = period / FINE
                    ticks = np.round((inside - a.line(k)) / step)
                    off = np.abs(inside - a.line(k) - ticks * step)
                    kept = int((off <= min(reach, 0.3 * step)).sum())
                    if kept < len(inside) or kept < len(gap) - 1:
                        continue
                    # A few onsets fit many tempos. The one meant puts them on plain parts of the
                    # beat (beats and halves, not thirds throughout), and is nearest its neighbours.
                    plain = float(np.mean([_PLAIN[int(t) % FINE] for t in ticks]))
                    if plain > 1.0:
                        continue
                    score = (-plain, -abs(float(np.log(period / np.sqrt(a.period * b.period)))))
                    if found is None or score > (found[0], found[1]):
                        found = (score[0], score[1], k, j, count)
        if found is not None:
            return "tempo", (found[2], found[3], found[4])
    return "loose", ()


def _smooth_phases(phases: list[int], count: int, change: float = 3.0) -> list[int]:
    """``phases`` (each bar line's place in a bar of ``count`` beats, as tracked) with the tracker's slips
    taken out: the sequence that disagrees least with them, a change of phase counting as ``change`` slips."""
    if not phases:
        return []
    cost = [[0.0 if p == phases[0] else 1.0 for p in range(count)]]
    back: list[list[int]] = []
    for phase in phases[1:]:
        previous = cost[-1]
        cheapest = min(range(count), key=lambda q: previous[q])
        row, origin = [], []
        for p in range(count):
            stay, move = previous[p], previous[cheapest] + change
            origin.append(p if stay <= move else cheapest)
            row.append(min(stay, move) + (0.0 if p == phase else 1.0))
        cost.append(row)
        back.append(origin)
    state = min(range(count), key=lambda q: cost[-1][q])
    out = [state]
    for origin in reversed(back):
        state = origin[state]
        out.append(state)
    return out[::-1]


def _phases(lines: list[float], tracked_bars: list[dict], usual: tuple[int, int]) -> list[tuple[int, int]] | None:
    """(index into ``lines``, smoothed place in the bar) of each tracked bar line within the beat lines ``lines``, or
    None when the bars tracked there are mostly not of the usual kind (``usual``: beats, unit)."""
    count = usual[0]
    inside = [b for b in tracked_bars if lines[0] - 0.25 * (lines[1] - lines[0]) <= float(b["start"]) <= lines[-1]]
    if len(inside) < 2 or sum((int(b.get("beats") or 4), int(b.get("unit") or 4)) == usual for b in inside) < 0.8 * len(
        inside
    ):
        return None
    at = [int(np.argmin(np.abs(np.asarray(lines) - float(b["start"])))) for b in inside]
    # A bar line tracked about halfway between two beats (the tracker had the off-beat for the
    # beat there) says nothing about which of the two begins the bar.
    beat = float(np.median(np.diff(lines)))
    clear = [i for i, b in zip(at, inside, strict=True) if abs(lines[i] - float(b["start"])) <= 0.3 * beat]
    if len(clear) < 2:
        return None
    return list(zip(clear, _smooth_phases([i % count for i in clear], count), strict=True))


def _on_onsets(a: float, b: float, count: int, gap: np.ndarray, before: float) -> list[float] | None:
    """``count`` beats between the lines ``a`` and ``b``, each on one of the onsets ``gap``, so that the beat
    changes length as evenly as it can (a ritardando: the notes still fall on the beats). None if too few onsets."""
    from itertools import combinations

    if count == 0:
        return []
    if len(gap) < count or len(gap) > 14:
        return None
    best: tuple[float, tuple[float, ...]] | None = None
    for chosen in combinations([float(t) for t in gap], count):
        lengths = np.diff([a, *chosen, b])
        cost = float(np.abs(np.diff(np.log([before, *lengths]))).sum())
        if best is None or cost < best[0]:
            best = (cost, chosen)
    return list(best[1]) if best else None


def _as_written(lines: list[float], times: np.ndarray, reach: float = 0.17) -> list[float]:
    """Beats heard as the score writer goes on to move them (``exact_score.snap_beats``): each onto the note struck
    within ``reach`` of a beat of it, the others keeping their place between those."""
    moved = list(lines)
    anchors: list[int] = []
    for i in range(len(lines)):
        around = [lines[j + 1] - lines[j] for j in (i - 1, i) if 0 <= j < len(lines) - 1]
        near = times[np.abs(times - lines[i]) <= reach * min(around)] if around else times[:0]
        if len(near):
            moved[i] = float(near[np.argmin(np.abs(near - lines[i]))])
            anchors.append(i)
    for a, b in zip(anchors, anchors[1:], strict=False):
        for i in range(a + 1, b):
            moved[i] = moved[a] + (lines[i] - lines[a]) / (lines[b] - lines[a]) * (moved[b] - moved[a])
    if anchors:
        for i in range(anchors[0]):
            moved[i] = lines[i] + moved[anchors[0]] - lines[anchors[0]]
        for i in range(anchors[-1] + 1, len(lines)):
            moved[i] = lines[i] + moved[anchors[-1]] - lines[anchors[-1]]
    return moved if all(b > a for a, b in zip(moved, moved[1:], strict=False)) else list(lines)


HALF = (0.35, 0.65)  # of the beat: how long a beat heard lasts where the tracker goes over to the off-beat, or back
TWICE = (1.6, 2.5)  # of the beat: how long a beat heard lasts where the tracker lost one


def _usual(beats: list[float]) -> np.ndarray:
    """About how long a beat is around each interval between ``beats``: the median of the dozen about it."""
    lengths = np.diff(beats)
    return np.array([float(np.median(lengths[max(0, k - 6) : k + 7])) for k in range(len(lengths))])


def mend(tracked: np.ndarray, times: np.ndarray, tracked_bars: list[dict]) -> tuple[list[float], dict[str, int]]:
    """The beats heard, each on the note struck on it, put right where the tracker contradicts itself; and how many
    beats were put in, taken out, and moved from the off-beat onto the beat.

    A tracker follows a player closely: through a bar broadened before a new verse, a beat
    hurried, a tempo that drifts from line to line. Its mistakes are few, and of a kind its own
    beats show, one against the next:

    - A beat twice as long as those about it, in a bar that then counts a beat too few, is two
      beats, one of them lost. So is every such beat before the first note and after the last,
      where the tracker has nothing to count by and the music's beat goes on as it was.
    - Two beats of half the length are one beat, counted twice.
    - One beat of half the length, or of one and a half with notes struck in it: from there on
      the tracker has the off-beat for the beat, or had it until there. The notes run on through
      the place, so on one side every beat is half a beat out: the side with fewer beats on it,
      unless the piece begins on a beat of that side and it holds a quarter of the beats or more
      (music begins on a beat far more often than off one). That side's beats are put midway
      between where they were heard.
    """
    beats = _as_written([float(t) for t in tracked], times)
    mended = {"put_in": 0, "taken_out": 0, "off_the_off_beat": 0}
    if len(beats) < 8 or len(times) == 0:
        return beats, mended
    first_note, last_note = float(times[0]), float(times[-1])

    # A beat counted twice.
    usual = _usual(beats)
    k = 0
    while k + 2 < len(beats):
        a, b = (beats[k + 1] - beats[k]) / usual[k], (beats[k + 2] - beats[k + 1]) / usual[k]
        if HALF[0] <= a <= HALF[1] and HALF[0] <= b <= HALF[1]:
            del beats[k + 1]
            mended["taken_out"] += 1
            usual = _usual(beats)
        else:
            k += 1

    # A beat lost: within the music, where its bar is a beat short; outside it, wherever the tracker took to longer beats.
    meters = {float(bar["start"]): int(bar.get("beats") or 4) for bar in tracked_bars}
    lines = sorted(meters)
    within = [b for b in beats if first_note - 1e-3 <= b <= last_note + 1e-3]
    music = float(np.median(np.diff(within))) if len(within) >= 3 else float(np.median(np.diff(beats)))
    out = [beats[0]]
    for k in range(1, len(beats)):
        length = beats[k] - beats[k - 1]
        outside = beats[k - 1] <= first_note - 0.25 * music or beats[k - 1] >= last_note - 0.25 * music
        about = music if outside else float(usual[k - 1])
        if TWICE[0] <= length / about <= TWICE[1]:
            short = False
            if not outside:
                bar = [line for line in lines if line <= beats[k - 1] + 0.25 * about]
                later = [line for line in lines if line > beats[k - 1] + 0.25 * about]
                if bar and later:
                    meter = meters[bar[-1]]
                    held = sum(bar[-1] - 0.25 * about <= t < later[0] - 0.25 * about for t in beats)
                    short = held < meter
            if outside or short:
                out.append((beats[k - 1] + beats[k]) / 2.0)
                mended["put_in"] += 1
        out.append(beats[k])
    beats = out

    # The off-beat taken for the beat: the sides of each place where the tracker changed its mind.
    usual = _usual(beats)
    flips = []
    for k in range(len(beats) - 1):
        if beats[k + 1] <= first_note or beats[k] >= last_note:
            continue
        length = (beats[k + 1] - beats[k]) / usual[k]
        struck = (
            (times > beats[k] + 0.15 * (beats[k + 1] - beats[k]))
            & (times < beats[k + 1] - 0.15 * (beats[k + 1] - beats[k]))
        ).any()
        if HALF[0] <= length <= HALF[1] or (1.0 + HALF[0] <= length <= 1.0 + HALF[1] and struck):
            flips.append(k)
    if not flips:
        return beats, mended
    edges = [0, *[k + 1 for k in flips], len(beats)]
    sides = [beats[a:b] for a, b in zip(edges, edges[1:], strict=False)]
    on_side = [sum(first_note - 1e-3 <= t <= last_note + 1e-3 for t in side) for side in sides]
    count = [sum(on_side[0::2]), sum(on_side[1::2])]
    wrong = int(count[1] < count[0])  # 1: the sides numbered odd are the fewer
    # Which side the piece begins on a beat of.
    begins = None
    for n, side in enumerate(sides):
        if side and (side[-1] >= first_note or n == len(sides) - 1):
            beat = float(np.median(np.diff(side))) if len(side) > 1 else music
            off = min(abs(first_note - t) for t in side) / beat
            begins = n % 2 if off <= 0.15 else 1 - n % 2 if abs(off - 0.5) <= 0.15 else None
            break
    if begins == wrong and count[wrong] >= 0.25 * sum(count):
        wrong = 1 - wrong
    out = []
    for n, side in enumerate(sides):
        if n % 2 != wrong:
            out += side
            continue
        beat = float(np.median(np.diff(side))) if len(side) > 1 else music
        moved = [
            side[0] - 0.5 * beat,
            *[(a + b) / 2.0 for a, b in zip(side, side[1:], strict=False)],
            side[-1] + 0.5 * beat,
        ]
        before = sides[n - 1][-1] if n > 0 and sides[n - 1] else None
        after = sides[n + 1][0] if n + 1 < len(sides) and sides[n + 1] else None
        if before is not None and moved[0] - before < 0.6 * beat:
            moved = moved[1:]
        if after is not None and after - moved[-1] < 0.6 * beat:
            moved = moved[:-1]
        out += moved
        mended["off_the_off_beat"] += len(moved)
    return out, mended


ASTRAY = (0.15, 0.35)  # of a beat: how far from the nearest line a beat heard is that lies on none, nor midway


def astray(tracked: np.ndarray, lines: list[float], fitted: list[bool]) -> int:
    """How many of the beats heard, among beat lines fitted to the notes, lie neither on a line nor midway
    between two: a quarter of a beat out, more or less.

    A tracker hearing a machine play is a few hundredths of a second off the lines, or half a
    beat off them (the off-beat taken for the beat), never between. Following a player through a
    broadened bar, or one whose lines have slipped, it is.
    """
    at = np.asarray(lines)
    count = 0
    for time in tracked:
        k = int(np.searchsorted(at, time))
        if 0 < k < len(at) and fitted[k - 1] and fitted[k]:
            off = min(time - at[k - 1], at[k] - time) / (at[k] - at[k - 1])
            count += int(ASTRAY[0] <= off <= ASTRAY[1])
    return count


def follow(
    tracked: np.ndarray,
    times: np.ndarray,
    tracked_bars: list[dict],
    stretches: list[Stretch],
    within: float,
    drift: float | None,
    strays: int,
) -> Steady:
    """The beats and bar lines of notes that keep no click's time: the beat heard, followed.

    Lines drawn evenly through such playing (``find_stretches``) hold for a few bars and then
    are a sixteenth out, or half a beat, with nothing in the notes to say so: a bar broadened by
    half a beat leaves everything after it on quarters of the old lines' beat. The beat heard
    goes with the player. It is put right where the tracker contradicts itself (``mend``), the
    score writer moves each beat onto the note struck on it, and the bar lines are counted in
    beats from the tracked ones that agree with each other.
    """
    beats, mended = mend(tracked, times, tracked_bars)
    usual = max(
        {(int(b.get("beats") or 4), int(b.get("unit") or 4)) for b in tracked_bars} or {(4, 4)},
        key=lambda kind: sum((int(b.get("beats") or 4), int(b.get("unit") or 4)) == kind for b in tracked_bars),
    )
    count, unit = usual
    found = _phases(beats, tracked_bars, usual) if len(beats) >= 2 else None
    bars: list[dict] = []
    regular = 0
    if found is None:
        bars = [dict(b) for b in tracked_bars]
    else:
        governing = 0
        for index, time in enumerate(beats):
            while governing + 1 < len(found) and found[governing + 1][0] <= index:
                governing += 1
            if index % count == found[governing][1]:
                bars.append({"start": time, "beats": count, "unit": unit, "meter": f"{count}/{unit}"})
                regular += 1
    # How exactly the notes keep the beat, for the writer: as a player does, not as a drawing does.
    scatter = float(np.median([s.scatter for s in stretches])) if stretches else 0.012
    scatter = max(scatter, 2.0 * PRECISE)
    return Steady(
        beats,
        [False] * len(beats),
        [scatter] * len(beats),
        bars,
        [],
        regular,
        len(bars) - regular,
        within,
        0,
        {"tempo_wanders": None if drift is None else round(drift, 4), "beats_heard_astray": strays, **mended},
    )


def steady_beats(annotation: dict, onsets: list[float]) -> Steady:
    """The tracked beats and bar lines of ``annotation`` replaced, wherever the notes keep a steady beat, by that beat.

    The beats returned are as many as the music has, not as many as were tracked: a tracker that
    lost or gained a beat is corrected, and bar lines are counted in beats from the tracked ones
    that agree with each other. Where the notes keep no steady beat the tracked beats stay.
    """
    beats = [float(b) for b in annotation["beats"]]
    tracked_bars = [dict(b) for b in annotation.get("bars") or []]
    tracked = np.asarray(beats, dtype=np.float64)
    within = 0.012
    times = moments(onsets, within)
    stretches = find_stretches(beats, times)
    if stretches and float(np.median([s.scatter for s in stretches])) > PRECISE:
        # Read coarsely (a screen recording at 30 frames a second puts the notes of one chord a
        # frame apart): what is struck within a frame of each other is one moment.
        within = COARSE_MOMENT
        times = moments(onsets, within)
        stretches = find_stretches(beats, times)
    if not stretches:
        return Steady(
            beats, [False] * len(beats), [float("inf")] * len(beats), tracked_bars, [], 0, len(tracked_bars), within
        )
    rephased = set_phases(stretches, tracked)
    first = [int(np.floor(s.index(s.start) + 1e-3)) for s in stretches]
    last = [int(np.ceil(s.index(s.end) - 1e-3)) for s in stretches]
    joins = [_join(a, b, times) for a, b in zip(stretches, stretches[1:], strict=False)]
    pieces: list[list[float]] = []  # the fitted lines of each stretch, with those of a tempo between it and the next
    for i, stretch in enumerate(stretches):
        if i > 0 and joins[i - 1][0] in ("line", "tempo"):
            first[i] = joins[i - 1][1][1]
        extra: list[float] = []
        if i < len(joins) and joins[i][0] == "line":
            last[i] = joins[i][1][0] - 1
        elif i < len(joins) and joins[i][0] == "tempo":
            k, j, count = joins[i][1]
            last[i] = k
            step = (stretches[i + 1].line(j) - stretch.line(k)) / count
            extra = [stretch.line(k) + step * n for n in range(1, count)]
        lines = [stretch.line(k) for k in range(first[i], last[i] + 1)]
        if stretch.scatter > PRECISE:
            lines = _follow(lines, times, stretch.period)
        pieces.append(lines + extra)

    # Chains of stretches joined beat for beat: within one the beats are counted, so its bar lines can be too.
    chains: list[list[int]] = [[0]]
    for i, (kind, _) in enumerate(joins):
        if kind == "loose":
            chains.append([i + 1])
        else:
            chains[-1].append(i + 1)
    usual = max(
        {(int(b.get("beats") or 4), int(b.get("unit") or 4)) for b in tracked_bars} or {(4, 4)},
        key=lambda kind: sum((int(b.get("beats") or 4), int(b.get("unit") or 4)) == kind for b in tracked_bars),
    )
    count, unit = usual
    starts = np.array([float(b["start"]) for b in tracked_bars])

    def phases(lines: list[float]) -> list[tuple[int, int]] | None:
        return _phases(lines, tracked_bars, usual)

    out = [float(t) for t in tracked if t < pieces[0][0] - 0.5 * stretches[0].period]
    fitted = [False] * len(out)
    scatter = [float("inf")] * len(out)
    bars: list[dict] = [b for b in tracked_bars if float(b["start"]) < pieces[0][0] - 0.5 * stretches[0].period]
    regular = 0
    for number, chain in enumerate(chains):
        lines = [t for i in chain for t in pieces[i]]
        scatter += [stretches[i].scatter for i in chain for _ in pieces[i]]
        found = phases(lines) if len(lines) >= 2 else None
        if found is None:
            bars += [
                b
                for b in tracked_bars
                if lines[0] - 0.25 * stretches[chain[0]].period <= float(b["start"]) <= lines[-1]
            ]
            place_of_last = None
        else:
            governing = 0
            for index, time in enumerate(lines):
                while governing + 1 < len(found) and found[governing + 1][0] <= index:
                    governing += 1
                if index % count == found[governing][1]:
                    bars.append({"start": time, "beats": count, "unit": unit, "meter": f"{count}/{unit}"})
                    regular += 1
            place_of_last = (len(lines) - 1 - found[-1][1]) % count
        out += lines
        fitted += [True] * len(lines)
        if number == len(chains) - 1:
            break
        # The gap to the next chain: no steady beat. Its beats are counted so that the bars on both
        # sides stay in step, and put on the notes; failing that the tracked beats stay.
        a, b = stretches[chain[-1]], stretches[chains[number + 1][0]]
        following = [t for i in chains[number + 1] for t in pieces[i]]
        after = phases(following) if len(following) >= 2 else None
        # Two fits of one beat, a little apart where a player's tempo drifts: the beat this chain ends on
        # may be the one the next begins on. It is the next chain's.
        while len(out) > 1 and out[-1] > following[0] - 0.5 * min(a.period, b.period):
            gone = out.pop()
            fitted.pop()
            scatter.pop()
            kept = [bar for bar in bars if float(bar["start"]) != gone]
            regular -= len(bars) - len(kept)
            bars = kept
            if place_of_last is not None:
                place_of_last = (place_of_last - 1) % count
        low, high = out[-1], following[0]
        heard = [float(t) for t in tracked if low + 0.5 * a.period < t < high - 0.5 * b.period]
        placed: list[float] | None = None
        if place_of_last is not None and after is not None:
            place_of_next = (0 - after[0][1]) % count
            wanted = (place_of_next - place_of_last - 1) % count
            candidates = sorted({wanted + count * n for n in range(0, 4)}, key=lambda n: abs(n - len(heard)))
            gap = times[(times > low + 1e-3) & (times < high - 1e-3)]
            placed = _on_onsets(low, high, candidates[0], gap, a.period)
            if placed is None and len(gap) < candidates[0]:
                # Fewer notes than beats: a chord held through a pause. Its beats are not on notes, only
                # counted, as many as keep the bars in step, and as long as the pause makes them.
                length = (high - low) / (candidates[0] + 1)
                if 0.6 * min(a.period, b.period) <= length <= 1.8 * max(a.period, b.period):
                    placed = [low + length * n for n in range(1, candidates[0] + 1)]
        if placed is None:
            placed = heard
            bars += [b for b in tracked_bars if low < float(b["start"]) < high - 0.25 * _tracked_beat(b, starts)]
        else:
            # Bar lines through the gap, in step with the chain before it.
            for offset, time in enumerate(placed, start=1):
                if (place_of_last + offset) % count == 0:
                    bars.append({"start": time, "beats": count, "unit": unit, "meter": f"{count}/{unit}"})
                    regular += 1
        out += placed
        fitted += [placed is not heard] * len(placed)
        scatter += [max(a.scatter, b.scatter) if placed is not heard else float("inf")] * len(placed)
    tail = [float(t) for t in tracked if t > out[-1] + 0.5 * stretches[-1].period]
    bars += [b for b in tracked_bars if float(b["start"]) > out[-1] + 0.25 * stretches[-1].period]
    out += tail
    fitted += [False] * len(tail)
    scatter += [float("inf")] * len(tail)
    bars.sort(key=lambda b: float(b["start"]))
    described = [
        {
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "bpm": round(60.0 / s.period, 3),
            "scatter_ms": round(1000.0 * s.scatter, 2),
            "onsets": s.onsets,
            "beats": len(piece),
            "then": joins[i][0] if i < len(joins) else "end",
        }
        for i, (s, piece) in enumerate(zip(stretches, pieces, strict=True))
    ]
    if within == COARSE_MOMENT:
        # Read coarsely, a drawing played by a machine and a player keeping time alone look alike
        # note by note. They differ in the long run, and the lines just drawn are only good for
        # the first: its tempo holds from window to window, and every beat heard lies on a line
        # or (the tracker having the off-beat for the beat) midway between two.
        drift = wander(tracked, times)
        strays = astray(tracked, out, fitted)
        if drift is None or drift > WANDERS or strays:
            return follow(tracked, times, tracked_bars, stretches, within, drift, strays)
    return Steady(out, fitted, scatter, bars, described, regular, len(bars) - regular, within, rephased)


def _tracked_beat(bar: dict, starts: np.ndarray) -> float:
    """About how long a beat of the tracked ``bar`` is, from the next tracked bar line."""
    later = starts[starts > float(bar["start"])]
    length = float(later[0] - float(bar["start"])) if len(later) else 2.0
    return length / float(bar.get("beats") or 4)

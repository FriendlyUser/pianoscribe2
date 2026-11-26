"""Read the falling notes of a piano-tutorial video back into a piano roll.

The notes of such a video scroll down the screen at a steady speed, so the video is a
piano roll seen through a moving window. This module puts the roll back together and
measures the bars on it:

1. **Zones.** For every key, a few narrow column ranges say whether a bar is over that
   key (``build_zones``). A white key is read between its black neighbours, where no
   other key's bar can be. A black key's bar lies over the edges of two white keys, so
   it is read in two parts, one over each neighbour, and the part over a neighbour that
   is itself sounding is not trusted.
2. **Strips.** Every frame contributes a thin strip of rows a little above the keyboard
   (above the flare drawn where notes land), reduced at once to per-zone numbers
   (``reduce_strip``): how much of the zone is not the empty background, how much of
   it is as bright as a bar, the colour of that part, and how uniform it is.
3. **Motion.** The same notes pass a second strip higher up some frames earlier; that
   delay gives the scrolling speed (``scroll_speed``). A screen recording does not show
   the roll at even intervals, though: its frames can repeat or skip, so how far the
   picture really moved between two frames is also measured (``frame_shifts``).
4. **Unrolling.** Each strip row is placed on the roll by how far it still has to travel
   to the keyboard (``unroll``). One row of the roll is one pixel of travel: a few
   milliseconds.
5. **Bars.** Along each key, runs of covered rows are notes (``read_bars``).

Two thresholds do different jobs. A bar is *there* wherever its key's zone is not
background, however dark: the outlines of the note names printed inside bars are nearly
black, and must not read as holes. Two bars are *separate* where the background shows at
the sides of the key between them: bars have rounded corners, so even two bars that touch
leave a notch there, while a note name never reaches the sides. A quarter of the notes in
a typical tutorial touch the next note on the same key, so the notch is what separates
repeated notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.visual.keyboard import Keyboard

# A pixel has the colour of a bar (or of a lit key) when a channel differs from the empty
# picture by more than SOLID. Backgrounds are near black and bars are saturated colours.
SOLID = 45
# A pixel is not background when a channel differs by more than FAINT: this also takes in
# the dark outlines drawn around bars and around the note names inside them.
FAINT = 20
COVERED = 0.5  # share of a zone that must be covered for a bar to be there
HOLE_COVERED = 0.75  # ... for a short hole in a black key's bar to be closed
EDGE_EMPTY = 0.35  # a corner strip this empty shows the background between two bars
EDGE_FULL = 0.6
SAME_COLOUR = 50.0  # largest channel difference between two readings of one bar's colour
UNIFORM = 25.0  # largest grey-level spread across a zone covered by one flat colour

Float = NDArray[np.floating[Any]]
Bool = NDArray[np.bool_]


@dataclass(frozen=True)
class Zones:
    """Column ranges read for each key. Index arrays are per key; -1 means "none"."""

    x0: NDArray[np.int64]  # [zones] first column
    x1: NDArray[np.int64]  # [zones] last column + 1
    core: NDArray[np.int64]  # white keys: the zone between the black neighbours
    left: NDArray[np.int64]  # black keys: the part over the white key to the left
    right: NDArray[np.int64]  # black keys: the part over the white key to the right
    edge_left: NDArray[np.int64]  # corner strip at the bar's left edge
    edge_right: NDArray[np.int64]
    white_left: NDArray[np.int64]  # black keys: index (into keys) of the white key to the left
    white_right: NDArray[np.int64]
    black_left: NDArray[np.int64]  # white keys: index of the black key on the left boundary
    black_right: NDArray[np.int64]
    readable: Bool  # keys with enough columns in the picture to be read
    coloured: int  # the first this-many zones are the ones colours are read from

    @property
    def count(self) -> int:
        return len(self.x0)

    def width(self, zone: NDArray[np.int64]) -> NDArray[np.int64]:
        return np.where(zone >= 0, self.x1[zone] - self.x0[zone], 0)


def build_zones(keyboard: Keyboard) -> Zones:
    keys = keyboard.keys
    period = keyboard.white_width
    width = keyboard.width
    margin = max(2, round(0.075 * period))  # clear of a black neighbour's bar and its label
    free = max(3, round(0.2 * period))  # clear of the rounded corners at a free side
    inset = max(1, round(0.06 * period))  # inside a black key's bar
    seam = max(1, round(0.04 * period))  # clear of the boundary between two white keys
    strip = max(2, round(0.09 * period))
    black_strip = max(2, round(0.06 * period))
    nudge = max(1, round(0.02 * period))

    spans: list[tuple[int, int]] = []

    def zone(start: float, stop: float, least: int = 2) -> int:
        a, b = max(0, int(np.ceil(start))), min(width, int(np.floor(stop)))
        if b - a < least:
            return -1
        spans.append((a, b))
        return len(spans) - 1

    count = len(keys)
    arrays = {name: np.full(count, -1, dtype=np.int64) for name in (
        "core", "left", "right", "edge_left", "edge_right", "white_left", "white_right", "black_left", "black_right",
    )}  # fmt: skip
    by_pitch = {key.pitch: index for index, key in enumerate(keys)}
    # The zones a bar's colour is read from come first, so that colours are kept for those only.
    for index, key in enumerate(keys):
        if key.black:
            assert key.boundary is not None
            arrays["left"][index] = zone(key.x0 + inset, key.boundary - seam)
            arrays["right"][index] = zone(key.boundary + seam, key.x1 - inset)
            arrays["white_left"][index] = by_pitch.get(key.pitch - 1, -1)
            arrays["white_right"][index] = by_pitch.get(key.pitch + 1, -1)
        else:
            below, above = by_pitch.get(key.pitch - 1, -1), by_pitch.get(key.pitch + 1, -1)
            has_left = below >= 0 and keys[below].black
            has_right = above >= 0 and keys[above].black
            start = keys[below].x1 + margin if has_left else key.x0 + free
            stop = keys[above].x0 - margin if has_right else key.x1 - free
            arrays["core"][index] = zone(start, stop, least=3)
            arrays["black_left"][index] = below if has_left else -1
            arrays["black_right"][index] = above if has_right else -1
    coloured = len(spans)
    for index, key in enumerate(keys):
        if key.black:
            arrays["edge_left"][index] = zone(key.x0 + inset, key.x0 + inset + black_strip)
            arrays["edge_right"][index] = zone(key.x1 - inset - black_strip, key.x1 - inset)
        else:
            arrays["edge_left"][index] = zone(key.x0 + nudge, key.x0 + nudge + strip)
            arrays["edge_right"][index] = zone(key.x1 - nudge - strip, key.x1 - nudge)
    readable = np.array(
        [
            (arrays["left"][i] >= 0 or arrays["right"][i] >= 0) if key.black else arrays["core"][i] >= 0
            for i, key in enumerate(keys)
        ]
    )
    x0 = np.array([a for a, _ in spans], dtype=np.int64)
    x1 = np.array([b for _, b in spans], dtype=np.int64)
    return Zones(x0=x0, x1=x1, readable=readable, coloured=coloured, **arrays)


def _short_gaps(dim: Bool, longest: int, axis: int) -> Bool:
    """The pixels of ``dim`` in a run along ``axis`` of at most ``longest``, with a pixel that is not dim at both ends."""
    size = dim.shape[axis]
    shape = [1] * dim.ndim
    shape[axis] = size
    place = np.arange(size, dtype=np.int32).reshape(shape)
    before = np.maximum.accumulate(np.where(dim, -1, place), axis=axis)
    after = np.flip(np.minimum.accumulate(np.flip(np.where(dim, size, place), axis=axis), axis=axis), axis=axis)
    short: Bool = dim & (before >= 0) & (after < size) & (after - before - 1 <= longest)
    return short


def enclosed(dim: Bool, across: int, along: int) -> Bool:
    """The pixels of ``dim`` ([frames, rows, columns]) in a gap between pixels that are not dim, of
    at most ``across`` columns side to side or ``along`` rows up and down: a closing of the rest.

    In front of a film, the dark outline of a bar and the dark strokes of the note name printed
    on it are as dim as the film. Unlike the film they lie in short gaps between bright parts of
    a bar; the film between two bars side by side is wider than ``across``. A gap is filled whole
    or not at all, so that the gap between two notes on one key is either kept or closed (and the
    notes are then told apart by their corners, as when they touch).
    """
    return _short_gaps(dim, along, 1) | _short_gaps(dim, across, 2)


@dataclass
class StripStats:
    """What each zone shows in each row of a strip, for a run of frames."""

    solid: Float  # [frames, rows, zones] share of the zone that has the colour of a bar
    grey: Float  # [frames, rows, zones] mean grey level of the zone, whatever is in it
    cover: Float  # [frames, rows, coloured zones] share that is not background, however dark
    colour: Float  # [frames, rows, coloured zones, 3] mean RGB of the solid pixels
    spread: Float  # [frames, rows, coloured zones] standard deviation of their grey level


def reduce_strip(
    pixels: NDArray[np.uint8],
    background: NDArray[np.int16],
    zones: Zones,
    *,
    faint: int = FAINT,
    ignore: Bool | None = None,
    covered: Bool | None = None,
) -> StripStats:
    """Per-zone statistics of ``pixels`` ([frames, rows, columns, 3]) against ``background`` ([rows, columns, 3]).

    With a film playing behind the notes, ``ignore`` ([frames, rows, columns]) marks the pixels
    to take as background whatever they differ by (dimmer than any bar), and ``covered`` those
    of them to count as covered all the same (the dark outlines inside a bar; see ``enclosed``).
    """
    frames, rows, width, _ = pixels.shape
    distance = np.abs(pixels.astype(np.int16) - background[None]).max(axis=-1)
    if ignore is not None:
        distance[ignore] = 0
    solid = distance > SOLID
    x0, x1 = zones.x0, zones.x1
    some = zones.coloured

    def zone_sums(values: NDArray[np.generic], count: int) -> NDArray[np.int64]:
        total = np.zeros((frames, rows, width + 1) + values.shape[3:], dtype=np.int64)
        np.cumsum(values, axis=2, out=total[:, :, 1:])
        return total[:, :, x1[:count]] - total[:, :, x0[:count]]

    size = (x1 - x0).astype(np.float32)
    solid_count = zone_sums(solid, zones.count)
    faint_count = zone_sums((distance > faint) | covered if covered is not None else distance > faint, some)
    present = np.maximum(solid_count[:, :, :some], 1).astype(np.float32)
    colour_sum = zone_sums(pixels * solid[..., None], some)
    grey = pixels.astype(np.uint16).sum(axis=-1) // 3
    grey_sum = zone_sums(grey * solid, some)
    square_sum = zone_sums(grey.astype(np.int64) ** 2 * solid, some)
    mean_grey = grey_sum / present
    variance = np.maximum(square_sum / present - mean_grey**2, 0.0)
    return StripStats(
        solid=(solid_count / size).astype(np.float32),
        grey=(zone_sums(grey, zones.count) / size).astype(np.float32),
        cover=(faint_count / size[:some]).astype(np.float32),
        colour=(colour_sum / present[..., None]).astype(np.float32),
        spread=np.sqrt(variance).astype(np.float32),
    )


def scroll_speed(upper: Float, lower: Float, distance: float, *, least: float, most: float) -> float | None:
    """Rows per frame at which notes fall, from the delay between two strips ``distance`` rows apart.

    ``upper`` and ``lower`` are [frames, zones] cover of the same zones in the higher and the
    lower strip. Returns None when too little passes through to tell.
    """
    frames = len(upper)
    lags = np.arange(max(1, int(np.floor(distance / most))), min(frames - 1, int(np.ceil(distance / least))) + 1)
    if len(lags) < 3:
        return None
    a = upper - upper.mean(axis=0, keepdims=True)
    b = lower - lower.mean(axis=0, keepdims=True)
    if float((a**2).sum()) < 1.0 or float((b**2).sum()) < 1.0:
        return None
    size = 1 << int(np.ceil(np.log2(2 * frames)))
    spectrum = np.conj(np.fft.rfft(a, size, axis=0)) * np.fft.rfft(b, size, axis=0)
    correlation = np.fft.irfft(spectrum.sum(axis=1), size)[lags]
    best = int(np.argmax(correlation))
    lag = float(lags[best])
    if 0 < best < len(lags) - 1:
        before, at, after = correlation[best - 1 : best + 2]
        bend = before - 2 * at + after
        if bend < 0:
            lag += float(0.5 * (before - after) / bend)
    norm = float(np.sqrt((a**2).sum() * (b**2).sum()))
    if correlation[best] < 0.3 * norm:
        return None
    return distance / lag


def frame_shifts(masks: Bool, most: int) -> Float:
    """Rows the picture moved down between each frame and the next, from ``masks`` ([frames, rows, columns]).

    ``masks`` marks the bar-coloured pixels of a tall strip of the roll in consecutive frames.
    Returns one value per pair of neighbouring frames; NaN where too little shows to tell
    (nothing on screen, or only bars that span the strip, which look the same wherever they are).
    """
    before, after = masks[:-1], masks[1:]
    count, rows = len(before), masks.shape[1]
    scores = np.zeros((most + 1, count), dtype=np.float32)
    for shift in range(most + 1):
        a, b = before[:, : rows - shift], after[:, shift:]
        union = (a | b).sum(axis=(1, 2))
        scores[shift] = (a & b).sum(axis=(1, 2)) / np.maximum(union, 1)
    best = scores.argmax(axis=0)
    top = scores.max(axis=0)
    index = np.arange(count)
    moved = best.astype(np.float32)
    inner = (best > 0) & (best < most)
    lower = scores[np.clip(best - 1, 0, most), index]
    upper = scores[np.clip(best + 1, 0, most), index]
    bend = lower - 2 * top + upper
    moved = np.where(inner & (bend < 0), moved + 0.5 * (lower - upper) / np.minimum(bend, -1e-6), moved)
    content = np.minimum(before.sum(axis=(1, 2)), after.sum(axis=(1, 2)))
    clear = (top >= 0.5) & (top - np.median(scores, axis=0) >= 0.08) & (content >= 0.002 * before[0].size)
    return np.where(clear, moved, np.nan).astype(np.float32)


def jump_shift(before: Bool, after: Bool, least: int, most: int) -> float:
    """Rows the picture jumped down between two frames, looked for from ``least`` to ``most`` rows; NaN if untold.

    ``before`` and ``after`` are the masks ``frame_shifts`` compares ([rows, columns]). A screen
    recording that stutters shows one picture for a while and then one much further on: further
    than a frame is followed in the ordinary way, so it is looked for over nearly the whole
    height of the strip, where the two pictures share only a few rows.
    """
    rows = before.shape[0]
    most = min(most, rows - 4)
    if most < least:
        return float("nan")
    scores = np.zeros(most - least + 1, dtype=np.float32)
    for n, shift in enumerate(range(least, most + 1)):
        a, b = before[: rows - shift], after[shift:]
        union = int((a | b).sum())
        if union >= 0.002 * before.size:  # enough of the two pictures in what they share
            scores[n] = float((a & b).sum()) / union
    best = int(scores.argmax())
    if scores[best] < 0.5 or scores[best] - float(np.median(scores)) < 0.08:
        return float("nan")
    return float(least + best)


@dataclass
class Roll:
    """The piano roll put back together: one row per pixel of travel."""

    solid: Float  # [rows, zones] share of each zone that has the colour of a bar
    grey: Float  # [rows, zones] mean grey level of each zone
    cover: Float  # [rows, coloured zones] share that is not background
    colour: Float  # [rows, coloured zones, 3]
    spread: Float  # [rows, coloured zones]
    seen: Bool  # [rows] rows some frame showed
    across: Float  # [rows] how much lighter or darker than the empty picture most of the row is (a line drawn across)
    origin: float  # how far the roll had scrolled where row 0 reaches the keyboard
    shown: NDArray[np.float64]  # [frames] how far the roll had scrolled in each frame, as shown
    even: NDArray[np.float64]  # [frames] the same with the unevenness of a screen recording taken out
    speed: float  # rows per frame, on average
    filled: int = 0  # rows that never passed through the strip, taken from above it (see ``unroll``)

    def frame_of(self, row: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """The frame in which roll row ``row`` is seen reaching the keyboard."""
        return np.interp(self.origin + np.asarray(row, dtype=np.float64), self.shown, np.arange(len(self.shown)))

    def clock_of(self, row: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """When roll row ``row`` reaches the keyboard, in frames of steady time.

        A screen recording shows the roll at uneven moments, but the roll itself runs
        evenly with the sound: a row's time is where it lies on the roll, not the frame
        that happened to show it land.
        """
        return np.interp(self.origin + np.asarray(row, dtype=np.float64), self.even, np.arange(len(self.even)))


def unroll(
    solid: NDArray[np.uint8],
    grey: NDArray[np.uint8],
    cover: NDArray[np.uint8],
    colour: NDArray[np.uint8],
    spread: NDArray[np.uint8],
    frames: NDArray[np.int64],
    shown: NDArray[np.float64],
    even: NDArray[np.float64],
    lead: float,
    *,
    across: Float | None = None,
    spare: dict[int, tuple[NDArray[np.uint8], ...]] | None = None,
) -> Roll:
    """Place every strip row on the roll.

    ``solid`` ([frames, rows, zones], 0..255), ``cover``, ``colour`` and ``spread`` hold the note
    strip of the frames numbered ``frames``; the strip's last row is ``lead`` rows above the row
    where notes land. ``shown`` says how far the roll had scrolled in every frame of the video.
    A row seen in several frames is averaged. ``across`` ([frames, rows]) is averaged the same way.

    ``spare`` holds, for a few frames, the same five readings of the rows just above the strip
    ([rows above, zones], the lowest of them last). Where the picture moved further between two
    frames than the strip is tall (a recording that stuttered, and caught up in one jump), the
    rows that never passed through the strip are taken from above it in the frame before.
    """
    rows, zones, coloured = solid.shape[1], solid.shape[2], cover.shape[2]
    # Row r of the strip is (lead + rows - 1 - r) rows above the landing row.
    travel = lead + (rows - 1 - np.arange(rows))
    scrolled = shown[frames]
    origin = float(scrolled[0] + travel.min())
    length = int(np.rint(scrolled[-1] + travel.max() - origin)) + 1
    out_solid = np.zeros((length, zones), dtype=np.float32)
    out_grey = np.zeros((length, zones), dtype=np.float32)
    out_cover = np.zeros((length, coloured), dtype=np.float32)
    out_colour = np.zeros((length, coloured, 3), dtype=np.float32)
    out_spread = np.zeros((length, coloured), dtype=np.float32)
    out_across = np.zeros(length, dtype=np.float32)
    times = np.zeros(length, dtype=np.float32)

    def place(
        index: NDArray[np.int64],
        solid_: NDArray[np.uint8],
        grey_: NDArray[np.uint8],
        cover_: NDArray[np.uint8],
        colour_: NDArray[np.uint8],
        spread_: NDArray[np.uint8],
    ) -> None:
        nonlocal times
        times += np.bincount(index, minlength=length).astype(np.float32)
        weight = solid_.reshape(-1, zones).astype(np.float32) / 255.0
        np.add.at(out_solid, index, weight)
        np.add.at(out_grey, index, grey_.reshape(-1, zones).astype(np.float32))
        np.add.at(out_cover, index, cover_.reshape(-1, coloured).astype(np.float32) / 255.0)
        part = weight[:, :coloured]
        np.add.at(out_colour, index, colour_.reshape(-1, coloured, 3).astype(np.float32) * part[..., None])
        np.add.at(out_spread, index, spread_.reshape(-1, coloured).astype(np.float32) * part)

    block = 1500  # frames at a time, to keep the float copies of the strips small
    for begin in range(0, len(frames), block):
        chosen = slice(begin, begin + block)
        position = scrolled[chosen, None] + travel[None, :]
        index = np.clip(np.rint(position - origin).astype(np.int64), 0, length - 1).ravel()
        place(index, solid[chosen], grey[chosen], cover[chosen], colour[chosen], spread[chosen])
        if across is not None:
            np.add.at(out_across, index, across[chosen].reshape(-1).astype(np.float32))
    filled = 0
    if spare:
        where = {int(frame): n for n, frame in enumerate(frames)}
        for frame, kept in sorted(spare.items()):
            n = where.get(frame)
            if n is None or n + 1 >= len(frames):
                continue
            high = kept[0].shape[0]  # rows above the strip, the lowest last: row r is (lead + rows + high - 1 - r) up
            position = scrolled[n] + lead + rows + (high - 1 - np.arange(high))
            index = np.rint(position - origin).astype(np.int64)
            # Every row of the roll passes through the strip of some frame, but those a jump passed over.
            fresh = (index >= 0) & (index < length)
            fresh[fresh] = times[index[fresh]] == 0
            if not fresh.any():
                continue
            place(index[fresh], *(part[fresh] for part in kept))
            filled += int(fresh.sum())
    weighed = np.maximum(out_solid[:, :coloured], 1e-6)
    out_colour /= weighed[..., None]
    out_spread /= weighed
    seen = np.maximum(times, 1.0)[:, None]
    out_solid /= seen
    out_grey /= seen
    out_cover /= seen
    out_across /= np.maximum(times, 1.0)
    # Strictly increasing, so that a place on the roll has one frame (a repeated frame scrolls nothing).
    ramp = 1e-6 * np.arange(len(shown))
    return Roll(
        solid=out_solid,
        grey=out_grey,
        cover=out_cover,
        colour=out_colour,
        spread=out_spread,
        seen=times > 0,
        across=out_across,
        filled=filled,
        origin=origin,
        shown=shown + ramp,
        even=even + ramp,
        speed=float((shown[frames[-1]] - shown[frames[0]]) / max(1, frames[-1] - frames[0])),
    )


LINE_LEVEL = 3.0  # grey levels most of a row must differ by from the empty picture for a line to be drawn there


def lines_across(roll: Roll, widest: int) -> list[float]:
    """The rows of the roll at which a thin line is drawn right across it, in order: the bar lines of the music.

    A roll made from a score often shows its bars: a faint line across the whole picture where
    each begins, falling with the notes. Such a line shifts most of a row away from the empty
    picture by the same few grey levels (``Roll.across``), which no note does, for no more than
    ``widest`` rows. The lines of one video all look alike: what is lighter where they are
    darker, or half as strong, is something else.
    """
    level = np.where(roll.seen, roll.across, 0.0).astype(np.float64)
    if len(level) == 0:
        return []
    noise = 1.4826 * float(np.median(np.abs(level - np.median(level))))
    floor = max(LINE_LEVEL, 8.0 * noise)
    found: list[tuple[float, float, int]] = []  # row, strength, lighter (+1) or darker (-1)
    for sign in (1, -1):
        for a, b in _runs(sign * level > floor):
            if b - a > widest:
                continue  # a caption, a fade: nothing that thin
            weight = sign * level[a:b]
            found.append((a + float((weight * np.arange(b - a)).sum() / weight.sum()), float(weight.max()), sign))
    if len(found) < 4:
        return []
    sign = 1 if sum(s for _, _, s in found) >= 0 else -1
    usual = float(np.median([strength for _, strength, s in found if s == sign]))
    return sorted(row for row, strength, s in found if s == sign and 0.5 * usual <= strength <= 2.0 * usual)


@dataclass(frozen=True)
class Bar:
    """One falling note as measured on the roll."""

    key: int  # index into Keyboard.keys
    start: float  # roll row of its lower edge (the moment the note is struck)
    end: float  # roll row just past its upper edge (the moment it is released)
    colour: tuple[float, float, float]  # mean RGB
    joined: bool = False  # separated from a touching bar by its corners, not by a gap
    doubtful: bool = False  # a black key read between two sounding white keys
    shared: bool = False  # on a key the other hand has a bar on too, one drawn over the other


def _runs(mask: Bool) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def _zone(values: Float, zone: int) -> Float:
    return values[:, zone] if zone >= 0 else np.zeros(len(values), dtype=np.float32)


def _differs(a: Float, b: Float) -> Bool:
    return np.abs(a - b).max(axis=-1) > SAME_COLOUR


def _black_presence(roll: Roll, zones: Zones, index: int, present: Bool) -> tuple[Bool, Bool, Bool]:
    """(present, doubtful, closable) rows of black key ``index``, given which white keys sound."""
    rows = len(roll.solid)
    left, right = int(zones.left[index]), int(zones.right[index])
    width_left = float(zones.x1[left] - zones.x0[left]) if left >= 0 else 0.0
    width_right = float(zones.x1[right] - zones.x0[right]) if right >= 0 else 0.0

    def both(values: Float) -> Float:
        return (_zone(values, left) * width_left + _zone(values, right) * width_right) / (width_left + width_right)

    white_left, white_right = int(zones.white_left[index]), int(zones.white_right[index])
    left_on = present[:, white_left] if white_left >= 0 else np.zeros(rows, dtype=bool)
    right_on = present[:, white_right] if white_right >= 0 else np.zeros(rows, dtype=bool)
    alone = ~left_on & ~right_on

    def own_colour(part: int, white: int) -> Bool:
        """Rows where ``part``, over the sounding white key ``white``, is one flat colour not that key's."""
        if part < 0 or white < 0:
            return np.zeros(rows, dtype=bool)
        flat = (roll.solid[:, part] >= 0.9) & (roll.spread[:, part] <= UNIFORM)
        return flat & _differs(roll.colour[:, part], roll.colour[:, zones.core[white]])

    # Beside a sounding white key, only the part over the other, silent white key can be this
    # key's own bar; it is asked for the colour of a bar, since the neighbour's dark outline reaches it.
    # A black key that lies nearly all over one neighbour has no part over the other (its zone
    # there is too narrow to read); beside that neighbour sounding, its bar is told by its colour.
    far_right = _zone(roll.solid, right) >= COVERED if right >= 0 else own_colour(left, white_left)
    far_left = _zone(roll.solid, left) >= COVERED if left >= 0 else own_colour(right, white_right)
    state = np.where(alone, both(roll.cover) >= COVERED, np.where(left_on & ~right_on, far_right, far_left))
    faint_right = _zone(roll.cover, right) >= HOLE_COVERED if right >= 0 else own_colour(left, white_left)
    faint_left = _zone(roll.cover, left) >= HOLE_COVERED if left >= 0 else own_colour(right, white_right)
    closable = np.where(alone, both(roll.cover) >= COVERED, np.where(left_on & ~right_on, faint_right, faint_left))
    # Between two sounding white keys both parts are covered whatever this key does. Its bar is
    # there only if the zone is one flat colour that is neither neighbour's.
    between = left_on & right_on
    if between.any():
        flat = np.ones(rows, dtype=bool)
        other = np.ones(rows, dtype=bool)
        for part, white in ((left, white_left), (right, white_right)):
            if part < 0:
                continue
            flat &= (roll.solid[:, part] >= 0.9) & (roll.spread[:, part] <= UNIFORM)
            other &= _differs(roll.colour[:, part], roll.colour[:, zones.core[white]])
        if left >= 0 and right >= 0:
            flat &= ~_differs(roll.colour[:, left], roll.colour[:, right])
        state = np.where(between, flat & other, state)
        closable = np.where(between, False, closable)
    return state, between, closable


def _close_holes(state: Bool, closable: Bool, longest: int) -> Bool:
    """Close short holes in the runs of ``state`` wherever ``closable`` holds throughout the hole."""
    out = state.copy()
    for a, b in _runs(~state):
        if a == 0 or b == len(state) or b - a > longest:
            continue
        if closable[a:b].all():
            out[a:b] = True
    return out


def key_presence(roll: Roll, zones: Zones, keyboard: Keyboard) -> tuple[Bool, Bool]:
    """(present, doubtful): for every roll row and key, whether a bar covers the key."""
    rows = len(roll.solid)
    keys = keyboard.keys
    present = np.zeros((rows, len(keys)), dtype=bool)
    doubtful = np.zeros((rows, len(keys)), dtype=bool)
    longest_hole = max(4, round(0.25 * keyboard.white_width))
    blacks = [i for i, key in enumerate(keys) if key.black and zones.readable[i]]
    for index, key in enumerate(keys):
        if not key.black and zones.core[index] >= 0:
            present[:, index] = roll.cover[:, zones.core[index]] >= COVERED
    for index in blacks:
        present[:, index], _, _ = _black_presence(roll, zones, index, present)
    # A note name printed in black on a bar reads as a hole in its middle; the bar is still
    # there if it carries on at the sides of the key.
    none = np.zeros(rows, dtype=bool)
    for index, key in enumerate(keys):
        if key.black or zones.core[index] < 0:
            continue
        sides = np.zeros(rows, dtype=bool)
        for strip, other in (
            (int(zones.edge_left[index]), int(zones.black_left[index])),
            (int(zones.edge_right[index]), int(zones.black_right[index])),
        ):
            if strip >= 0:
                hidden = present[:, other] if other >= 0 else none
                sides |= ~hidden & (roll.solid[:, strip] >= EDGE_FULL)
        present[:, index] = _close_holes(present[:, index], sides, longest_hole)
    for index in blacks:
        state, between, closable = _black_presence(roll, zones, index, present)
        present[:, index] = _close_holes(state, closable, longest_hole)
        doubtful[:, index] = between
    return present, doubtful


DIM = 0.5  # a zone this much darker than the bar around it, measured from the background, is a line across it
# Below this many columns per white key the corners of a bar are a pixel or two and do not
# survive video compression; the dark line between two touching bars is read instead. Above
# it the corners are read alone: there, the strokes of a note name can pass for such a line.
SMALL_KEYS = 40.0


def _dim(level: Float, floor: float) -> Bool:
    """Rows where ``level`` (grey along one bar) falls well below the bar's own level."""
    body = float(np.median(level))
    if body - floor < 30.0:
        return np.zeros(len(level), dtype=bool)
    return (level - floor) <= DIM * (body - floor)


def _split_at_notches(
    start: int,
    end: int,
    edges: list[tuple[Float, Bool, Float]],
    middle: Float,
    spread: Float,
    grey: Float,
    tone: Float,
    floor: float,
    least: int,
    longest: int,
    lines: bool,
) -> list[tuple[int, int, bool]]:
    """Cut the run [start, end) wherever two touching bars meet.

    Two signs are read. The corners: bars are rounded, so where two meet the background shows
    at the sides (``edges``: per side, the cover of its corner strip, the rows where another
    key's bar hides it, and its grey level). And the outlines: each bar is drawn with a dark
    edge, so where two meet a dark line crosses the whole width, sides and ``grey`` middle
    alike. The second sign is what is left of the first in a small picture, and is read
    only there (``lines``).

    ``edges`` holds, for each side of the bar, how much of its corner strip has the colour
    of a bar, and the rows where another key's bar hides that strip. A notch is a short
    stretch where every visible strip shows the background, with the bar carrying on at
    its sides both before and after. Where both strips are hidden, the bar's ``middle``
    losing its colour (the dark outlines of two bars) is the only sign.

    The glyphs of a note name can darken both sides of a narrow bar at once. They are told
    from a notch by the bar's middle: across a name it is a mix of colours (``spread``) or
    lighter than the bar (``tone``, the grey of what has a bar's colour there: a name is
    printed in white); across a notch it is the bar's own colour or its dark outline.
    """
    length = end - start
    visible = np.zeros(length, dtype=bool)
    empty = np.ones(length, dtype=bool)
    full = np.zeros(length, dtype=bool)
    dark = _dim(grey[start:end], floor) if lines else np.zeros(length, dtype=bool)
    for cover, hidden, level in edges:
        seen = ~hidden[start:end]
        visible |= seen
        empty &= ~seen | (cover[start:end] < EDGE_EMPTY)
        full |= seen & (cover[start:end] >= EDGE_FULL)
        dark &= ~seen | _dim(level[start:end], floor)
    unseen = ~visible
    body_tone = float(np.median(tone[start:end]))
    notch = (visible & (empty | dark)) | (unseen & (middle[start:end] < COVERED))
    full = np.where(unseen, middle[start:end] >= EDGE_FULL, full)
    pieces: list[tuple[int, int, bool]] = []
    cursor, joined = start, False
    for a, b in _runs(notch):
        if b - a > longest:
            continue
        # The run's own ends are rounded too: a notch needs a stretch of whole bar on both sides.
        if full[:a].sum() < least or full[b:].sum() < least:
            continue
        if float(spread[start + a : start + b].min()) > UNIFORM:
            continue
        if float(tone[start + a : start + b].max()) > body_tone + 25.0:
            continue
        cut = start + (a + b) // 2
        if cut - cursor >= least and end - cut >= least:
            pieces.append((cursor, cut, joined))
            cursor, joined = cut, True
    pieces.append((cursor, end, joined))
    return pieces


def _split_at_colour_change(start: int, end: int, colour: Float, least: int) -> list[tuple[int, int]]:
    """Cut the run where its colour changes for good (two hands on one key, back to back)."""
    if end - start < 2 * least:
        return [(start, end)]
    values = colour[start:end]
    chroma = np.stack([values[:, 0] - values[:, 1], (values[:, 0] + values[:, 1]) / 2 - values[:, 2]], axis=1)
    best, best_gain = None, 0.0
    total = chroma.sum(axis=0)
    running = np.cumsum(chroma, axis=0)
    for cut in range(least, end - start - least + 1):
        head = running[cut - 1] / cut
        tail = (total - running[cut - 1]) / (end - start - cut)
        gain = float(np.linalg.norm(head - tail))
        if gain > best_gain:
            best, best_gain = cut, gain
    if best is None:
        return [(start, end)]
    head = running[best - 1] / best
    tail = (total - running[best - 1]) / (end - start - best)
    strength = min(float(np.linalg.norm(head)), float(np.linalg.norm(tail)))
    if strength < 25.0:
        return [(start, end)]
    cosine = float(head @ tail) / (float(np.linalg.norm(head)) * float(np.linalg.norm(tail)))
    if cosine > np.cos(np.radians(40.0)):
        return [(start, end)]
    return [(start, start + best), (start + best, end)]


def _chroma(values: Float) -> Float:
    """What colour each of ``values`` ([n, 3] RGB) is, whatever its brightness: [n, 2]."""
    return np.stack([values[:, 0] - values[:, 1], (values[:, 0] + values[:, 1]) / 2 - values[:, 2]], axis=1)


def _unlike(a: Float, b: Float) -> bool:
    """Whether two colours (as ``_chroma`` has them) are different colours, both clearly coloured."""
    strength = min(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    if strength < 25.0:
        return False
    cosine = float(a @ b) / (float(np.linalg.norm(a)) * float(np.linalg.norm(b)))
    return bool(cosine <= np.cos(np.radians(40.0)))


def _colour_pieces(start: int, end: int, colour: Float, least: int) -> list[tuple[int, int, int]] | None:
    """The run [start, end) cut wherever it goes from one of two colours to the other:
    (first row, row after the last, which colour: 0 or 1) in order, or None if it has one colour throughout.

    Unlike ``_split_at_colour_change`` this finds every change, not the one that divides the run
    best: a bar of one hand drawn over the middle of a longer bar of the other is three pieces.
    """
    if end - start < 2 * least:
        return None
    chroma = _chroma(colour[start:end].astype(np.float64))
    centred = chroma - chroma.mean(axis=0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]  # the direction along which the colours differ most
    along = centred @ axis
    labels = along > (float(along.min()) + float(along.max())) / 2.0
    for _ in range(4):
        if labels.all() or not labels.any():
            return None
        low, high = chroma[~labels].mean(axis=0), chroma[labels].mean(axis=0)
        again = np.linalg.norm(chroma - high, axis=1) < np.linalg.norm(chroma - low, axis=1)
        if bool((again == labels).all()):
            break
        labels = again
    if labels.all() or not labels.any() or not _unlike(chroma[~labels].mean(axis=0), chroma[labels].mean(axis=0)):
        return None
    pieces = [[a, b, 1] for a, b in _runs(labels)] + [[a, b, 0] for a, b in _runs(~labels)]
    pieces.sort()
    while len(pieces) > 1:
        # Where two colours meet a row or two are neither: a piece too short to be a bar belongs to its neighbours.
        k = min(range(len(pieces)), key=lambda i: pieces[i][1] - pieces[i][0])
        if pieces[k][1] - pieces[k][0] >= least:
            break
        if 0 < k < len(pieces) - 1:
            pieces[k - 1 : k + 2] = [[pieces[k - 1][0], pieces[k + 1][1], pieces[k - 1][2]]]
        elif k == 0:
            pieces[:2] = [[pieces[0][0], pieces[1][1], pieces[1][2]]]
        else:
            pieces[-2:] = [[pieces[-2][0], pieces[-1][1], pieces[-2][2]]]
    if len(pieces) < 2:
        return None
    return [(start + a, start + b, label) for a, b, label in pieces]


FAINT_NOTCH = 0.3  # share of a bar's corner strips that must show through for a faint seam
DEEP_NOTCH = 0.45  # ... for one that needs nothing else to say so (one corner strip of two showing through)


def faint_notches(
    roll: Roll, zones: Zones, keyboard: Keyboard, bars: list[Bar], *, least: int, wide: int
) -> list[tuple[int, float, float]]:
    """(index into ``bars``, roll row, depth) of each place where a bar looks faintly like two bars meeting.

    Where two rounded bars meet, the background shows at their corners. In a small or blurred
    picture that notch is a row or two of the corner strips somewhat less covered than the
    rest of the bar: too little to cut a bar on outright (``_split_at_notches`` asks for
    strips all but empty), enough to count with something else that says a note is struck
    there. Only white keys are looked at (the name printed on a black key's narrow bar reaches
    its sides), only strips no neighbouring key's bar lies over, and a notch has whole bar
    within ``wide`` rows on both sides and lies ``least`` rows or more from either end: the
    bar's own rounded ends are no seam.
    """
    present, _ = key_presence(roll, zones, keyboard)
    found: list[tuple[int, float, float]] = []
    for number, bar in enumerate(bars):
        if keyboard.keys[bar.key].black:
            continue
        sides = [
            (int(zones.edge_left[bar.key]), int(zones.black_left[bar.key])),
            (int(zones.edge_right[bar.key]), int(zones.black_right[bar.key])),
        ]
        first, last = int(np.ceil(bar.start)), int(np.floor(bar.end))
        if last - first < 2 * least + 3:
            continue
        total = np.zeros(last - first, dtype=np.float64)
        seen = np.zeros(last - first, dtype=np.float64)
        for strip, other in sides:
            if strip < 0:
                continue
            clear = ~present[first:last, other] if other >= 0 else np.ones(last - first, dtype=bool)
            total += np.where(clear, roll.solid[first:last, strip], 0.0)
            seen += clear
        if not (seen > 0).any():
            continue
        cover = np.where(seen > 0, total / np.maximum(seen, 1.0), np.nan)
        usual = float(np.nanmedian(cover[1:-1]))
        row = least
        while row < len(cover) - least:
            depth = usual - cover[row]
            lowest = np.nanmin(cover[max(0, row - 2) : row + 3])
            if not np.isnan(depth) and depth >= FAINT_NOTCH and cover[row] <= lowest:
                before, after = cover[max(0, row - wide) : row], cover[row + 1 : row + wide + 1]
                whole = usual - 0.5 * FAINT_NOTCH
                if np.nanmax(np.append(before, -1.0)) >= whole and np.nanmax(np.append(after, -1.0)) >= whole:
                    found.append((number, float(first + row), float(depth)))
                    row += wide
            row += 1
    return found


def without_a_bars_colour(bars: list[Bar], roll: Roll, zones: Zones, keyboard: Keyboard) -> list[bool]:
    """Per bar, whether it nowhere has the colour of a bar.

    A pixel that differs a little from the empty picture is not background (``FAINT``): that
    takes in the dark outline drawn around a bar, so that an outline does not part the bar it
    runs through. It also takes in what a screen recording leaves a few rows past a bar's end:
    a glow, dimmer than the outline, and where another bar follows closely on the same key, a
    piece of its own between the two. A piece made of that alone is covered, and is no bar: a
    bar has a bar's colour (``SOLID``) across half its key in one row of it at the least.
    """
    faint = []
    for bar in bars:
        if keyboard.keys[bar.key].black:
            parts = [int(zones.left[bar.key]), int(zones.right[bar.key])]
        else:
            parts = [int(zones.core[bar.key])]
        widths = [float(zones.x1[part] - zones.x0[part]) if part >= 0 else 0.0 for part in parts]
        a, b = int(np.floor(bar.start)), max(int(np.ceil(bar.end)), int(np.floor(bar.start)) + 1)
        share = sum(_zone(roll.solid, part)[a:b] * width for part, width in zip(parts, widths, strict=True)) / sum(
            widths
        )
        faint.append(bool(float(np.max(share)) < COVERED))
    return faint


@dataclass(frozen=True)
class _Stretch:
    """A stretch of one key covered without a seam: one bar, or bars of two colours touching or overlapping."""

    key: int
    start: int
    end: int
    joined: bool
    reference: int  # the zone its colour is read from
    colour: tuple[float, float, float]
    cuts: list[tuple[int, int]]  # as two bars back to back, if its colour changes for good (else itself)
    pieces: list[tuple[int, int, int]] | None  # every piece of one colour, in order (see ``_colour_pieces``)


def read_bars(roll: Roll, zones: Zones, keyboard: Keyboard, *, least_rows: int = 4) -> list[Bar]:
    """Every bar on the roll, in order of key and time."""
    present, doubtful = key_presence(roll, zones, keyboard)
    rows = len(roll.solid)
    keys = keyboard.keys
    none = np.zeros(rows, dtype=bool)
    longest_notch = max(6, round(0.6 * keyboard.white_width))
    floor = float(np.percentile(roll.grey[roll.seen], 20)) if roll.seen.any() else 0.0  # the empty background
    bars: list[Bar] = []
    stretches: list[_Stretch] = []
    for index, key in enumerate(keys):
        if not zones.readable[index]:
            continue
        if key.black:
            sides = [
                (int(zones.edge_left[index]), int(zones.white_left[index])),
                (int(zones.edge_right[index]), int(zones.white_right[index])),
            ]
            parts = [(int(zones.left[index]), int(zones.white_left[index]))]
            parts.append((int(zones.right[index]), int(zones.white_right[index])))
        else:
            sides = [
                (int(zones.edge_left[index]), int(zones.black_left[index])),
                (int(zones.edge_right[index]), int(zones.black_right[index])),
            ]
            parts = [(int(zones.core[index]), -1)]
        widths = [float(zones.x1[part] - zones.x0[part]) if part >= 0 else 0.0 for part, _ in parts]
        middle = sum(_zone(roll.solid, part) * width for (part, _), width in zip(parts, widths, strict=True)) / sum(
            widths
        )
        solid = sum(_zone(roll.solid, part) * width for (part, _), width in zip(parts, widths, strict=True))
        mixed = sum(
            _zone(roll.spread, part) * _zone(roll.solid, part) * width
            for (part, _), width in zip(parts, widths, strict=True)
        ) / np.maximum(solid, 1e-6)
        edges = [
            (roll.solid[:, strip], present[:, other] if other >= 0 else none, roll.grey[:, strip])
            for strip, other in sides
            if strip >= 0
        ]
        tone = sum(
            roll.colour[:, part].mean(axis=1) * _zone(roll.solid, part) * width
            for (part, _), width in zip(parts, widths, strict=True)
            if part >= 0
        ) / np.maximum(solid, 1e-6)
        level = sum(_zone(roll.grey, part) * width for (part, _), width in zip(parts, widths, strict=True)) / sum(
            widths
        )
        for start, end in _runs(present[:, index]):
            if end - start < least_rows:
                continue
            for a, b, joined in _split_at_notches(
                start,
                end,
                edges,
                np.asarray(middle),
                mixed,
                np.asarray(level),
                tone,
                floor,
                least_rows,
                longest_notch,
                lines=keyboard.white_width < SMALL_KEYS,
            ):
                # Colour of the bar: from the parts no other key's bar lies under, where there are any.
                weights, sums = 0.0, np.zeros(3, dtype=np.float64)
                for trusted in (True, False):
                    for part, other in parts:
                        if part < 0:
                            continue
                        clear = ~present[a:b, other] if other >= 0 else np.ones(b - a, dtype=bool)
                        rows_used = clear if trusted else np.ones(b - a, dtype=bool)
                        weight = roll.solid[a:b, part] * rows_used
                        weights += float(weight.sum())
                        sums += (roll.colour[a:b, part] * weight[:, None]).sum(axis=0)
                    if weights > 0:
                        break
                reference = parts[0][0] if parts[0][0] >= 0 else parts[1][0]
                whole = sums / max(weights, 1e-6)
                stretches.append(
                    _Stretch(
                        index,
                        a,
                        b,
                        joined,
                        reference,
                        (float(whole[0]), float(whole[1]), float(whole[2])),
                        _split_at_colour_change(a, b, roll.colour[:, reference], 3 * least_rows),
                        _colour_pieces(a, b, roll.colour[:, reference], 3 * least_rows),
                    )
                )

    def tone_of(reference: int, c: int, d: int) -> Float:
        piece = roll.solid[c:d, reference]
        mean: Float = (roll.colour[c:d, reference] * piece[:, None]).sum(axis=0) / max(float(piece.sum()), 1e-6)
        return mean

    def bar(stretch: _Stretch, c: int, d: int, mean: Float | None = None) -> Bar:
        colour = stretch.colour if mean is None else (float(mean[0]), float(mean[1]), float(mean[2]))
        return Bar(
            key=stretch.key,
            start=float(c),
            end=float(d),
            colour=colour,
            joined=stretch.joined or c != stretch.start,
            doubtful=bool(doubtful[c:d, stretch.key].mean() > 0.5),
            shared=stretch.pieces is not None,
        )

    covered: list[tuple[int, float, float]] = []  # a bar whose end is hidden: where it was covered, where it may end
    for stretch in stretches:
        pieces = stretch.pieces
        if pieces is None:
            for c, d in stretch.cuts:
                same = (c, d) == (stretch.start, stretch.end)
                bars.append(bar(stretch, c, d, None if same else tone_of(stretch.reference, c, d)))
            continue
        if len(pieces) == 2 and len(stretch.cuts) == 2:
            pieces = [(*stretch.cuts[0], pieces[0][2]), (*stretch.cuts[1], pieces[1][2])]
        # Two colours on one key with no seam between: two hands on it at once. A bar is drawn over
        # whatever began before it, so going up the roll a colour that has not been there is a note
        # that begins (over the other, which may run on beneath), and a colour that comes back is the
        # other note ending, the one beneath showing again.
        begun = {pieces[0][2]: pieces[0][0]}  # each colour's note that has begun and not ended: where it began
        hidden: dict[int, int] = {}  # ... and where it was covered, if it is
        shown: dict[int, list[tuple[int, int]]] = {0: [], 1: []}  # the pieces of each colour's note that show
        shown[pieces[0][2]].append(pieces[0][:2])
        sounding: list[
            tuple[int, int, int, int | None, list[tuple[int, int]]]
        ] = []  # colour, from, to, covered at, shown
        for (_, row, old), (c, d, colour) in zip(pieces, pieces[1:], strict=False):
            if colour in begun:
                sounding.append((old, begun.pop(old), row, None, shown[old]))
                shown[old] = []
                hidden.pop(colour, None)
            else:
                begun[colour] = row
                hidden[old] = row
            shown[colour].append((c, d))
        last = pieces[-1][2]
        sounding.append((last, begun.pop(last), pieces[-1][1], None, shown[last]))
        for colour, row in begun.items():
            sounding.append((colour, row, hidden[colour], hidden[colour], shown[colour]))
        for _colour, first_row, last_row, covered_at, parts_shown in sorted(sounding, key=lambda note: note[1]):
            weight = np.array([float(roll.solid[c:d, stretch.reference].sum()) for c, d in parts_shown])
            mean = np.sum(
                [tone_of(stretch.reference, c, d) * w for (c, d), w in zip(parts_shown, weight, strict=True)], axis=0
            ) / max(float(weight.sum()), 1e-6)
            bars.append(bar(stretch, first_row, last_row, mean))
            if covered_at is not None:
                covered.append((len(bars) - 1, float(covered_at), float(stretch.end)))
    # Where a bar ends beneath another it cannot be seen to. The notes its hand struck with it say: a chord is
    # let go together. Failing them it ends where it was covered.
    together = 2.0 * least_rows
    for position, from_row, at_most in covered:
        beneath = bars[position]
        own = _chroma(np.asarray(beneath.colour)[None])[0]
        ends = [
            other.end
            for k, other in enumerate(bars)
            if k != position
            and other.key != beneath.key
            and abs(other.start - beneath.start) <= together
            and not _unlike(_chroma(np.asarray(other.colour)[None])[0], own)
        ]
        if ends and from_row <= float(np.median(ends)) <= at_most + together:
            last_seen = min(float(np.median(ends)), at_most)
            bars[position] = Bar(
                beneath.key, beneath.start, last_seen, beneath.colour, beneath.joined, beneath.doubtful, True
            )
    bars.sort(key=lambda bar: (bar.start, keys[bar.key].pitch))
    return bars

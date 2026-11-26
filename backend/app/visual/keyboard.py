"""Find the on-screen keyboard of a piano-tutorial video and name every key.

A tutorial video ("Synthesia style") shows a keyboard along the bottom and notes
falling onto it. Everything the reader needs to know about a pitch follows from
where its key is, so the keyboard is measured first:

* **Rows.** The band of black keys is a row pattern no other part of a frame has:
  bright and dark runs alternating dozens of times. Below it, the fronts of the white
  keys are a bright band.
* **White keys.** The thin lines between white keys, seen below the black keys, lie
  on a regular lattice; one period is one white key.
* **Black keys and note names.** Black keys sit on five of every seven white-key
  boundaries (not on E|F or B|C). That pattern has no symmetry, so matching it
  names every key.
* **Octave.** The picture alone names the octave only when all 88 keys are shown.
  Otherwise middle C is taken to be the C nearest the middle of the keyboard, which
  the reader then checks against the pitch of the audio (``app.visual.audio``).

Lit keys and falling notes are removed first by taking the median of many frames.

A drawn keyboard has two flat tones, and fixed grey levels tell them apart. A **filmed**
keyboard may be lit unevenly (dim at the ends, tinted by a strip of lights along its top) and
is seen at an angle, so that its keys are a little narrower along the top, where the notes
land, than at the front. When the fixed levels find no keyboard, it is looked for again by
each stretch's own two tones (``filmed_rows``), and its keys are measured where the notes land,
on the lattice its black keys themselves lie on (``measure_filmed_keyboard``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.errors import JobError

BRIGHT = 170  # grey level above which a pixel is a white key front
DARK = 75  # grey level below which a pixel is a black key
# Share of a row of the keyboard that is clearly one of its two tones. A drawn keyboard is all
# but entirely bright or dark (DRAWN: rows across the black keys, rows of white key fronts); a
# filmed one has shading on its white keys, reflections on its black keys and the player's hands
# over both, which leave about a fifth of a row in between (FILMED). A drawn keyboard is looked
# for first: the looser level also moves where a drawn keyboard's rows end, and with them, by a
# fraction of a pixel, where its keys' sides are measured to be.
DRAWN = (0.82, 0.8)
FILMED = (0.72, 0.72)
HIDDEN_CHROMA = 40  # spread between the strongest and weakest channel above which a pixel is not a key (skin)
WHITE_SEMITONES = (0, 2, 4, 5, 7, 9, 11)  # C D E F G A B
# Whether a black key sits on the boundary to the right of each white key, C to B.
BLACK_AFTER = (True, True, False, True, True, True, False)
MIDDLE_C = 60
FULL_KEYBOARD_WHITES = 52

Gray = NDArray[np.float32]


@dataclass(frozen=True)
class Key:
    pitch: int  # MIDI note number
    black: bool
    x0: float  # left edge, in columns
    x1: float  # right edge (exclusive)
    boundary: float | None = None  # black keys: the white-key boundary the key sits on

    @property
    def center(self) -> float:
        return (self.x0 + self.x1) / 2.0


@dataclass(frozen=True)
class Keyboard:
    top: int  # first row of the keys; notes fall onto this row
    black_bottom: int  # last row of the black keys
    bottom: int  # last row of the keyboard
    width: int  # columns of the frame
    white_width: float  # columns per white key
    keys: tuple[Key, ...]  # in pitch order
    octave_certain: bool  # True when all 88 keys are visible, so the octave is not a guess

    @property
    def whites(self) -> list[Key]:
        return [key for key in self.keys if not key.black]

    @property
    def blacks(self) -> list[Key]:
        return [key for key in self.keys if key.black]

    @property
    def lowest(self) -> int:
        return self.keys[0].pitch

    @property
    def highest(self) -> int:
        return self.keys[-1].pitch

    def transposed(self, semitones: int) -> Keyboard:
        """The same keys named ``semitones`` higher (used to move the octave guess)."""
        keys = tuple(Key(k.pitch + semitones, k.black, k.x0, k.x1, k.boundary) for k in self.keys)
        return Keyboard(
            self.top, self.black_bottom, self.bottom, self.width, self.white_width, keys, self.octave_certain
        )


def to_gray(frames: NDArray[np.uint8]) -> Gray:
    return frames.astype(np.float32).mean(axis=-1)


def _runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Runs of True as (first index, last index + 1)."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def keyboard_rows(gray: Gray, tones: tuple[float, float] = DRAWN) -> tuple[int, int, int] | None:
    """(top, black_bottom, bottom) rows of the keyboard in one frame, or None if it shows none.

    ``tones``: the share of a row that must be clearly bright or dark, across the black keys and
    across the white key fronts (DRAWN or FILMED).
    """
    across_black, across_white = tones
    height, width = gray.shape
    bright = gray > BRIGHT
    dark = gray < DARK
    bright_share = bright.mean(axis=1)
    dark_share = dark.mean(axis=1)
    changes = (bright[:, 1:] != bright[:, :-1]).sum(axis=1)
    black_band = (
        (bright_share > 0.3) & (bright_share < 0.8) & (dark_share > 0.2) & (bright_share + dark_share > across_black)
    ) & (changes >= 24)
    runs = [run for run in _runs(black_band) if run[1] - run[0] >= max(8, 0.04 * height)]
    if not runs:
        return None
    top, end = max(runs, key=lambda run: run[1] - run[0])
    # Note names printed across the black keys break the band for a few rows: join what belongs together.
    for start, stop in _runs(black_band):
        if 0 < start - end <= max(3, 0.2 * (end - top)):
            end = stop
    white_band = bright_share > across_white
    below = [run for run in _runs(white_band) if 0 <= run[0] - end <= max(4, 0.01 * height)]
    if not below:
        return None
    bottom = max(below, key=lambda run: run[1] - run[0])[1]
    if bottom - end < max(4, 0.01 * height) or top < 0.25 * height:
        return None  # no white key fronts under the black keys, or no room above for falling notes
    return top, end - 1, bottom - 1


def _lattice(positions: NDArray[np.float64], width: int) -> tuple[float, float] | None:
    """Period and offset of the regular lattice most of ``positions`` lie on, or None."""
    if len(positions) < 8:
        return None
    gaps = np.diff(positions)
    period = float(np.median(gaps))
    usual = gaps[(gaps > 0.6 * period) & (gaps < 1.4 * period)]
    if len(usual) < 6:
        return None
    period = float(np.median(usual))
    origin = float(positions[len(positions) // 2])
    for _ in range(3):
        index = np.round((positions - origin) / period)
        inliers = np.abs(positions - (origin + index * period)) < 0.15 * period
        if inliers.sum() < 8:
            return None
        slope, intercept = np.polyfit(index[inliers], positions[inliers], 1)
        period, origin = float(slope), float(intercept)
    index = np.round((positions - origin) / period)
    inliers = np.abs(positions - (origin + index * period)) < 0.15 * period
    if inliers.mean() < 0.7 or not (8.0 <= period <= width / 7.0):
        return None
    return period, origin


def _white_boundaries(fronts: Gray, width: int) -> tuple[float, float] | None:
    """(white key width, column of one boundary) from the lines between white key fronts."""
    profile = np.median(fronts, axis=0)
    level = float(np.median(profile))
    dips = profile < level - max(18.0, 0.12 * level)
    centers = np.array([(a + b - 1) / 2.0 for a, b in _runs(dips) if b - a <= 0.25 * width / 20.0])
    return _lattice(centers, width)


def _black_keys(band: Gray, period: float) -> list[tuple[float, float]]:
    """Left and right edges of each black key seen whole in the black-key band."""
    profile = np.median(band, axis=0)
    # A black key is where the picture is nearer black than white. On a filmed keyboard the sloping
    # sides of the black keys catch the light: a fixed dark level finds only their middles.
    dark, light = np.percentile(profile, [5, 90])
    spans = []
    for start, stop in _runs(profile < max(DARK + 25, (dark + light) / 2.0)):
        if 0.3 * period <= stop - start <= 0.95 * period and start > 0 and stop < band.shape[1]:
            spans.append((float(start), float(stop)))
    return spans


def hidden_columns(clean: NDArray[np.uint8], rows: tuple[int, int, int]) -> NDArray[np.bool_]:
    """Columns where something coloured lies over the black keys in ``clean`` ([rows, columns, 3]).

    Keys are white, grey or black. A filmed keyboard has the player's hands over it, and where
    they rest most of the time they are in the clean picture too, hiding the keys under them.
    """
    top, black_bottom, _ = rows
    depth = black_bottom - top + 1
    band = clean[top + int(0.15 * depth) : top + int(0.6 * depth)].astype(np.int16)
    chroma = band.max(axis=-1) - band.min(axis=-1)
    hidden: NDArray[np.bool_] = (chroma > HIDDEN_CHROMA).mean(axis=0) > 0.3
    return hidden


def measure_keyboard(clean: Gray, rows: tuple[int, int, int], hidden: NDArray[np.bool_] | None = None) -> Keyboard:
    """Name every key of a keyboard picture with no lit keys (``clean`` is one grey frame).

    ``hidden`` marks columns where the keys cannot be seen (hands over them): the pattern of
    black keys is matched where they can.
    """
    top, black_bottom, bottom = rows
    height, width = clean.shape
    if hidden is None:
        hidden = np.zeros(width, dtype=bool)
    failure = JobError(
        "keyboard_not_found",
        "No piano keyboard could be read in this video. The piano-video option needs a tutorial video with "
        "notes falling onto a keyboard along the bottom of the picture.",
    )
    fronts_top = black_bottom + 1 + int(0.3 * (bottom - black_bottom))
    fronts_bottom = max(fronts_top + 2, bottom + 1 - int(0.1 * (bottom - black_bottom)))
    lattice = _white_boundaries(clean[fronts_top:fronts_bottom], width)
    if lattice is None:
        raise failure
    period, origin = lattice

    depth = black_bottom - top + 1
    spans = _black_keys(clean[top + int(0.15 * depth) : top + int(0.6 * depth)], period)
    if len(spans) < 5:
        raise failure
    first = int(np.floor((0 - origin) / period))
    last = int(np.ceil((width - origin) / period))
    boundaries = {n: origin + n * period for n in range(first, last + 1)}
    black_at: dict[int, tuple[float, float]] = {}
    for x0, x1 in spans:
        n = int(round(((x0 + x1) / 2.0 - origin) / period))
        if n in boundaries and abs((x0 + x1) / 2.0 - boundaries[n]) < 0.35 * period:
            black_at[n] = (x0, x1)

    # White key j lies between boundaries j and j+1. Find which white key is a C.
    def seen(n: int) -> bool:
        a, b = int(boundaries[n] - 0.35 * period), int(boundaries[n] + 0.35 * period) + 1
        return not hidden[max(0, a) : min(width, b)].any()

    checked = [n for n in boundaries if 0.5 * period < boundaries[n] < width - 0.5 * period and seen(n)]
    scores = []
    for offset in range(7):
        agree = sum((n in black_at) == BLACK_AFTER[(n - 1 + offset) % 7] for n in checked)
        scores.append(agree / max(1, len(checked)))
    offset = int(np.argmax(scores))
    ranked = sorted(scores, reverse=True)
    if ranked[0] < 0.9 or ranked[0] - ranked[1] < 0.15 or len(checked) < 7:
        raise failure
    # A black key under a hand: where the pattern puts one, with the width and the place
    # beside its boundary that the black keys of its kind have where they can be seen.
    widths = [x1 - x0 for x0, x1 in black_at.values()]
    places: dict[int, list[float]] = {}
    for n, (x0, x1) in black_at.items():
        places.setdefault((n - 1 + offset) % 7, []).append((x0 + x1) / 2.0 - boundaries[n])
    for n in boundaries:
        kind = (n - 1 + offset) % 7
        if n in black_at or not BLACK_AFTER[kind] or seen(n):
            continue
        centre = boundaries[n] + float(np.median(places.get(kind, [0.0])))
        half = float(np.median(widths)) / 2.0
        if centre - half > 0 and centre + half < width:
            black_at[n] = (centre - half, centre + half)

    whites = [j for j in range(first, last) if min(boundaries[j + 1], width) - max(boundaries[j], 0) >= 0.4 * period]
    cs = [j for j in whites if (j + offset) % 7 == 0]
    if not cs:
        raise failure
    # Every white key of a piano, the lowest an A: the whole keyboard (A0 to C8), even with a sliver of an end key
    # cut off by the edge of the picture.
    certain = len(whites) == FULL_KEYBOARD_WHITES and (whites[0] + offset) % 7 == 5
    if certain:
        middle_c = whites[0] + 23
    else:
        middle = (boundaries[whites[0]] + boundaries[whites[-1] + 1]) / 2.0
        middle_c = min(cs, key=lambda j: (round(abs((boundaries[j] + boundaries[j + 1]) / 2.0 - middle), 3), j))

    keys: list[Key] = []
    for j in whites:
        steps = j - middle_c
        pitch = MIDDLE_C + 12 * (steps // 7) + WHITE_SEMITONES[steps % 7]
        keys.append(Key(pitch, False, max(0.0, boundaries[j]), min(float(width), boundaries[j + 1])))
        if j + 1 in black_at:
            x0, x1 = black_at[j + 1]
            keys.append(Key(pitch + 1, True, x0, x1, boundaries[j + 1]))
    keys.sort(key=lambda key: key.pitch)
    return Keyboard(top, black_bottom, bottom, width, period, tuple(keys), certain)


TOLD_APART = 60.0  # grey levels between the dark and the light tone of a stretch for the two to be told apart


def _tones(values: Gray, reach: int) -> tuple[Gray, Gray]:
    """The darkest and the lightest value within ``reach`` columns of each column of ``values`` ([..., columns])."""
    pad = [(0, 0)] * (values.ndim - 1) + [(reach, reach)]
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(values, pad, mode="edge"), 2 * reach + 1, axis=-1)
    return windows.min(axis=-1), windows.max(axis=-1)


def filmed_rows(gray: Gray) -> tuple[int, int, int] | None:
    """(top, black_bottom, bottom) rows of a keyboard in uneven light, or None if ``gray`` shows none.

    Across the black keys every stretch of a row holds a dark tone and a light one, far apart,
    however bright that stretch of the keyboard is; a pixel is light or dark by the middle of
    the two tones around it. The fronts of the white keys are the rows below that are as light,
    column by column, as the white keys are between the black ones.
    """
    height, width = gray.shape
    low, high = _tones(gray, max(8, width // 32))
    told = high - low >= TOLD_APART
    light = gray > (low + high) / 2.0
    changes = ((light[:, 1:] != light[:, :-1]) & told[:, 1:] & told[:, :-1]).sum(axis=1)
    light_share = (light & told).sum(axis=1) / np.maximum(told.sum(axis=1), 1)
    black_band = (told.mean(axis=1) > 0.85) & (changes >= 24) & (light_share > 0.3) & (light_share < 0.8)
    runs = [run for run in _runs(black_band) if run[1] - run[0] >= max(8, 0.04 * height)]
    if not runs:
        return None
    top, end = max(runs, key=lambda run: run[1] - run[0])
    depth = end - top
    tone = np.median(high[top + depth // 4 : top + (3 * depth) // 4], axis=0)  # the white keys, column by column
    fronts = (gray >= 0.6 * tone[None, :]).mean(axis=1) > 0.7
    below = [run for run in _runs(fronts) if run[0] - end <= max(4, 0.01 * height) and run[1] > end]
    if not below:
        return None
    bottom = max(below, key=lambda run: run[1] - run[0])[1]
    if bottom - end < max(4, 0.01 * height) or top < 0.25 * height:
        return None
    return top, end - 1, bottom - 1


def _crossing(profile: Gray, level: Gray, inside: int, outside: int) -> float:
    """Where ``profile`` crosses ``level`` between a column inside a dark run and the column outside it."""
    a, b = float(profile[inside]), float(profile[outside])
    middle = float(level[inside] + level[outside]) / 2.0
    share = 0.5 if b <= a else min(1.0, max(0.0, (middle - a) / (b - a)))
    return inside + (outside - inside) * share


def _filmed_black_keys(band: Gray, period: float) -> list[tuple[float, float]]:
    """Left and right edges, to a fraction of a column, of each black key seen whole in ``band``."""
    profile = np.median(band, axis=0)
    low, high = _tones(profile, max(4, round(period)))
    middle = (low + high) / 2.0
    dark = (profile < middle) & (high - low >= TOLD_APART)
    spans = []
    for start, stop in _runs(dark):
        if start == 0 or stop == len(profile):
            continue
        x0 = _crossing(profile, middle, start, start - 1) + 0.5  # a column's value is that of its middle
        x1 = _crossing(profile, middle, stop - 1, stop) + 0.5
        if 0.3 * period <= x1 - x0 <= 0.95 * period:
            spans.append((x0, x1))
    return spans


# Which side of its boundary a black key sits on, by the white key to its left (C to B): C sharp
# and F sharp to the left, D sharp and A sharp as far to the right, G sharp on the boundary.
_SIDE = (-1, 1, 0, -1, 0, 1, 0)
_OF_THREE = (False, False, False, True, True, True, False)  # ... and whether it is one of the group of three


def _boundary_steps(centres: NDArray[np.float64]) -> NDArray[np.int64] | None:
    """How many white-key boundaries each black key lies beyond the first, from the gaps between them.

    Neighbours in a group are a little over one white key apart; across E|F or B|C, a little
    under two. Counting boundaries from one black key to the next needs no ruler for the whole
    keyboard, only the size of a white key about there: a filmed keyboard's keys are not all
    the same width in the picture (they are wider towards the ends of a wide-angle shot).
    """
    if len(centres) < 5:
        return None
    gaps = np.diff(centres)
    within = float(np.percentile(gaps, 25))  # most gaps are between neighbours in a group
    across = gaps[(gaps > 1.3 * within) & (gaps < 1.8 * within)]
    if len(across) == 0:
        return None
    period = (3 * within + 2 * float(np.median(across))) / 7.0  # an octave: three gaps within, two across
    steps = np.rint(gaps / period).astype(np.int64)
    if (steps < 1).any():
        return None
    return np.concatenate(([0], np.cumsum(steps)))


def measure_filmed_keyboard(clean: Gray, rows: tuple[int, int, int]) -> Keyboard | None:
    """Name every key of a filmed keyboard (``clean``: one grey frame without notes), measured where notes land.

    The keys are measured across the upper part of the black keys: next to the row the notes
    fall onto, and where the player's hands, which reach in from the front, seldom are. The
    boundaries between white keys cannot be seen there; they follow from the black keys, each
    of which sits on one, to one side or the other by its kind. A smooth curve through them
    gives every boundary, wherever the keys are wider or narrower in the picture.
    Returns None when the picture holds no keyboard that can be named.
    """
    top, black_bottom, bottom = rows
    width = clean.shape[1]
    depth = black_bottom - top + 1
    band = clean[top + int(0.12 * depth) : top + max(int(0.12 * depth) + 3, int(0.5 * depth))]
    rough = np.diff(np.flatnonzero(np.diff((np.median(band, axis=0) < np.median(band)).astype(np.int8)) > 0))
    if len(rough) < 5:
        return None
    spans = _filmed_black_keys(band, float(np.median(rough)) / 1.4)  # five black keys to seven white ones
    centres = np.array([(x0 + x1) / 2.0 for x0, x1 in spans])
    steps = _boundary_steps(centres)
    if steps is None:
        return None
    # Which white key is a C: the black keys must all sit on boundaries that carry one, and
    # the boundaries between the first and the last of them that carry one should all show it.
    span = range(int(steps[0]), int(steps[-1]) + 1)
    scores = []
    for offset in range(7):
        agree = sum((n in set(steps.tolist())) == BLACK_AFTER[(n - 1 + offset) % 7] for n in span)
        scores.append(agree / len(span))
    offset = int(np.argmax(scores))
    ranked = sorted(scores, reverse=True)
    if ranked[0] < 0.9 or ranked[0] - ranked[1] < 0.15 or len(span) < 7:
        return None
    kinds = [(int(n) - 1 + offset) % 7 for n in steps]
    seen = np.array([BLACK_AFTER[kind] for kind in kinds])
    if seen.sum() < 5:
        return None
    # centre = curve(boundary number) + side * (how far a black key of a group of two, or of three, sits off).
    degree = 3 if seen.sum() >= 20 else 2 if seen.sum() >= 12 else 1
    middle = float(steps.mean())
    powers = np.stack([(steps - middle) ** d for d in range(degree + 1)], axis=1).astype(np.float64)
    side = np.array([_SIDE[kind] for kind in kinds], dtype=np.float64)
    three = np.array([_OF_THREE[kind] for kind in kinds])
    design = np.column_stack((powers, side * ~three, side * three))
    solved = np.linalg.lstsq(design[seen], centres[seen], rcond=None)[0]
    curve, off_two, off_three = solved[: degree + 1], float(solved[-2]), float(solved[-1])

    def boundary(n: int) -> float:
        return float(sum(c * (n - middle) ** d for d, c in enumerate(curve)))

    period = boundary(int(round(middle)) + 1) - boundary(int(round(middle)))
    if not 8.0 <= period <= width / 7.0:
        return None
    first, last = int(steps[0]), int(steps[-1])
    while boundary(first) > 0 and first > steps[0] - 60:
        first -= 1
    while boundary(last) < width and last < steps[-1] + 60:
        last += 1
    boundaries = {n: boundary(n) for n in range(first, last + 1)}
    if any(boundaries[n + 1] - boundaries[n] < 0.5 * period for n in range(first, last)):
        return None  # the curve folds back beyond the black keys it was drawn through
    black_at = {int(n): span_ for n, span_, ok in zip(steps, spans, seen, strict=True) if ok}
    # A black key that could not be seen (a hand rests over it): where the pattern puts one.
    half = float(np.median([x1 - x0 for x0, x1 in black_at.values()])) / 2.0
    for n in range(first + 1, last):
        kind = (n - 1 + offset) % 7
        if n in black_at or not BLACK_AFTER[kind]:
            continue
        centre = boundaries[n] + _SIDE[kind] * (off_three if _OF_THREE[kind] else off_two)
        if centre - half >= 0 and centre + half <= width:
            black_at[n] = (centre - half, centre + half)

    whites = [j for j in range(first, last) if min(boundaries[j + 1], width) - max(boundaries[j], 0) >= 0.4 * period]
    cs = [j for j in whites if (j + offset) % 7 == 0]
    if not cs:
        return None
    # Every white key of a piano, the lowest an A: the whole keyboard, even with a sliver of an end key cut off.
    certain = len(whites) == FULL_KEYBOARD_WHITES and (whites[0] + offset) % 7 == 5
    if certain:
        middle_c = whites[0] + 23
    else:
        centre_of = (boundaries[whites[0]] + boundaries[whites[-1] + 1]) / 2.0
        middle_c = min(cs, key=lambda j: (round(abs((boundaries[j] + boundaries[j + 1]) / 2.0 - centre_of), 3), j))
    keys: list[Key] = []
    for j in whites:
        along = j - middle_c
        pitch = MIDDLE_C + 12 * (along // 7) + WHITE_SEMITONES[along % 7]
        keys.append(Key(pitch, False, max(0.0, boundaries[j]), min(float(width), boundaries[j + 1])))
        if j + 1 in black_at and j + 1 <= whites[-1]:
            x0, x1 = black_at[j + 1]
            keys.append(Key(pitch + 1, True, x0, x1, boundaries[j + 1]))
    keys.sort(key=lambda key: key.pitch)
    return Keyboard(top, black_bottom, bottom, width, period, tuple(keys), certain)


@dataclass(frozen=True)
class KeyboardView:
    keyboard: Keyboard
    usable: NDArray[np.bool_]  # which of the sampled frames show this keyboard
    clean: NDArray[np.uint8]  # median of the usable frames: no notes, no lit keys
    filmed: bool = False  # found by each stretch's own tones, and measured where the notes land


def _drawn_keyboard(frames: NDArray[np.uint8]) -> KeyboardView:
    gray = to_gray(frames)
    enough = max(3, 0.2 * len(frames))
    for tones in (DRAWN, FILMED):
        found = [keyboard_rows(frame, tones) for frame in gray]
        seen = [rows for rows in found if rows is not None]
        if len(seen) >= enough:
            break
    if len(seen) < enough:
        raise JobError(
            "keyboard_not_found",
            "No piano keyboard could be found in this video. The piano-video option needs a tutorial video with "
            "notes falling onto a keyboard along the bottom of the picture.",
        )
    typical = tuple(int(np.median([rows[i] for rows in seen])) for i in range(3))
    tolerance = max(3, int(0.006 * frames.shape[1]))
    usable = np.array(
        [rows is not None and all(abs(rows[i] - typical[i]) <= tolerance for i in range(3)) for rows in found]
    )
    if usable.sum() < 3:
        raise JobError("keyboard_not_found", "The keyboard in this video moves; its keys could not be read.")
    clean = np.median(frames[usable], axis=0).astype(np.uint8)
    rows = (typical[0], typical[1], typical[2])
    keyboard = measure_keyboard(clean.astype(np.float32).mean(axis=-1), rows, hidden_columns(clean, rows))
    return KeyboardView(keyboard, usable, clean)


SAME_PICTURE = 0.8  # correlation with the keyboard's own picture above which a frame shows the keyboard


def _filmed_keyboard(frames: NDArray[np.uint8]) -> KeyboardView | None:
    """A filmed keyboard in uneven light, from the median of ``frames``; None when there is none."""
    clean = np.median(frames, axis=0).astype(np.uint8)
    rows = filmed_rows(clean.astype(np.float32).mean(axis=-1))
    if rows is None:
        return None
    top, black_bottom, _ = rows
    depth = black_bottom - top + 1
    strip = slice(top + int(0.12 * depth), top + max(int(0.12 * depth) + 3, int(0.5 * depth)))
    # The frames that show this keyboard: across the upper part of its black keys they look like the median does.
    profiles = frames[:, strip].astype(np.float32).mean(axis=(1, 3))
    reference = clean[strip].astype(np.float32).mean(axis=(0, 2))
    centred = profiles - profiles.mean(axis=1, keepdims=True)
    against = reference - reference.mean()
    likeness = (centred * against).sum(axis=1) / (np.linalg.norm(centred, axis=1) * np.linalg.norm(against) + 1e-6)
    usable = likeness >= SAME_PICTURE
    if usable.sum() < max(3, 0.2 * len(frames)):
        return None
    clean = np.median(frames[usable], axis=0).astype(np.uint8)
    gray = clean.astype(np.float32).mean(axis=-1)
    rows = filmed_rows(gray) or rows
    keyboard = measure_filmed_keyboard(gray, rows)
    if keyboard is None:
        return None
    return KeyboardView(keyboard, usable, clean, filmed=True)


def find_keyboard(frames: NDArray[np.uint8]) -> KeyboardView:
    """The keyboard shown in most of ``frames`` ([n, rows, columns, 3])."""
    try:
        return _drawn_keyboard(frames)
    except JobError as error:
        if error.code != "keyboard_not_found":
            raise
        view = _filmed_keyboard(frames)
        if view is None:
            raise
        return view

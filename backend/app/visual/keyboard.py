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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.errors import JobError

BRIGHT = 170  # grey level above which a pixel is a white key front
DARK = 75  # grey level below which a pixel is a black key
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


def keyboard_rows(gray: Gray) -> tuple[int, int, int] | None:
    """(top, black_bottom, bottom) rows of the keyboard in one frame, or None if it shows none."""
    height, width = gray.shape
    bright = gray > BRIGHT
    dark = gray < DARK
    bright_share = bright.mean(axis=1)
    dark_share = dark.mean(axis=1)
    changes = (bright[:, 1:] != bright[:, :-1]).sum(axis=1)
    black_band = (
        (bright_share > 0.3) & (bright_share < 0.8) & (dark_share > 0.2) & (bright_share + dark_share > 0.82)
    ) & (changes >= 24)
    runs = [run for run in _runs(black_band) if run[1] - run[0] >= max(8, 0.04 * height)]
    if not runs:
        return None
    top, end = max(runs, key=lambda run: run[1] - run[0])
    # Note names printed across the black keys break the band for a few rows: join what belongs together.
    for start, stop in _runs(black_band):
        if 0 < start - end <= max(3, 0.2 * (end - top)):
            end = stop
    white_band = bright_share > 0.8
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
    spans = []
    for start, stop in _runs(profile < DARK + 25):
        if 0.3 * period <= stop - start <= 0.95 * period and start > 0 and stop < band.shape[1]:
            spans.append((float(start), float(stop)))
    return spans


def measure_keyboard(clean: Gray, rows: tuple[int, int, int]) -> Keyboard:
    """Name every key of a keyboard picture with no lit keys (``clean`` is one grey frame)."""
    top, black_bottom, bottom = rows
    height, width = clean.shape
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
    checked = [n for n in boundaries if 0.5 * period < boundaries[n] < width - 0.5 * period]
    scores = []
    for offset in range(7):
        agree = sum((n in black_at) == BLACK_AFTER[(n - 1 + offset) % 7] for n in checked)
        scores.append(agree / max(1, len(checked)))
    offset = int(np.argmax(scores))
    ranked = sorted(scores, reverse=True)
    if ranked[0] < 0.9 or ranked[0] - ranked[1] < 0.15 or len(checked) < 7:
        raise failure

    whites = [j for j in range(first, last) if min(boundaries[j + 1], width) - max(boundaries[j], 0) >= 0.4 * period]
    cs = [j for j in whites if (j + offset) % 7 == 0]
    if not cs:
        raise failure
    full = [j for j in whites if boundaries[j] >= -0.1 * period and boundaries[j + 1] <= width + 0.1 * period]
    certain = len(full) == FULL_KEYBOARD_WHITES and (full[0] + offset) % 7 == 5  # starts on A: A0 to C8
    if certain:
        middle_c = full[0] + 23
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


@dataclass(frozen=True)
class KeyboardView:
    keyboard: Keyboard
    usable: NDArray[np.bool_]  # which of the sampled frames show this keyboard
    clean: NDArray[np.uint8]  # median of the usable frames: no notes, no lit keys


def find_keyboard(frames: NDArray[np.uint8]) -> KeyboardView:
    """The keyboard shown in most of ``frames`` ([n, rows, columns, 3])."""
    gray = to_gray(frames)
    found = [keyboard_rows(frame) for frame in gray]
    seen = [rows for rows in found if rows is not None]
    if len(seen) < max(3, 0.2 * len(frames)):
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
    keyboard = measure_keyboard(clean.astype(np.float32).mean(axis=-1), (typical[0], typical[1], typical[2]))
    return KeyboardView(keyboard, usable, clean)

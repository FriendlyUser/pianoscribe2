"""Read a piano video whose keys are small, or filmed, from one picture of its whole roll.

``app.visual.reader`` reads each frame through a thin strip above the keyboard, key by key.
That needs keys wide enough to give every key a few columns of its own, and a picture in
which a bar is the only thing that differs from the empty background. A 480-line video of a
filmed piano has neither: a white key is 16 columns wide, the notes glow, and sparks fly
where they land. But every point of the roll is on screen for a second or two, in fifty
frames or more, at every height above the keyboard. This module puts all of that to use:

1. **Light.** What a block's body looks like is taken from the brightest things that move
   above the keyboard (``block_light``): the channels in which every block is bright. A
   block with a white core and a coloured glow is read in the channel the glow lacks; a
   coloured block in the channel its colour has.
2. **Watching.** One pass over the video follows two thin rows of the picture to time the
   fall of the notes (``watch``), and notes where the player's hands are.
3. **Stacking.** A second pass lays every frame onto one picture of the roll, shifted by how
   far the roll had scrolled (``stack``). Each of four bands of the screen gives a picture
   of its own, and the middle value of the four is kept: a block is in all of them, a spark
   or a caption in one. The result is the roll as the video's maker drew it, clean and far
   sharper in time than a single frame.
4. **Blocks.** On that picture a block is a run of bright columns with sharp ends
   (``read_rows``). Where it lies says which key it is: the lanes follow the filmed keys.
   Blocks that touch side by side are told apart by where the run begins and ends.

The notes come out as ``app.visual.reader`` gives them, and are checked against the audio
in the same way.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.config import Settings
from app.errors import JobError
from app.visual.frames import VideoInfo, read_strips
from app.visual.hands import skin_columns, skin_floor
from app.visual.keyboard import KeyboardView
from app.visual.roll import scroll_speed

# Columns per white key below which a video is read this way (and any filmed keyboard is).
STACK_BELOW = 20.0
GATE = 20  # a pixel shows something when its strongest channel exceeds the still picture by more than this
# The bands of the screen stacked separately, as shares of the height above the keyboard. The
# top of the screen is left out (notes fade in there) and so is the part just above the
# keyboard, where notes flare as they land and captions and watermarks sit.
BANDS = ((0.06, 0.26), (0.26, 0.46), (0.46, 0.66), (0.66, 0.84))
SLITS = (0.3, 0.72)  # the two rows followed to time the fall, as shares of the height above the keyboard
SAME_PICTURE = 0.8  # correlation with the clean keyboard above which a frame shows it

Progress = Callable[[float, str], None]
Bytes = NDArray[np.uint8]
Float = NDArray[np.float32]


@dataclass(frozen=True)
class Light:
    """What tells the body of a block from everything else above the keyboard."""

    channels: tuple[int, ...]  # the channels every block is bright in (none: the strongest channel is read)
    level: float  # how bright a block's body is in them
    still: Bytes  # [rows above the keyboard, columns] strongest channel of the still picture behind the notes

    def read(self, frames: Bytes) -> Bytes:
        """How much each pixel of ``frames`` ([frames, rows, columns, 3], the rows above the keyboard) is a block."""
        strongest = frames.max(axis=-1)
        there = strongest > self.still[None, : frames.shape[1]].astype(np.int16) + GATE
        chosen = frames[..., list(self.channels)].min(axis=-1) if self.channels else strongest
        out: Bytes = np.where(there, chosen, 0).astype(np.uint8)
        return out


def block_light(samples: Bytes, view: KeyboardView) -> Light:
    """How the blocks of this video are lit, from the sampled frames."""
    top = view.keyboard.top
    area = samples[view.usable][:, :top]
    strongest = area.max(axis=-1)
    # A pixel a block crosses is dark most of the time: the lower fifth of what it shows is the still picture.
    still = np.percentile(strongest, 20, axis=0).astype(np.uint8)
    rows = slice(int(BANDS[0][0] * top), int(BANDS[-1][1] * top))
    moving = strongest[:, rows] > still[None, rows].astype(np.int16) + GATE
    if moving.sum() < 500:
        raise JobError(
            "no_notes_found",
            "The keyboard was found, but nothing falls onto it. The piano-video option reads notes that fall "
            "onto the keys.",
        )
    brightest = float(np.percentile(strongest[:, rows][moving], 99))
    bodies = area[:, rows][moving & (strongest[:, rows] >= 0.95 * brightest)].astype(np.float32)
    channels = tuple(c for c in range(3) if np.percentile(bodies[:, c], 10) >= 0.8 * np.percentile(bodies[:, c], 90))
    chosen = bodies[:, list(channels)].min(axis=1) if channels else bodies.max(axis=1)
    return Light(channels, float(np.median(chosen)), still)


def _even(value: float) -> int:
    return int(value) - int(value) % 2


@dataclass
class Watched:
    """What the first pass over the video kept."""

    high: Float  # [frames, columns] the block light in the higher of the two rows followed
    low: Float
    present: NDArray[np.bool_]  # [frames] the keyboard is on screen
    gain: Float  # [frames] brightness relative to the clean picture (fades)
    skin: Bytes  # [frames, columns] rows of each column that show skin, over the front of the keys and just below
    distance: float  # rows between the two rows followed


def watch(
    video: Path,
    info: VideoInfo,
    settings: Settings,
    samples: Bytes,
    view: KeyboardView,
    light: Light,
    progress: Progress | None = None,
) -> Watched:
    """The first pass: two rows of the roll in every frame, whether the keyboard shows, and the hands."""
    keyboard = view.keyboard
    top, black_bottom, bottom = keyboard.top, keyboard.black_bottom, keyboard.bottom
    depth = black_bottom - top + 1
    high, low = _even(SLITS[0] * top), _even(SLITS[1] * top)
    seen_top = _even(top + 0.12 * depth)  # the upper part of the black keys: the keyboard as it is known
    seen_rows = max(2, _even(0.38 * depth))
    # The hands: from the lower part of the black keys (above it a strip of lights may tint everything)
    # to a little below the keyboard, where the backs of the hands are.
    hands_top = _even(top + 0.6 * depth)
    reach = min(info.height - 2, bottom + int(0.3 * (bottom - top)))
    hands_rows = max(2, _even(reach - hands_top))
    strips = [(high, 2), (low, 2), (seen_top, seen_rows), (hands_top, hands_rows)]
    still = light.still
    reference = view.clean[seen_top : seen_top + seen_rows].astype(np.float32).mean(axis=(0, 2))
    whites = reference > np.median(reference)
    against = reference - reference.mean()
    floor = skin_floor(samples[view.usable][:, hands_top : hands_top + hands_rows])

    def slit(rows: Bytes, at: int) -> Float:
        strongest = rows.max(axis=-1)
        there = strongest > still[None, at : at + 2].astype(np.int16) + GATE
        chosen = rows[..., list(light.channels)].min(axis=-1) if light.channels else strongest
        out: Float = np.where(there, chosen, 0).astype(np.float32).mean(axis=1)
        return out

    parts: dict[str, list[NDArray[np.generic]]] = {name: [] for name in ("high", "low", "present", "gain", "skin")}
    expected, done = info.frames + 8, 0
    for chunk in read_strips(video, info, settings, strips):
        parts["high"].append(slit(chunk[:, 0:2], high))
        parts["low"].append(slit(chunk[:, 2:4], low))
        seen = chunk[:, 4 : 4 + seen_rows].astype(np.float32).mean(axis=(1, 3))  # [frames, columns]
        centred = seen - seen.mean(axis=1, keepdims=True)
        likeness = (centred * against).sum(axis=1) / (np.linalg.norm(centred, axis=1) * np.linalg.norm(against) + 1e-6)
        gain = np.median(seen[:, whites] / np.maximum(reference[whites], 1.0), axis=1)
        parts["present"].append((likeness >= SAME_PICTURE) & (gain >= 0.35) & (gain <= 1.25))
        parts["gain"].append(gain.astype(np.float32))
        parts["skin"].append(skin_columns(chunk[:, 4 + seen_rows :], floor))
        done += len(chunk)
        if progress is not None:
            progress(min(1.0, done / expected), "Timing the falling notes")
    if done == 0:
        raise JobError("decode_failed", "No video frames could be read.")
    joined = {name: np.concatenate(values) for name, values in parts.items()}
    return Watched(distance=float(low - high), **joined)


def measure_speed(watched: Watched, info: VideoInfo, top: int) -> float:
    """Rows per frame at which the notes fall."""
    shown = watched.present[:, None]
    speed = scroll_speed(
        watched.high * shown,
        watched.low * shown,
        watched.distance,
        least=top / (12.0 * info.fps),  # the slowest roll worth reading: twelve seconds on screen
        most=top / (0.5 * info.fps),
    )
    if speed is None:
        raise JobError(
            "no_notes_found",
            "No falling notes could be followed in this video. The piano-video option needs notes that fall "
            "onto a keyboard at a steady speed.",
        )
    return speed


@dataclass
class Stack:
    """The whole roll in one picture, one row per pixel of travel."""

    picture: Float  # [rows, columns] the block light, the middle value of the bands
    colour: Bytes  # [rows, columns, 3] mean colour over all the bands
    speed: float  # rows per frame at which the notes fall, as the bands agree on it
    level: float  # the brightness of a block's body on the picture
    rows_per_frame: float  # rows of the picture per frame (the speed the frames were laid down with)
    origin: float  # the row that reaches the keyboard in frame 0
    bands_late: list[float]  # rows by which each band showed the blocks later than the lowest, before alignment

    def frame_of(self, row: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """The frame (and fraction of one) in which row ``row`` of the picture reaches the keyboard."""
        return (np.asarray(row, dtype=np.float64) - self.origin) / self.rows_per_frame


def _later(reference: Float, other: Float, reach: int = 3) -> float:
    """Rows by which ``other`` shows the same blocks later than ``reference`` (both [rows, columns])."""
    a = np.diff(np.nan_to_num(reference[reach:-reach]), axis=0)
    b = np.diff(np.nan_to_num(other), axis=0)
    lags = np.arange(-reach, reach + 1)
    scores = np.array([float((a * b[reach + lag : reach + lag + len(a)]).sum()) for lag in lags])
    best = int(np.argmax(scores))
    if scores[best] <= 0:
        return 0.0
    late = float(lags[best])
    if 0 < best < len(lags) - 1:
        before, at, after = scores[best - 1 : best + 2]
        bend = before - 2 * at + after
        if bend < 0:
            late += float(0.5 * (before - after) / bend)
    return late


def _moved(picture: Float, rows: float) -> Float:
    """``picture`` with what is ``rows`` rows later brought to each row (between rows: in proportion)."""
    whole = int(np.floor(rows))
    share = rows - whole
    count = len(picture)
    index = np.arange(count) + whole
    low = picture[np.clip(index, 0, count - 1)]
    high = picture[np.clip(index + 1, 0, count - 1)]
    out: Float = ((1.0 - share) * low + share * high).astype(np.float32)
    return out


def stack(
    video: Path,
    info: VideoInfo,
    settings: Settings,
    view: KeyboardView,
    light: Light,
    watched: Watched,
    speed: float,
    progress: Progress | None = None,
) -> Stack:
    """The second pass: every frame laid onto one picture of the roll."""
    top = view.keyboard.top
    read_rows_count = _even(top)
    bands = [(int(a * top), int(b * top)) for a, b in BANDS]
    frames = len(watched.present)
    rows = int(np.ceil(frames * speed)) + top + 2
    width = info.width
    totals = np.zeros((len(bands), rows, width), dtype=np.float32)
    counts = np.zeros((len(bands), rows), dtype=np.float32)
    colour = np.zeros((rows, width, 3), dtype=np.float32)
    number = 0
    for chunk in read_strips(video, info, settings, [(0, read_rows_count)]):
        lit = light.read(chunk).astype(np.float32)
        for index in range(len(chunk)):
            frame = number + index
            if frame >= frames or not watched.present[frame]:
                continue
            gain = float(watched.gain[frame])
            scale = 1.0 / gain if gain < 0.97 else 1.0  # a fade dims every pixel alike
            scrolled = int(round(frame * speed))
            for band, (first, last) in enumerate(bands):
                # Screen row y of this frame is roll row (scrolled + top - y): the band's rows, upside down.
                start = scrolled + top - (last - 1)
                totals[band, start : start + (last - first)] += lit[index, first:last][::-1] * scale
                counts[band, start : start + (last - first)] += 1.0
            first, last = bands[0][0], bands[-1][1]
            start = scrolled + top - (last - 1)
            colour[start : start + (last - first)] += chunk[index, first:last][::-1].astype(np.float32) * scale
        number += len(chunk)
        if progress is not None:
            progress(min(1.0, number / (frames + 8)), "Reading the falling notes")
    pictures = totals  # each band's own picture, in place: NaN where the band saw nothing
    for band in range(len(bands)):
        pictures[band] /= np.maximum(counts[band], 1.0)[:, None]
        pictures[band][counts[band] == 0] = np.nan
    # The speed is known to a part in a thousand, which over the height of the screen is a
    # fraction of a row: each band shows the same blocks that fraction earlier or later than
    # the lowest band. That is measured and taken out before the bands are compared.
    late = [_later(pictures[-1], pictures[band]) for band in range(len(bands) - 1)] + [0.0]
    for band, rows_late in enumerate(late):
        if rows_late:
            pictures[band] = _moved(pictures[band], rows_late)
    picture = np.zeros((rows, width), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # rows no band saw
        for first in range(0, rows, 2048):  # a part at a time: the median makes copies
            part = np.nanmedian(pictures[:, first : first + 2048], axis=0)
            picture[first : first + 2048] = np.nan_to_num(part, nan=0.0)
    del pictures, totals
    colour /= np.maximum(counts.sum(axis=0), 1.0)[:, None, None]
    bright = picture[picture >= 0.3 * light.level]
    if bright.size < 200:
        raise JobError(
            "no_notes_found",
            "The keyboard was found, but no falling notes. The piano-video option reads blocks that fall onto "
            "the keys.",
        )
    centres = [(first + last - 1) / 2.0 for first, last in bands]
    apart = centres[-1] - centres[0]
    drift = late[0] / apart if apart else 0.0  # rows too late per screen row further up
    return Stack(
        picture,
        np.clip(np.rint(colour), 0, 255).astype(np.uint8),
        speed=speed / (1.0 - drift),
        level=float(np.percentile(bright, 90)),
        rows_per_frame=speed,
        origin=(top - centres[-1]) * drift,
        bands_late=[round(value, 3) for value in late],
    )


def bright_runs(picture: Float, level: float) -> tuple[NDArray[np.int64], Float, Float]:
    """(row, left edge, right edge) of every run of columns at least ``level`` bright.

    The edges are where the brightness crosses ``level``, to a fraction of a column; column
    ``i`` reaches from ``i`` to ``i + 1``.
    """
    rows, width = picture.shape
    lit = np.zeros((rows, width + 2), dtype=bool)
    lit[:, 1:-1] = picture >= level
    padded = np.pad(picture, ((0, 0), (1, 1)))
    row, first = np.nonzero(
        lit[:, 1:] & ~lit[:, :-1]
    )  # ``first`` indexes ``picture``; padded[:, first] is the column before
    _, after = np.nonzero(~lit[:, 1:] & lit[:, :-1])  # one past the last lit column
    inside, outside = padded[row, first + 1], padded[row, first]
    left = first + 0.5 - (inside - level) / np.maximum(inside - outside, 1e-6)
    inside, outside = padded[row, after], padded[row, after + 1]
    right = after - 0.5 + (inside - level) / np.maximum(inside - outside, 1e-6)
    return row.astype(np.int64), left.astype(np.float32), right.astype(np.float32)


@dataclass(frozen=True)
class Lane:
    """Where the blocks of one key fall."""

    pitch: int
    black: bool
    x0: float  # the block's left edge, in columns
    x1: float

    @property
    def center(self) -> float:
        return (self.x0 + self.x1) / 2.0


@dataclass(frozen=True)
class Lanes:
    lanes: tuple[Lane, ...]  # in pitch order
    white_step: float  # columns from one white key's lane to the next
    first_white: float  # centre of the lane of the lowest white key of the keyboard
    white_width: float  # how wide the block of a white key is
    black_width: float
    seen_white: int  # lanes the measurement rests on
    seen_black: int
    off_keys: float  # the furthest a lane's centre lies from the centre of its key as filmed, in columns


_BLACK_CLASSES = (1, 3, 6, 8, 10)


def _white_pitch(keyboard_whites: list[int], index: int) -> int:
    """The pitch of the white key ``index`` places from the keyboard's lowest (also beyond either end)."""
    if 0 <= index < len(keyboard_whites):
        return keyboard_whites[index]
    scale = (0, 2, 4, 5, 7, 9, 11)
    base = keyboard_whites[0]
    degree = scale.index(base % 12)
    octave, step = divmod(degree + index, 7)
    return base - base % 12 + 12 * octave + scale[step]


def _together(values: Float, weights: Float, period: float) -> complex:
    """Where ``values`` fall within ``period``, added up as directions round a circle (length 1: all at one place)."""
    return complex((weights * np.exp(2j * np.pi * values / period)).sum() / weights.sum())


def find_lanes(picture: Float, level: float, view: KeyboardView) -> Lanes:
    """Where each key's blocks fall, measured on the picture of the roll itself.

    The maker of a video with a filmed keyboard lays an evenly spaced keyboard over the
    picture for the notes to fall onto, and a camera does not film a keyboard evenly: the
    lanes can lie a third of a key off the keys at the ends of the keyboard. So the lanes
    are measured where the blocks are: runs of one width at one place, thousands of them,
    are a lane; the wide lanes, the white keys', lie on an even lattice; the narrow ones
    sit on its boundaries, to one side by their kind. The filmed keys only name the lanes.
    """
    keyboard = view.keyboard
    period = keyboard.white_width
    _, left, right = bright_runs(picture, 0.5 * level)
    width, centre = right - left, (left + right) / 2.0
    usable = (width >= 0.3 * period) & (width <= 1.45 * period)
    failure = JobError(
        "no_notes_found",
        "Blocks fall onto the keyboard of this video, but not in lanes that could be matched to its keys.",
    )
    if usable.sum() < 200:
        raise failure
    # Lanes: places where many runs have their centre.
    bins = np.arange(0.0, picture.shape[1] + 0.25, 0.25)
    counts = np.histogram(centre[usable], bins=bins)[0].astype(np.float32)
    smooth = np.convolve(counts, np.ones(3, dtype=np.float32), mode="same")
    least = max(12.0, 0.002 * float(usable.sum()))
    found: list[tuple[float, float, float]] = []  # centre, width, runs
    peaks = (smooth >= least) & (smooth >= np.roll(smooth, 1)) & (smooth > np.roll(smooth, -1))
    for index in np.flatnonzero(peaks).tolist():
        members = usable & (np.abs(centre - (bins[index] + 0.125)) <= 0.75)
        if members.sum() < least:
            continue
        place = float(np.median(centre[members]))
        if found and place - found[-1][0] < 1.0:
            if members.sum() <= found[-1][2]:
                continue
            found.pop()
        # The rounded ends of a block are narrower runs at the same place: the block's width is that of most runs.
        found.append((place, float(np.percentile(width[members], 60)), float(members.sum())))
    if len(found) < 3:
        raise failure
    places = np.array([f[0] for f in found], dtype=np.float32)
    widths = np.array([f[1] for f in found], dtype=np.float32)
    weight = np.array([f[2] for f in found], dtype=np.float32)
    # The widths blocks have. A run is one block or, where neighbouring keys sound together, several
    # joined into a wider one. The white keys' blocks are the ones of one width whose lanes lie on an
    # even lattice, a white key apart: runs of joined blocks lie wherever their keys happen to be.
    edges = np.arange(0.3 * period, 1.45 * period + 0.5, 0.5)
    spread = np.convolve(np.histogram(widths, bins=edges, weights=weight)[0], np.ones(3), mode="same")
    kinds: list[float] = []  # the widths most runs have, the most common first, well apart from one another
    for peak in np.argsort(-spread, kind="stable").tolist():
        if spread[peak] < 0.03 * spread.max() or len(kinds) == 4:
            break
        if all(abs(edges[peak] + 0.25 - kind) >= 0.2 * period for kind in kinds):
            kinds.append(float(edges[peak] + 0.25))
    tried = np.linspace(0.96 * period, 1.04 * period, 161)

    def evenness(kind: float) -> float:
        """How nearly the lanes of blocks this wide lie on one lattice about a white key apart (1: exactly)."""
        lanes_of = np.flatnonzero(np.abs(widths - kind) <= 0.1 * period)
        if len(lanes_of) < 3:
            return 0.0
        return max(abs(_together(places[lanes_of], weight[lanes_of], step)) for step in tried)

    even = [evenness(kind) for kind in kinds]
    if not kinds or max(even) < 0.8:
        raise failure
    wide_width = kinds[int(np.argmax(even))]
    wide = np.flatnonzero(np.abs(widths - wide_width) <= 0.1 * period)
    narrower = [kind for kind in kinds if kind <= wide_width - 0.2 * period]
    narrow = np.flatnonzero(np.abs(widths - narrower[0]) <= 0.1 * period) if narrower else wide[:0]
    wide_width = float(np.average(widths[wide], weights=weight[wide]))

    # The white lanes' lattice: the step on which their centres fall most nearly at one place, then least squares.
    lengths = [abs(_together(places[wide], weight[wide], step)) for step in tried]
    step = float(tried[int(np.argmax(lengths))])
    for _ in range(3):
        origin = float(np.angle(_together(places[wide], weight[wide], step)) / (2 * np.pi) * step)
        number = np.rint((places[wide] - origin) / step)
        fits = np.abs(places[wide] - (origin + number * step)) <= 0.1 * step
        if fits.sum() < 3 or len(np.unique(number[fits])) < 2:
            raise failure
        slope, intercept = np.polyfit(number[fits], places[wide][fits], 1, w=np.sqrt(weight[wide][fits]))
        step, origin = float(slope), float(intercept)
    number = np.rint((places[wide] - origin) / step).astype(np.int64)
    fits = np.abs(places[wide] - (origin + number * step)) <= 0.1 * step
    if fits.mean() < 0.6:
        raise failure
    # Which key each lane is: the filmed white key nearest to it, for most lanes the same count along.
    whites = keyboard.whites
    filmed = np.array([key.center for key in whites])
    nearest = np.abs(places[wide][fits][:, None] - filmed[None, :]).argmin(axis=1)
    shifts, votes = np.unique(nearest - number[fits], return_counts=True)
    if votes.max() < 0.6 * votes.sum():
        raise failure
    shift = int(shifts[int(np.argmax(votes))])
    first_white = origin - shift * step  # centre of the lane of whites[0]
    white_pitches = [key.pitch for key in whites]

    def boundary(index: int) -> float:
        """The lattice boundary to the right of the white key ``index`` places from the lowest."""
        return first_white + (index + 0.5) * step

    # The black lanes: narrow, beside a boundary that carries a black key.
    sides: dict[int, list[tuple[float, float, float]]] = {}  # pitch class: (offset from the boundary, width, runs)
    for lane in narrow:
        index = int(np.rint((places[lane] - first_white) / step - 0.5))
        offset = float(places[lane] - boundary(index))
        pitch = _white_pitch(white_pitches, index) + 1
        if pitch % 12 in _BLACK_CLASSES and abs(offset) <= 0.3 * step:
            sides.setdefault(pitch % 12, []).append((offset, float(widths[lane]), float(weight[lane])))
    seen_black = sum(len(values) for values in sides.values())
    as_filmed: dict[int, float] = {}
    for kind in _BLACK_CLASSES:
        offsets = [key.center - key.boundary for key in keyboard.blacks if key.pitch % 12 == kind and key.boundary]
        as_filmed[kind] = float(np.median(offsets)) if offsets else 0.0
    measured = {kind: float(np.median([value[0] for value in values])) for kind, values in sides.items()}
    # A kind of black key never played: beside its boundary as filmed, moved as the kinds that were played are.
    apart = float(np.median([measured[kind] - as_filmed[kind] for kind in measured])) if measured else 0.0
    beside = {kind: measured.get(kind, as_filmed[kind] + apart) for kind in _BLACK_CLASSES}
    if seen_black:
        every = [value for values in sides.values() for value in values]
        black_width = float(np.average([v[1] for v in every], weights=[v[2] for v in every]))
    else:
        filmed_black = (
            float(np.median([key.x1 - key.x0 for key in keyboard.blacks])) if keyboard.blacks else 0.6 * period
        )
        black_width = wide_width * filmed_black / period

    lanes: list[Lane] = []
    lowest = int(np.ceil((0.0 - first_white) / step - 0.5))
    highest = int(np.floor((picture.shape[1] - first_white) / step - 0.5)) + 1
    for index in range(lowest, highest + 1):
        pitch = _white_pitch(white_pitches, index)
        middle = float(first_white + index * step)
        if 0.0 < middle < picture.shape[1]:
            lanes.append(Lane(pitch, False, middle - wide_width / 2.0, middle + wide_width / 2.0))
        if (pitch + 1) % 12 in _BLACK_CLASSES:
            middle = float(boundary(index) + beside[(pitch + 1) % 12])
            if middle - black_width / 2.0 >= 0.0 and middle + black_width / 2.0 <= picture.shape[1]:
                lanes.append(Lane(pitch + 1, True, middle - black_width / 2.0, middle + black_width / 2.0))
    lanes.sort(key=lambda lane: lane.pitch)
    by_pitch = {key.pitch: key.center for key in keyboard.keys}
    off = max((abs(lane.center - by_pitch[lane.pitch]) for lane in lanes if lane.pitch in by_pitch), default=0.0)
    return Lanes(
        tuple(lanes), step, first_white, wide_width, black_width, int(fits.sum()), seen_black, round(float(off), 2)
    )


THERE = 0.5  # share of a block's brightness above which a part of a lane is covered
CLEAR = 0.5  # columns kept between a part of a lane that is read and the edge of a block beside it


def _mean(picture: Float, start: float, stop: float) -> Float:
    """The mean of each row of ``picture`` over columns ``start`` to ``stop`` (fractions of a column count in part)."""
    width = picture.shape[1]
    start, stop = max(0.0, start), min(float(width), stop)
    if stop - start <= 1e-6:
        return np.zeros(len(picture), dtype=np.float32)
    first, last = int(np.floor(start)), int(np.ceil(stop))
    weights = np.ones(last - first, dtype=np.float32)
    weights[0] -= start - first
    weights[-1] -= last - stop
    out: Float = (picture[:, first:last] @ weights) / float(weights.sum())
    return out


def _part(start: float, stop: float, *, inner_left: bool, inner_right: bool) -> tuple[float, float]:
    """``start`` to ``stop`` moved in by ``CLEAR`` at the ends that touch another block, less if little is left."""
    clear = min(CLEAR, max(0.0, (stop - start - 1.0) / 2.0))
    return (
        start + clear if inner_left else start + min(clear, 0.25),
        stop - clear if inner_right else stop - min(clear, 0.25),
    )


@dataclass
class Presence:
    """Which keys' blocks each row of the picture shows."""

    on: NDArray[np.bool_]  # [rows, lanes]
    hidden: NDArray[np.bool_]  # [rows, lanes] a black key between two white keys that are both on: it cannot be seen
    own: Float  # [rows, lanes] how covered the part of each lane is that tells its key (share of a block's brightness)
    corners: Float  # [rows, lanes] the more covered of the two corner strips of each lane's block
    unexplained: Float  # [rows] columns lit that no key that is on accounts for
    missing: Float  # [rows] columns of the blocks of keys that are on that are not lit


def read_rows(picture: Float, level: float, lanes: Lanes) -> Presence:
    """Which keys' blocks each row shows.

    A white key's block is the only one that can cover the part of its lane between the
    blocks of its neighbours: lit there, the key is on. A black key's block lies over the
    edges of two white keys' blocks, so each half of it is read where the white key on that
    side does not reach, and trusted only while that white key is off.
    """
    keys = lanes.lanes
    count, rows = len(keys), len(picture)
    by_pitch = {lane.pitch: index for index, lane in enumerate(keys)}
    scaled = picture / level
    own = np.zeros((rows, count), dtype=np.float32)
    left_part = np.zeros((rows, count), dtype=np.float32)
    right_part = np.zeros((rows, count), dtype=np.float32)
    has_left = np.zeros(count, dtype=bool)
    has_right = np.zeros(count, dtype=bool)
    corner_left = np.zeros((rows, count), dtype=np.float32)
    corner_right = np.zeros((rows, count), dtype=np.float32)
    white_left = np.full(count, -1, dtype=np.int64)  # black keys: the lane of the white key on either side
    white_right = np.full(count, -1, dtype=np.int64)
    for index, lane in enumerate(keys):
        strip = max(1.5, 0.15 * (lane.x1 - lane.x0))
        corner_left[:, index] = _mean(scaled, lane.x0 + 0.5, lane.x0 + 0.5 + strip)
        corner_right[:, index] = _mean(scaled, lane.x1 - 0.5 - strip, lane.x1 - 0.5)
        if lane.black:
            below, above = by_pitch.get(lane.pitch - 1, -1), by_pitch.get(lane.pitch + 1, -1)
            white_left[index], white_right[index] = below, above
            # The half no block of the white key on the other side reaches.
            stop = min(lane.x1, keys[above].x0) if above >= 0 else lane.x1
            if stop - lane.x0 >= 1.0:
                a, b = _part(lane.x0, stop, inner_left=False, inner_right=above >= 0)
                left_part[:, index], has_left[index] = _mean(scaled, a, b), True
            start = max(lane.x0, keys[below].x1) if below >= 0 else lane.x0
            if lane.x1 - start >= 1.0:
                a, b = _part(start, lane.x1, inner_left=below >= 0, inner_right=False)
                right_part[:, index], has_right[index] = _mean(scaled, a, b), True
            own[:, index] = _mean(scaled, lane.x0 + 0.25, lane.x1 - 0.25)
        else:
            # The widest stretch of the lane that no other key's block covers.
            cuts = sorted(
                (max(other.x0, lane.x0), min(other.x1, lane.x1))
                for other in keys[max(0, index - 3) : index + 4]
                if other is not lane and other.x1 > lane.x0 and other.x0 < lane.x1
            )
            free, at = [], lane.x0
            for a, b in cuts:
                if a > at:
                    free.append((at, a))
                at = max(at, b)
            if at < lane.x1:
                free.append((at, lane.x1))
            if free:
                a, b = max(free, key=lambda span: span[1] - span[0])
                a, b = _part(a, b, inner_left=a > lane.x0, inner_right=b < lane.x1)
                own[:, index] = _mean(scaled, a, b)
    black = np.array([lane.black for lane in keys])
    on = np.zeros((rows, count), dtype=bool)
    on[:, ~black] = own[:, ~black] >= THERE
    hidden = np.zeros((rows, count), dtype=bool)
    for index in np.flatnonzero(black).tolist():
        below, above = int(white_left[index]), int(white_right[index])
        left_on = on[:, below] if below >= 0 else np.zeros(rows, dtype=bool)
        right_on = on[:, above] if above >= 0 else np.zeros(rows, dtype=bool)
        left_says = (left_part[:, index] >= THERE) & ~left_on if has_left[index] else np.zeros(rows, dtype=bool)
        right_says = (right_part[:, index] >= THERE) & ~right_on if has_right[index] else np.zeros(rows, dtype=bool)
        alone = (own[:, index] >= THERE) & ~left_on & ~right_on
        on[:, index] = np.where(left_on | right_on, left_says | right_says, alone)
        hidden[:, index] = left_on & right_on
        # What tells this key, row by row: its whole block, or the half its sounding neighbour leaves in view.
        own[:, index] = np.where(
            left_on & ~right_on, right_part[:, index], np.where(right_on & ~left_on, left_part[:, index], own[:, index])
        )
    # Where two blocks on one key meet end to end, their rounded corners leave both corner strips
    # of the lane dark for a row or two. (The more covered of the two is kept: a blemish dims one.)
    # A corner strip under the block of a neighbour that is on says nothing about this key's block.
    corners = np.maximum(corner_left, corner_right)
    for index, lane in enumerate(keys):
        covered_left = np.zeros(rows, dtype=bool)
        covered_right = np.zeros(rows, dtype=bool)
        for other_index in range(max(0, index - 3), min(count, index + 4)):
            other = keys[other_index]
            if other_index == index:
                continue
            if other.x0 < lane.x0 + 0.5 < other.x1:
                covered_left |= on[:, other_index]
            if other.x0 < lane.x1 - 0.5 < other.x1:
                covered_right |= on[:, other_index]
        corners[:, index] = np.where(
            covered_left & covered_right,
            1.0,
            np.where(
                covered_left, corner_right[:, index], np.where(covered_right, corner_left[:, index], corners[:, index])
            ),
        )
    # How well the keys that are on account for what is lit.
    lit = picture >= THERE * level
    told = np.zeros_like(lit)
    for index, lane in enumerate(keys):
        a, b = max(0, int(np.floor(lane.x0 + 0.5))), min(picture.shape[1], int(np.ceil(lane.x1 - 0.5)))
        told[:, a:b] |= on[:, index : index + 1]
    reach = np.zeros_like(lit)
    for index, lane in enumerate(keys):  # a column more either way: an edge falls between columns
        a, b = max(0, int(np.floor(lane.x0 - 1.0))), min(picture.shape[1], int(np.ceil(lane.x1 + 1.0)))
        reach[:, a:b] |= on[:, index : index + 1]
    inner = np.zeros_like(lit)
    for index, lane in enumerate(keys):
        a, b = max(0, int(np.ceil(lane.x0 + 1.5))), min(picture.shape[1], int(np.floor(lane.x1 - 1.5)))
        if b > a:
            inner[:, a:b] |= on[:, index : index + 1]
    unexplained = (lit & ~reach).sum(axis=1).astype(np.float32)
    missing = (inner & ~lit).sum(axis=1).astype(np.float32)
    del told
    return Presence(on, hidden, own, corners, unexplained, missing)


PINCH = 0.85  # both corners of a lane this much darker than on either side: two blocks meet there end to end
FAINT_PINCH = 0.88  # ... a pinch that shallow needs another sign that a note begins there


@dataclass(frozen=True)
class Block:
    """One falling note as measured on the picture of the roll."""

    lane: int  # index into Lanes.lanes
    start: float  # the row its lower end is at: the moment the note is struck
    end: float  # the row just past its upper end: the moment it is released
    joined: bool = False  # separated from the block before it on its key by a pinch, not by a gap
    cut_short: bool = False  # a black key that went out of sight between two white keys that were both on


def _runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def _edge(values: Float, inside: int, outside: int) -> float:
    """The row, between ``inside`` (covered) and ``outside`` (not), at which ``values`` passes ``THERE``."""
    if outside < 0 or outside >= len(values):
        return float(inside)
    covered, bare = float(values[inside]), float(values[outside])
    if covered <= bare:
        return float(inside)
    share = min(1.0, max(0.0, (covered - THERE) / (covered - bare)))
    return inside + (outside - inside) * share


def pinches(corners: Float, start: int, end: int, reach: int) -> list[tuple[int, float]]:
    """(row, depth) of the places between ``start`` and ``end`` where a lane's corner strips dip and recover.

    ``reach``: about how many rows a block's rounded corner takes. The depth is the corner
    strips' cover at the dip as a share of the lesser of what it is on either side.
    """
    found: list[tuple[int, float]] = []
    for row in range(start + 2 * reach, end - 2 * reach):
        if corners[row] > corners[row - reach : row + reach + 1].min():
            continue
        before = corners[max(start + reach, row - 3 * reach - 1) : row - reach + 1]
        after = corners[row + reach : min(end - reach, row + 3 * reach + 2)]
        if len(before) == 0 or len(after) == 0:
            continue
        depth = float(corners[row]) / max(1e-6, float(min(before.max(), after.max())))
        if depth <= FAINT_PINCH and (not found or row - found[-1][0] > reach):
            found.append((row, depth))
    return found


def read_blocks(
    presence: Presence, lanes: Lanes, *, least_rows: int = 3
) -> tuple[list[Block], list[tuple[int, int, float]], int]:
    """(blocks, faint pinches not cut on as (block number, row, depth), runs too short to be notes)."""
    reach = max(2, int(round(0.18 * lanes.white_step)))
    blocks: list[Block] = []
    faint: list[tuple[int, int, float]] = []
    slivers = 0
    for lane in range(len(lanes.lanes)):
        on, own = presence.on[:, lane], presence.own[:, lane]
        for first, last in _runs(on):
            if last - first < least_rows:
                slivers += 1
                continue
            # A black key last seen as the white keys either side of it both came on: its end is not seen.
            unseen = last < len(on) and bool(presence.hidden[last, lane])
            found = pinches(presence.corners[:, lane], first, last, reach)
            edges = [_edge(own, first, first - 1), *[float(row) for row, depth in found if depth <= PINCH]]
            edges.append(_edge(own, last - 1, last))
            base = len(blocks)
            for piece, (a, b) in enumerate(zip(edges, edges[1:], strict=False)):
                blocks.append(Block(lane, a, b, joined=piece > 0, cut_short=unseen and piece == len(edges) - 2))
            for row, depth in found:
                if depth > PINCH:
                    piece = sum(1 for edge in edges[1:-1] if edge <= row)
                    faint.append((base + piece, row, depth))
    return blocks, faint, slivers

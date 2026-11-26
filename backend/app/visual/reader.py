"""Piano-tutorial video -> the notes it shows.

``read_video`` is the whole reader: find the keyboard, read the falling bars back into
a piano roll, turn the bars into notes with a hand each, and check the result two
independent ways before handing it on:

* **The keys.** The video lights each key while its note sounds. That is a second
  reading of the same notes from a different part of the picture, at frame precision.
  Every bar is checked against it, it sets the exact moment a bar counts as struck,
  and it supplies the few notes that were already past the reading strip when the
  roll first appeared.
* **The audio.** The notes are laid over the spectrum of the recording to fix the
  octave when the keyboard shown is not a full one, and timed against its attacks.

Nothing here is a model: the notes are measured, not estimated, and the measurements
that came out are reported with the notes (``VisualReading.facts``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.config import Settings
from app.errors import JobError
from app.logging import get_logger
from app.visual import audio as sound
from app.visual.frames import VideoInfo, probe_video, read_strips, sample_frames
from app.visual.hands import LEFT, RIGHT, UNKNOWN, assign_hands, split_by_pitch
from app.visual.keyboard import Keyboard, KeyboardView, find_keyboard
from app.visual.roll import (
    FAINT,
    SOLID,
    Bar,
    Roll,
    Zones,
    build_zones,
    enclosed,
    frame_shifts,
    read_bars,
    reduce_strip,
    scroll_speed,
    unroll,
)

log = get_logger(__name__)

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
KEY_ROWS = 4
SPEED_ROWS = 2
PIANO_LOWEST, PIANO_HIGHEST = 21, 108
Progress = Callable[[float, str], None]


def pitch_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


@dataclass(frozen=True)
class VisualNote:
    start: float  # seconds
    end: float
    pitch: int
    hand: str  # "right", "left" or "unknown"
    source: str = "bars"  # "bars": read from a falling bar; "keys": from a lit key only


@dataclass
class VisualReading:
    notes: list[VisualNote]
    facts: dict[str, object]

    def save(self, path: Path) -> None:
        payload = {"facts": self.facts, "notes": [asdict(note) for note in self.notes]}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> VisualReading:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls([VisualNote(**note) for note in payload["notes"]], payload["facts"])


@dataclass(frozen=True)
class Layout:
    """Which rows of each frame are read."""

    band_top: int
    band_rows: int
    lead: int  # rows from the band's last row down to the row where notes land
    speed_top: int
    key_top: int
    above: int  # rows read above the band, with it, to follow how far the picture moves each frame

    @property
    def strips(self) -> list[tuple[int, int]]:
        return [
            (self.band_top - self.above, self.above + self.band_rows),
            (self.speed_top, SPEED_ROWS),
            (self.key_top, KEY_ROWS),
        ]

    @property
    def speed_distance(self) -> float:
        """Rows between the middle of the speed strip and the middle of the band's last two rows."""
        return (self.band_top + self.band_rows - 1.5) - (self.speed_top + 0.5)


def plan_layout(keyboard: Keyboard, fps: float, *, band_rows: int | None = None) -> Layout:
    roll_height = keyboard.top
    period = keyboard.white_width
    # High enough to clear the flare drawn where a note lands, low enough to stay near it. A flare
    # bursts at the moment a note lands: in one video it reaches nearly two white keys up, over the
    # neighbouring keys.
    lead = max(round(2.4 * period), round(0.1 * roll_height))
    if band_rows is None:
        # Tall enough that no row of the roll slips between two frames unless the
        # whole height is crossed in under a second.
        band_rows = int(np.ceil(roll_height / fps)) + 2
    band_rows = max(6, band_rows + band_rows % 2)
    band_top = keyboard.top - lead - band_rows + 1
    if band_top % 2:
        band_top -= 1
        lead += 1
    speed_top = band_top - round(0.38 * roll_height)
    speed_top -= speed_top % 2
    depth = keyboard.black_bottom - keyboard.top + 1
    key_top = keyboard.top + round(0.35 * depth)
    key_top -= key_top % 2
    above = 4 * band_rows
    if speed_top < 2 or band_top - above < speed_top + 8:
        raise JobError("keyboard_not_found", "The picture leaves too little room above the keyboard to read notes.")
    return Layout(band_top, band_rows, lead, speed_top, key_top, above)


def roll_background(samples: NDArray[np.uint8], view: KeyboardView, rows: range) -> NDArray[np.int16]:
    """The empty picture behind the falling notes in ``rows``: [rows, columns, 3].

    The median of the sampled frames in each row, unless a key is covered so much of the
    time that this median is a note: then the median over the whole height is used.
    """
    usable = samples[view.usable]
    own: NDArray[np.int16] = np.median(usable[:, rows.start : rows.stop], axis=0).astype(np.int16)
    top = view.keyboard.top
    spread = np.unique(np.linspace(0.15 * top, 0.85 * top, 24).astype(int))
    pooled = np.median(usable[:, spread].reshape(-1, usable.shape[2], 3), axis=0).astype(np.int16)
    suspicious = (np.abs(own - pooled[None]) > 40).any(axis=-1)
    own[suspicious] = np.broadcast_to(pooled[None], own.shape)[suspicious]
    return own


MOVING = 0.05  # share of a typical frame that changes far from any bar, above which a film plays behind the notes
BAR_BRIGHT = 128  # as bright as the dimmest bar colour can be: what counts as "near a bar" when looking for a film


def _grow(mask: NDArray[np.bool_], reach: int) -> NDArray[np.bool_]:
    """``mask`` ([frames, rows, columns]) widened by ``reach`` pixels up, down and to either side."""
    grown = mask
    for axis in (1, 2):
        size = grown.shape[axis]
        total = np.cumsum(grown, axis=axis, dtype=np.int32)
        total = np.concatenate((np.zeros_like(np.take(total, [0], axis=axis)), total), axis=axis)
        upto = np.minimum(np.arange(size) + reach + 1, size)
        since = np.maximum(np.arange(size) - reach, 0)
        grown = (np.take(total, upto, axis=axis) - np.take(total, since, axis=axis)) > 0
    return grown


def _split_level(levels: NDArray[np.uint8]) -> int:
    """The level that best splits ``levels`` into a darker and a brighter group (Otsu's threshold)."""
    counts = np.bincount(levels.ravel(), minlength=256).astype(np.float64)
    values = np.arange(256, dtype=np.float64)
    below = np.cumsum(counts)
    above = below[-1] - below
    mean_below = np.cumsum(counts * values) / np.maximum(below, 1.0)
    mean_above = ((counts * values).sum() - np.cumsum(counts * values)) / np.maximum(above, 1.0)
    return int(np.argmax(below * above * (mean_below - mean_above) ** 2)) + 1


def backdrop_floor(samples: NDArray[np.uint8], view: KeyboardView) -> int | None:
    """How bright the bars are, when a film plays behind the notes; None when the picture there holds still.

    Behind the notes of most tutorials is a still picture, nearly black: whatever differs from it
    is a bar, its outline or its name. Some videos play a film behind the notes instead, dimmed
    so that the bars stand out. Then a typical frame differs from the still picture (the median
    of the sampled frames) in large areas far from anything as bright as a bar, which a still
    picture never does; and only brightness tells a bar from the film. The level returned is
    the one that best splits the brightness (the strongest channel) of everything that changed.
    """
    keyboard = view.keyboard
    rows = np.arange(int(0.15 * keyboard.top), int(0.85 * keyboard.top), 2)
    picture = samples[view.usable][:, rows, ::2]  # every other row and column is plenty
    still = np.median(picture, axis=0).astype(np.int16)
    level = picture.max(axis=-1)
    changed = np.abs(picture.astype(np.int16) - still[None]).max(axis=-1) > FAINT
    near = _grow(level >= BAR_BRIGHT, max(1, round(0.125 * keyboard.white_width)))  # a quarter of a white key
    away = (changed & ~near).mean(axis=(1, 2))
    if len(away) == 0 or float(np.median(away)) < MOVING or not changed.any():
        return None
    return _split_level(level[changed])


@dataclass
class Readings:
    """Everything kept from the pass over all frames."""

    solid: NDArray[np.uint8]  # [frames, band rows, zones], see app.visual.roll.StripStats
    grey: NDArray[np.uint8]  # [frames, band rows, zones]
    cover: NDArray[np.uint8]  # [frames, band rows, coloured zones]
    colour: NDArray[np.uint8]  # [frames, band rows, coloured zones, 3]
    spread: NDArray[np.uint8]  # [frames, band rows, coloured zones]
    upper: NDArray[np.float32]  # [frames, zones] cover in the speed strip
    lit: NDArray[np.bool_]  # [frames, keys]
    present: NDArray[np.bool_]  # [frames] the keyboard is on screen
    moved: NDArray[np.float32]  # [frames] rows the picture moved since the frame before (NaN: cannot be told)
    gain: NDArray[np.float32]  # [frames] brightness relative to the clean picture (fades)


def _key_profile(zones: Zones, keyboard: Keyboard) -> NDArray[np.int64]:
    """One zone per key to recognise the keyboard by: its key front or body in the key strip."""
    chosen = []
    for index, key in enumerate(keyboard.keys):
        if not key.black:
            chosen.append(int(zones.core[index]))
        else:
            left, right = int(zones.left[index]), int(zones.right[index])
            chosen.append(max((left, right), key=lambda z: zones.x1[z] - zones.x0[z] if z >= 0 else -1))
    return np.array(chosen, dtype=np.int64)


def read_all_frames(
    video: Path,
    info: VideoInfo,
    settings: Settings,
    samples: NDArray[np.uint8],
    view: KeyboardView,
    zones: Zones,
    layout: Layout,
    progress: Progress | None = None,
    *,
    floor: int | None = None,
) -> Readings:
    """One pass over every frame. ``floor``: how bright a bar is, when a film plays behind the notes."""
    keyboard = view.keyboard
    band = range(layout.band_top - layout.above, layout.band_top + layout.band_rows)
    speed = range(layout.speed_top, layout.speed_top + SPEED_ROWS)
    keys = range(layout.key_top, layout.key_top + KEY_ROWS)
    background = np.concatenate(
        [
            roll_background(samples, view, band),
            roll_background(samples, view, speed),
            view.clean[keys.start : keys.stop].astype(np.int16),
        ]
    )
    tall = layout.above + layout.band_rows
    key_rows = slice(tall + SPEED_ROWS, tall + SPEED_ROWS + KEY_ROWS)
    most = min(tall // 2, 3 * layout.band_rows)  # the furthest the picture can be followed from frame to frame
    last_mask: NDArray[np.bool_] | None = None
    # In front of a film: the widest dark stroke inside a bar (its outline, the note name printed
    # on it), side to side and up and down. The film between two separate bars is wider than that.
    across, along = max(3, round(0.3 * keyboard.white_width)), max(2, round(0.2 * keyboard.white_width))

    profile = _key_profile(zones, keyboard)
    usable = profile >= 0
    px0, px1 = zones.x0[profile[usable]], zones.x1[profile[usable]]
    is_white = np.array([not key.black for key in keyboard.keys])[usable]
    clean_grey = view.clean[keys.start : keys.stop].astype(np.float32).mean(axis=(0, 2))
    total = np.concatenate(([0.0], np.cumsum(clean_grey)))
    clean_profile = (total[px1] - total[px0]) / (px1 - px0)

    expected = info.frames + 8
    parts: dict[str, list[NDArray[np.generic]]] = {
        name: [] for name in ("solid", "grey", "cover", "colour", "spread", "upper", "lit", "present", "gain", "moved")
    }
    done = 0
    for chunk in read_strips(video, info, settings, layout.strips):
        frames = len(chunk)
        grey = chunk[:, key_rows].astype(np.float32).mean(axis=(1, 3))  # [frames, columns]
        sums = np.concatenate((np.zeros((frames, 1), dtype=np.float32), np.cumsum(grey, axis=1)), axis=1)
        seen = (sums[:, px1] - sums[:, px0]) / (px1 - px0)  # [frames, keys]
        # A fade scales every pixel alike; a lit key changes one. The median over the white keys is the fade.
        gain = np.median(seen[:, is_white] / np.maximum(clean_profile[is_white], 1.0), axis=1)
        centred = seen - seen.mean(axis=1, keepdims=True)
        reference = clean_profile - clean_profile.mean()
        likeness = (centred * reference).sum(axis=1) / (
            np.linalg.norm(centred, axis=1) * np.linalg.norm(reference) + 1e-6
        )
        present = (likeness >= 0.85) & (gain >= 0.35) & (gain <= 1.25)
        faded = np.flatnonzero(present & (gain < 0.97))
        if len(faded):
            chunk = chunk.copy()
            scaled = chunk[faded].astype(np.float32) / gain[faded, None, None, None]
            chunk[faded] = np.clip(scaled, 0, 255).astype(np.uint8)
        # How far the picture moved since the frame before, from the tall strip (every sixth column is plenty).
        thin = chunk[:, :tall, ::6].astype(np.int16) - background[None, :tall, ::6]
        masks = (np.abs(thin) > SOLID).any(axis=-1)
        ignore = covered = None
        if floor is not None:
            # Behind the notes, a pixel dimmer than a bar is the film; the keys are read as they are.
            dim = chunk[:, : tall + SPEED_ROWS].max(axis=-1) < floor
            masks &= ~dim[:, :tall, ::6]
            # The band and, for the gaps up and down, the rows just above it: a gap reaching further
            # up is longer than ``along`` and stays open whatever lies there.
            context = min(layout.above, along)
            inside = enclosed(dim[:, layout.above - context : tall], across, along)[:, context:]
            ignore = np.zeros(chunk.shape[:3], dtype=bool)[:, layout.above :]
            ignore[:, : layout.band_rows + SPEED_ROWS] = dim[:, layout.above :]
            covered = np.zeros_like(ignore)
            covered[:, : layout.band_rows] = inside
        joined = masks if last_mask is None else np.concatenate((last_mask[None], masks))
        moved = frame_shifts(joined, most)
        parts["moved"].append(moved if last_mask is not None else np.concatenate(([np.nan], moved)).astype(np.float32))
        last_mask = masks[-1]
        stats = reduce_strip(
            chunk[:, layout.above :], background[layout.above :], zones, ignore=ignore, covered=covered
        )
        band_rows_cut = slice(0, layout.band_rows)
        speed_rows_cut = slice(layout.band_rows, layout.band_rows + SPEED_ROWS)
        key_rows_cut = slice(layout.band_rows + SPEED_ROWS, layout.band_rows + SPEED_ROWS + KEY_ROWS)
        key_cover = stats.solid[:, key_rows_cut].mean(axis=1)  # [frames, zones]
        lit = np.zeros((frames, len(keyboard.keys)), dtype=bool)
        for index, key in enumerate(keyboard.keys):
            if not zones.readable[index]:
                continue
            if key.black:
                both = [z for z in (int(zones.left[index]), int(zones.right[index])) if z >= 0]
                widths = np.array([zones.x1[z] - zones.x0[z] for z in both], dtype=np.float32)
                lit[:, index] = (key_cover[:, both] * widths).sum(axis=1) / widths.sum() >= 0.5
            else:
                lit[:, index] = key_cover[:, zones.core[index]] >= 0.5
        lit &= present[:, None]
        parts["solid"].append(np.rint(stats.solid[:, band_rows_cut] * 255).astype(np.uint8))
        parts["grey"].append(np.rint(stats.grey[:, band_rows_cut]).astype(np.uint8))
        parts["cover"].append(np.rint(stats.cover[:, band_rows_cut] * 255).astype(np.uint8))
        parts["colour"].append(np.rint(stats.colour[:, band_rows_cut]).astype(np.uint8))
        parts["spread"].append(np.clip(np.rint(stats.spread[:, band_rows_cut]), 0, 255).astype(np.uint8))
        parts["upper"].append(stats.solid[:, speed_rows_cut].mean(axis=1))
        parts["lit"].append(lit)
        parts["present"].append(present)
        parts["gain"].append(gain.astype(np.float32))
        done += frames
        if progress is not None:
            progress(min(1.0, done / expected), "Reading the falling notes")
    if done == 0:
        raise JobError("decode_failed", "No video frames could be read.")
    joined = {name: np.concatenate(values) for name, values in parts.items()}
    return Readings(**joined)


def measure_speed(readings: Readings, zones: Zones, layout: Layout, info: VideoInfo, roll_height: int) -> float:
    """Rows per frame at which the notes fall."""
    cores = zones.core[zones.core >= 0]
    lower = readings.solid[:, layout.band_rows - 2 :, cores].astype(np.float32).mean(axis=1) / 255.0
    upper = readings.upper[:, cores]
    lower = lower * readings.present[:, None]
    upper = upper * readings.present[:, None]
    speed = scroll_speed(
        upper,
        lower,
        layout.speed_distance,
        least=roll_height / (12.0 * info.fps),  # the slowest roll worth reading: twelve seconds on screen
        most=roll_height / (0.5 * info.fps),
    )
    if speed is None:
        raise JobError(
            "no_notes_found",
            "No falling notes could be followed in this video. The piano-video option needs notes that fall "
            "onto a keyboard at a steady speed.",
        )
    return speed


UNEVEN = 0.75  # rows: a picture that moves less evenly than this from frame to frame is a screen recording


def scroll_positions(
    readings: Readings, speed: float, fps: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """(shown, even, steady): how far the roll had scrolled in each frame.

    A video rendered from the notes moves the same distance every frame: ``speed`` times the
    frame number says it all, exactly. A screen recording repeats and skips frames, so the
    distance moved is taken as measured, frame by frame. Its roll still runs evenly with
    the sound, which a running average of the measured positions recovers.
    """
    count = len(readings.moved)
    measured = readings.moved[readings.present & ~np.isnan(readings.moved)]
    line = speed * np.arange(count, dtype=np.float64)
    if len(measured) < 30:
        return line, line, True
    spread = float(np.median(np.abs(measured - np.median(measured)))) * 1.4826
    if spread < UNEVEN and abs(float(np.median(measured)) - speed) < 0.2 * speed:
        return line, line, True
    moved = np.where(np.isnan(readings.moved) | ~readings.present, speed, readings.moved).astype(np.float64)
    shown = np.cumsum(moved)
    window = max(9, int(round(fps)) | 1)
    padded = np.concatenate(
        (
            shown[0] - moved[0] * np.arange(window // 2, 0, -1),
            shown,
            shown[-1] + moved[-1] * np.arange(1, window // 2 + 1),
        )
    )
    even = np.convolve(padded, np.ones(window) / window, mode="valid")
    return shown, even, False


def unroll_readings(readings: Readings, shown: NDArray[np.float64], even: NDArray[np.float64], layout: Layout) -> Roll:
    frames = np.flatnonzero(readings.present)
    if len(frames) == 0:
        raise JobError("keyboard_not_found", "The keyboard is never on screen long enough to read notes.")
    return unroll(
        readings.solid[frames],
        readings.grey[frames],
        readings.cover[frames],
        readings.colour[frames],
        readings.spread[frames],
        frames,
        shown,
        even,
        float(layout.lead),
    )


def _starts(mask: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Indices where a run of True begins."""
    return np.flatnonzero(mask & ~np.concatenate(([False], mask[:-1])))


def _runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


KEYS_AGREE = 0.5  # share of bars whose key must be seen lit for the keys to count as lighting up


@dataclass
class KeyCheck:
    shift: float  # frames to add to a bar's time so that it is struck when its key lights
    onsets_used: int
    confirmed: NDArray[np.bool_]  # per bar: the key was lit while the bar says it sounds
    unexplained: list[tuple[int, int, int]]  # (key, first frame, last frame + 1) lit with no bar
    keys_light: bool


def check_against_keys(bars: list[Bar], roll: Roll, readings: Readings) -> KeyCheck:
    """Compare every bar with the lighting of its key."""
    lit = readings.lit
    frames = len(lit)
    if lit.sum() < 10:
        return KeyCheck(0.0, 0, np.zeros(len(bars), dtype=bool), [], keys_light=False)
    start = np.array([float(roll.frame_of(bar.start)) for bar in bars])
    end = np.array([float(roll.frame_of(bar.end)) for bar in bars])
    # The moment a key lights, against the moment its bar reaches the landing row.
    differences = []
    last_end: dict[int, float] = {}
    for i, bar in enumerate(bars):
        fresh = start[i] - last_end.get(bar.key, -1e9) >= 4.0  # the key had time to go dark
        last_end[bar.key] = end[i]
        if not fresh or bar.joined:
            continue
        low, high = int(np.floor(start[i])) - 10, int(np.ceil(start[i])) + 10
        if low < 1 or high >= frames or not readings.present[low:high].all():
            continue
        begins = _starts(lit[low:high, bar.key])
        if len(begins) == 1:
            differences.append(low + int(begins[0]) - start[i])
    shift = 0.0
    if len(differences) >= 12:
        spread = float(np.subtract(*np.percentile(differences, [75, 25])))
        if spread <= 2.5:
            # A key seen lit in frame j lit up between frames j-1 and j: half a frame earlier on average.
            shift = float(np.median(differences)) - 0.5
    confirmed = np.zeros(len(bars), dtype=bool)
    explained = np.zeros_like(lit)
    for i, bar in enumerate(bars):
        a, b = start[i] + shift, end[i] + shift
        # A key is lit for most of a long note; of a short one it may show only a frame or two,
        # since the light comes on and goes off over a few frames.
        inner_a, inner_b = int(np.ceil(a)), int(np.floor(b))
        window = lit[max(0, inner_a) : min(frames, inner_b), bar.key]
        if len(window) >= 8:
            confirmed[i] = float(window.mean()) >= 0.5
        else:
            around = lit[max(0, int(np.floor(a)) - 1) : min(frames, int(np.ceil(b)) + 2), bar.key]
            confirmed[i] = bool(around.any())
        explained[max(0, int(np.floor(a)) - 2) : min(frames, int(np.ceil(b)) + 3), bar.key] = True
    if confirmed.mean() < KEYS_AGREE:
        # Most bars land on a key that stays as it was: the keys do not light, and what changed on
        # them was something else (the player's hands over a filmed keyboard). They say nothing.
        return KeyCheck(0.0, 0, np.zeros(len(bars), dtype=bool), [], keys_light=False)
    unexplained = []
    for key in range(lit.shape[1]):
        for a, b in _runs(lit[:, key] & ~explained[:, key]):
            if b - a >= 3:
                unexplained.append((key, a, b))
    return KeyCheck(shift, len(differences), confirmed, unexplained, keys_light=True)


def read_video(
    video: Path,
    settings: Settings,
    *,
    audio: tuple[NDArray[np.float32], int] | None = None,
    progress: Progress | None = None,
) -> VisualReading:
    """Read the notes a piano-tutorial video shows. ``audio`` is its decoded sound (samples, rate)."""

    def report(fraction: float, message: str) -> None:
        if progress is not None:
            progress(fraction, message)

    report(0.0, "Finding the keyboard")
    info = probe_video(video, settings)
    samples = sample_frames(video, info, settings, count=settings.visual_sample_frames)
    view = find_keyboard(samples)
    keyboard = view.keyboard
    zones = build_zones(keyboard)
    layout = plan_layout(keyboard, info.fps)
    floor = backdrop_floor(samples, view)
    readings = read_all_frames(
        video, info, settings, samples, view, zones, layout, lambda f, m: report(0.05 + 0.75 * f, m), floor=floor
    )
    speed = measure_speed(readings, zones, layout, info, keyboard.top)
    seen_moving = readings.moved[readings.present & ~np.isnan(readings.moved)]
    fastest = max(speed, float(np.percentile(seen_moving, 99.5)) if len(seen_moving) else 0.0)
    if fastest > layout.band_rows - 1:
        # The notes fall further in a frame than the strip is tall: rows would slip between frames. Read again, taller.
        layout = plan_layout(keyboard, info.fps, band_rows=int(np.ceil(max(2 * speed, fastest + 2))) + 2)
        readings = read_all_frames(
            video, info, settings, samples, view, zones, layout, lambda f, m: report(0.05 + 0.75 * f, m), floor=floor
        )
        speed = measure_speed(readings, zones, layout, info, keyboard.top)
    del samples
    report(0.82, "Measuring the notes")
    shown, even, steady = scroll_positions(readings, speed, info.fps)
    roll = unroll_readings(readings, shown, even, layout)
    least_rows = max(3, int(round(0.02 * info.fps * speed)))  # nothing shorter than a fiftieth of a second
    bars = read_bars(roll, zones, keyboard, least_rows=least_rows)
    if not bars:
        raise JobError(
            "no_notes_found",
            "The keyboard was found, but no falling notes. The piano-video option reads coloured bars that fall "
            "onto the keys.",
        )
    check = check_against_keys(bars, roll, readings)

    keys = keyboard.keys
    hands = assign_hands(
        [bar.colour for bar in bars], [keys[bar.key].pitch for bar in bars], [keys[bar.key].black for bar in bars]
    )
    hands_from = "colours"
    if all(hand == UNKNOWN for hand in hands):
        # One colour for both hands: the picture does not say which hand plays what.
        hands = split_by_pitch([keys[bar.key].pitch for bar in bars])
        hands_from = "pitch"
    first_frame = int(np.flatnonzero(readings.present)[0])
    head = first_frame + layout.lead / speed + layout.band_rows / speed + 3.0
    rows: list[tuple[float, float, int, str, str]] = []  # start frame, end frame, key, hand, source
    # The keys light a little before a bar's lower edge reaches the row of the keys; as rows, that is ...
    early = check.shift * speed
    for bar, hand in zip(bars, hands, strict=True):
        rows.append(
            (float(roll.clock_of(bar.start + early)), float(roll.clock_of(bar.end + early)), bar.key, hand, "bars")
        )
    recovered = 0
    leftover = []
    for key, a, b in check.unexplained:
        if a <= head:
            # Lit before its bar could have reached the reading strip: the roll appeared with this note under way.
            rows.append((max(float(first_frame), a - 0.5), b - 0.5, key, UNKNOWN, "keys"))
            recovered += 1
        else:
            leftover.append((key, a, b))
    if recovered:
        known = [(keys[k].pitch, h) for _, _, k, h, s in rows if s == "bars" and h != UNKNOWN]
        if known:
            rows = [
                (a, b, k, min(known, key=lambda item: abs(item[0] - keys[k].pitch))[1] if s == "keys" else h, s)
                for a, b, k, h, s in rows
            ]

    pitches = [keys[k].pitch for _, _, k, _, _ in rows]
    seconds = [(a / info.fps, b / info.fps) for a, b, _, _, _ in rows]
    octave: dict[str, object] = {"source": "full keyboard" if keyboard.octave_certain else "middle of the keyboard"}
    offset_fact: dict[str, object] = {}
    shift_seconds = 0.0
    transpose = 0
    if audio is not None:
        samples_audio, rate = audio
        report(0.9, "Checking the notes against the audio")
        if not keyboard.octave_certain:
            scores = sound.octave_scores(
                [(s, e, p) for (s, e), p in zip(seconds, pitches, strict=True)], samples_audio, rate
            )
            octave["agreement"] = {str(k): round(v, 4) for k, v in scores.items()}
            best = sound.best_octave(scores)
            if best is not None:
                octave["source"] = "audio"
                if best != 0 and min(pitches) + best >= PIANO_LOWEST and max(pitches) + best <= PIANO_HIGHEST:
                    transpose = best
        bar_onsets = [s for (s, _), (_, _, _, _, source) in zip(seconds, rows, strict=True) if source == "bars"]
        measured = sound.audio_offset(bar_onsets, samples_audio, rate)
        if measured is not None:
            offset_fact = {"seconds": round(measured.seconds, 4), "sharpness": round(measured.sharpness, 1)}
            if measured.reliable:
                shift_seconds = measured.seconds
                offset_fact["applied"] = True
    octave["shift"] = transpose

    notes = [
        VisualNote(
            start=round(max(0.0, s + shift_seconds), 4),
            end=round(max(0.0, s + shift_seconds) + max(e - s, 0.02), 4),
            pitch=p + transpose,
            hand=hand,
            source=source,
        )
        for (s, e), p, (_, _, _, hand, source) in zip(seconds, pitches, rows, strict=True)
    ]
    notes.sort(key=lambda note: (note.start, note.pitch))
    by_hand = {name: sum(note.hand == name for note in notes) for name in (RIGHT, LEFT, UNKNOWN)}
    present = readings.present
    facts: dict[str, object] = {
        "video": {
            "width": info.source_width,
            "height": info.source_height,
            "read_at": [info.width, info.height],
            "fps": info.fps,
        },
        "keyboard": {
            "keys": len(keys),
            "lowest": pitch_name(keyboard.lowest + transpose),
            "highest": pitch_name(keyboard.highest + transpose),
            "white_key_width": round(keyboard.white_width, 2),
            "top_row": keyboard.top,
        },
        "octave": octave,
        "roll": {
            "first_second": round(first_frame / info.fps, 3),
            "last_second": round(float(np.flatnonzero(present)[-1]) / info.fps, 3),
            "rows_per_second": round(speed * info.fps, 2),
            "steady": steady,
            "seconds_on_screen": round(keyboard.top / (speed * info.fps), 2),
            "film_behind": floor is not None,
            "bar_brightness": floor,
            "keyboard_shown": round(float(present.mean()), 4),
        },
        "notes": {
            "total": len(notes),
            "right_hand": by_hand[RIGHT],
            "left_hand": by_hand[LEFT],
            "hand_unknown": by_hand[UNKNOWN],
            "hands_from": hands_from,
            "separated_by_corners": sum(bar.joined for bar in bars),
            "between_two_sounding_keys": sum(bar.doubtful for bar in bars),
            "from_lit_keys_only": recovered,
        },
        "key_check": {
            "keys_light": check.keys_light,
            "bars_confirmed": int(check.confirmed.sum()),
            "bars": len(bars),
            "lit_without_a_bar": len(leftover),
            "timing_shift_frames": round(check.shift, 3),
            "onsets_timed": check.onsets_used,
        },
        "audio_offset": offset_fact,
    }
    log.info("visual.read", **{k: v for k, v in facts.items() if k in ("notes", "key_check", "octave", "roll")})
    return VisualReading(notes, facts)


def write_midi(notes: list[VisualNote], path: Path, *, velocity: int = 80) -> Path:
    """The notes as one piano track, the form the velocity model and the score model read."""
    import mido

    ticks_per_beat, tempo = 960, 500_000  # 1920 ticks a second
    per_second = ticks_per_beat * 1_000_000 / tempo
    events: list[tuple[int, int, int]] = []  # (tick, 0 for off / 1 for on, pitch)
    for note in notes:
        on = int(round(note.start * per_second))
        off = max(on + 1, int(round(note.end * per_second)))
        events += [(on, 1, note.pitch), (off, 0, note.pitch)]
    events.sort()
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="piano", time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    now = 0
    for tick, kind, pitch in events:
        message = "note_on" if kind else "note_off"
        track.append(mido.Message(message, note=pitch, velocity=velocity if kind else 0, time=tick - now))
        now = tick
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    midi.tracks.append(track)
    tmp = path.with_suffix(".tmp.mid")
    midi.save(str(tmp))
    tmp.replace(path)
    return path

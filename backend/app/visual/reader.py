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

A video whose keys are too small for that, or whose keyboard is filmed, is read from one
picture of its whole roll instead (``app.visual.stacked``); its notes are checked against
the audio alone, since a filmed key does not light, and where it draws both hands in one
colour the hands are told apart by where the player's own hands are.

Nothing here is a model: the notes are measured, not estimated, and the measurements
that came out are reported with the notes (``VisualReading.facts``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.config import Settings
from app.errors import JobError
from app.logging import get_logger
from app.visual import audio as sound
from app.visual import stacked
from app.visual.frames import VideoInfo, probe_video, read_strips, sample_frames
from app.visual.hands import LEFT, RIGHT, UNKNOWN, assign_hands, divide_hands, hands_by_divide, split_by_pitch
from app.visual.keyboard import Keyboard, KeyboardView, find_keyboard
from app.visual.roll import (
    COVERED,
    DEEP_NOTCH,
    FAINT,
    SOLID,
    Bar,
    Roll,
    Zones,
    build_zones,
    enclosed,
    faint_notches,
    frame_shifts,
    jump_shift,
    lines_across,
    read_bars,
    reduce_strip,
    scroll_speed,
    unroll,
    without_a_bars_colour,
)

log = get_logger(__name__)

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
KEY_ROWS = 4
SPEED_ROWS = 2
PIANO_LOWEST, PIANO_HIGHEST = 21, 108
Progress = Callable[[float, str], None]


def _even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


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
    lines: list[float] = field(default_factory=list)  # seconds at which a bar line drawn across the roll lands

    def save(self, path: Path) -> None:
        payload = {"facts": self.facts, "notes": [asdict(note) for note in self.notes], "lines": self.lines}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> VisualReading:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            [VisualNote(**note) for note in payload["notes"]],
            payload["facts"],
            [float(t) for t in payload.get("lines") or []],
        )


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


def plan_layout(
    keyboard: Keyboard, fps: float, *, band_rows: int | None = None, film: bool = False, flare: int = 0
) -> Layout:
    """Where to read: ``film`` when a film plays behind the notes (see ``backdrop_floor``); ``flare``: the
    rows above the keyboard that the light of a landing note reaches (see ``flare_reach``)."""
    roll_height = keyboard.top
    period = keyboard.white_width
    # High enough to clear the flare drawn where a note lands, low enough to stay near it: the
    # further up, the more a screen recording's uneven motion blurs the moment a bar lands. The
    # one video seen with a film behind its notes also bursts a flare where a note lands, nearly
    # two white keys up over the neighbouring keys; its notes are read higher.
    lead = max(round((2.4 if film else 1.6) * period), round(0.1 * roll_height))
    if flare >= lead:
        # A glow that floods the keys around a landing note higher than that: read clear above it.
        lead = flare + round(FLARE_CLEAR * period)
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


FLARE = 0.15  # share of the covered keys of a row, beyond the roll's own, that hold no bar: a landing note's light
FLARE_CLEAR = 0.5  # white keys above the end of that light at which the notes are then read
FLARE_SEEN = 6  # covered keys (counted row by row) a frame must show at a height to say anything about it


def flare_reach(samples: NDArray[np.uint8], view: KeyboardView, zones: Zones) -> int:
    """Rows above the keyboard up to which the light drawn where a note lands is seen; 0 when none is.

    Most videos burst a little light where a note lands, and the strip the notes are read from
    lies above it. Some flood the keys around the note with it, a white glow some white keys
    high: read inside that glow, every key beside a landing note is covered (a note is read
    there) and a note that has ended is covered still. Such light is told from a bar by what a
    bar is everywhere else on the roll: one flat colour, of the few the bars have. Row by row
    from the keyboard up, the keys that are covered without being that (not solidly, or in a
    colour no bar has higher up) are counted against all the keys that are covered, frame by
    frame; the light ends where the typical frame shows no more of them than the rest of the
    roll does.
    """
    keyboard = view.keyboard
    top, period = keyboard.top, keyboard.white_width
    usable = samples[view.usable]
    furthest = min(top // 2, round(6 * period))  # as far up as a flare is looked for
    first = int(0.15 * top)
    above = top - furthest - first  # rows of the roll further up than that: what bars alone look like
    if len(usable) < 4 or above < 2 * period:
        return 0
    background = roll_background(samples, view, range(first, top))
    some = zones.coloured
    covered = np.zeros((len(usable), top - first, some), dtype=bool)
    solid = np.zeros_like(covered)
    code = np.zeros(covered.shape, dtype=np.int16)
    for number, frame in enumerate(usable):  # one at a time: the sums over a whole picture are large
        stats = reduce_strip(frame[None, first:top], background, zones)
        covered[number] = stats.cover[0] >= COVERED
        solid[number] = stats.solid[0, :, :some] >= COVERED
        level = np.clip(stats.colour[0] / 16.0, 0, 15).astype(np.int16)
        code[number] = (level[..., 0] << 8) | (level[..., 1] << 4) | level[..., 2]
    bar = covered & solid
    # The colours bars have: those of a fair share of the solid keys higher up, in more frames than a caption stays.
    seen = np.stack([np.bincount(code[n, :above][bar[n, :above]], minlength=4096) for n in range(len(usable))])
    total = seen.sum(axis=0)
    known = (total >= 0.002 * max(1, int(total.sum()))) & ((seen > 0).sum(axis=0) >= max(2, round(0.1 * len(usable))))
    padded = np.pad(known.reshape(16, 16, 16), 1)
    near = np.zeros((16, 16, 16), dtype=bool)
    for dr in range(3):
        for dg in range(3):
            for db in range(3):
                near |= padded[dr : dr + 16, dg : dg + 16, db : db + 16]
    no_bar = covered & ~(solid & near.reshape(-1)[code])
    shown = covered[:, :above].sum(axis=(1, 2))
    telling = shown >= 50
    if telling.sum() < 3:
        return 0
    usual = float(np.median(no_bar[:, :above].sum(axis=(1, 2))[telling] / shown[telling]))
    step = max(1, round(period / 8))
    reach = 0
    for up in range(0, furthest, step):
        rows = slice(top - first - up - step, top - first - up)
        keys = covered[:, rows].sum(axis=(1, 2))
        telling = keys >= FLARE_SEEN
        if telling.sum() < 3:
            if reach or up > period:
                break  # nothing lands here often enough to tell
            continue
        share = float(np.median(no_bar[:, rows].sum(axis=(1, 2))[telling] / keys[telling]))
        if share < usual + FLARE:
            break
        reach = up + step
    return reach


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
    across: NDArray[np.float32]  # [frames, band rows] how far most of each row is from the empty picture (grey levels)
    # For the few frames after which the picture jumped: the rows above the band, read like it (see roll.unroll).
    spare: dict[int, tuple[NDArray[np.uint8], ...]]
    # How far the picture jumped into a frame after standing still, where that is further than ``moved`` can tell.
    jumped: dict[int, float]


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
        name: []
        for name in ("solid", "grey", "cover", "colour", "spread", "upper", "lit", "present", "gain", "moved", "across")
    }
    spare: dict[int, tuple[NDArray[np.uint8], ...]] = {}
    jumped: dict[int, float] = {}
    last_tall: NDArray[np.uint8] | None = None  # the tall strip of the frame before this run of frames
    last_moved = float("nan")
    grey_behind = background[layout.above : tall].astype(np.float32).mean(axis=-1)  # [band rows, columns]
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
        moved = moved if last_mask is not None else np.concatenate(([np.nan], moved)).astype(np.float32)
        before = len(joined) - frames  # 1 when ``joined`` begins with the frame before this run
        for i in range(frames):
            # A picture that stood still and then moved further than it can be followed from one frame
            # to the next: a recording that stuttered. How far it jumped is looked for over the whole strip.
            stood = moved[i - 1] == 0.0 if i > 0 else last_moved == 0.0
            if stood and np.isnan(moved[i]) and i + before >= 1:
                jump = jump_shift(joined[i + before - 1], joined[i + before], 0, tall - 4)
                if not np.isnan(jump):
                    jumped[done + i] = jump
            leaps = moved[i] > layout.band_rows - 1.5 or (stood and np.isnan(moved[i]))
            if floor is None and leaps and len(spare) < SPARES:
                # Rows will have passed the band unread: they are still above it in the frame before.
                pixels = chunk[i - 1, : layout.above] if i > 0 else last_tall
                if pixels is not None:
                    above = reduce_strip(pixels[None, : layout.above], background[: layout.above], zones)
                    spare[done + i - 1] = (
                        np.rint(above.solid[0] * 255).astype(np.uint8),
                        np.rint(above.grey[0]).astype(np.uint8),
                        np.rint(above.cover[0] * 255).astype(np.uint8),
                        np.rint(above.colour[0]).astype(np.uint8),
                        np.clip(np.rint(above.spread[0]), 0, 255).astype(np.uint8),
                    )
        parts["moved"].append(moved)
        last_mask, last_moved, last_tall = masks[-1], float(moved[-1]), chunk[-1, : layout.above].copy()
        stats = reduce_strip(
            chunk[:, layout.above :], background[layout.above :], zones, ignore=ignore, covered=covered
        )
        # What most of each row of the band differs by from the empty picture: a line drawn right across
        # the roll (a bar line) shifts the whole row, and no note does.
        band_grey = chunk[:, layout.above : tall].astype(np.float32).mean(axis=-1)
        parts["across"].append(
            np.median(band_grey - grey_behind[None], axis=2).astype(np.float32)
            if floor is None
            else np.zeros((frames, layout.band_rows), dtype=np.float32)
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
    return Readings(**joined, spare=spare, jumped=jumped)


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
STILL = 0.25  # rows: a picture that moved less than this since the frame before stood still
SETTLED = 6  # frames of the usual motion after which a stutter is over
SLOWEST_TOLD = 2.0  # rows a frame: a roll slower than this stands still, to the row, in many an ordinary frame
SPARES = 400  # frames whose rows above the strip are kept at the most


@dataclass
class Scroll:
    """How far the roll had scrolled in each frame."""

    shown: NDArray[np.float64]  # [frames] as the picture shows it
    even: NDArray[np.float64]  # [frames] as it runs with the sound: the unevenness of the recording taken out
    steady: bool  # the picture moves the same distance every frame (a rendered video, not a screen recording)
    placed: NDArray[np.bool_]  # [frames] where the picture was is known (False: between two jumps of unknown size)
    stutters: int = 0  # times the picture stood still and then caught up, or went on from where it stood
    frames_still: int = 0


def _through_stutters(
    moved: NDArray[np.float32], present: NDArray[np.bool_], speed: float, jumped: dict[int, float]
) -> Scroll:
    """The steady line ``speed`` times the frame number, bent wherever the picture stood still.

    A recording of a screen can stutter: one picture stays for some frames, and then the next
    shows the roll where it has got to meanwhile (the sound never stopped), or carries on from
    where it stood (the player waited too). Laid along the steady line, the picture that stayed
    would be written down the roll again and again, and what the jump passed over would be put
    where it never was. So from a still frame until the picture moves as usual again, each frame
    is placed by how far the picture is measured to have moved. If that comes to what the steady
    line makes it (the recording caught up), the roll's clock runs on unbent; if not, the roll
    waited, and its clock with it. One jump of unknown size is what is missing to catch up; the
    frames between two such jumps cannot be placed and are left out.
    """
    count = len(moved)
    line = speed * np.arange(count, dtype=np.float64)
    moved = moved.copy()
    for frame, rows in jumped.items():
        moved[frame] = rows
    measured = present & ~np.isnan(moved)
    usual = measured & (np.abs(moved - speed) <= max(1.0, 0.25 * speed))
    still = measured & (moved <= STILL)
    # Before the roll has ever been seen to move it has not stuttered: it has not begun.
    begun = int(np.argmax(usual)) if usual.any() else count
    still[:begun] = False
    scroll = Scroll(line, line, True, np.ones(count, dtype=bool))
    if not still.any() or speed < SLOWEST_TOLD:
        return scroll
    shown, even = line.copy(), line.copy()
    behind = 0.0  # rows the roll has fallen behind the steady line for good
    frame = 0
    while frame < count:
        if not still[frame]:
            shown[frame] = even[frame] = line[frame] - behind
            frame += 1
            continue
        first, quiet, last = frame, 0, frame
        while last < count and quiet < SETTLED and last - first < 20 * SETTLED + int(still[first:last].sum()):
            leapt = np.isnan(moved[last]) and last > 0 and still[last - 1]
            quiet = 0 if (still[last] or leapt or (measured[last] and not usual[last])) else quiet + 1
            last += 1
        last -= quiet  # the stutter is frames first .. last - 1
        steps = np.where(usual[first:last] | ~measured[first:last], speed, moved[first:last]).astype(np.float64)
        steps[still[first:last]] = 0.0
        unknown = [
            n for n in range(last - first) if np.isnan(moved[first + n]) and first + n > 0 and still[first + n - 1]
        ]
        net = float(steps.sum()) - speed * (last - first)
        start = shown[first - 1] if first > 0 else -speed - behind
        forward = start + np.cumsum(steps)
        # Caught up: on the steady line again once it is over. What the measured steps leave over
        # belongs to the jump (the one of unknown size, or else the largest step).
        backward = (line[last - 1] - behind) - (np.cumsum(steps[::-1])[::-1] - steps)
        turn, again = (unknown[0], unknown[-1]) if unknown else (int(np.argmax(steps)),) * 2
        before = forward[turn - 1] if turn > 0 else start
        # (A jump over nearly the whole strip is measured to a few rows: what it leaves over is not a wait.)
        stood = speed * float(still[first:last].sum())
        caught_up = (bool(unknown) or abs(net) <= max(2.0, 0.75 * speed, 0.25 * stood)) and backward[again] >= before
        if caught_up:
            shown[first : first + turn] = forward[:turn]
            shown[first + again : last] = backward[again:]
            if again > turn:
                # Between two jumps of unknown size: somewhere between where it stood and where it got to.
                shown[first + turn : first + again] = np.linspace(before, backward[again], again - turn + 2)[1:-1]
                scroll.placed[first + turn : first + again] = False
            even[first:last] = line[first:last] - behind
        else:
            # It went on from where it stood: the roll waited, and its clock with it.
            shown[first:last] = even[first:last] = forward
            behind -= net
        scroll.stutters += 1
        scroll.frames_still += int(still[first:last].sum())
        frame = last
    scroll.shown, scroll.even = shown, even
    return scroll


def scroll_positions(readings: Readings, speed: float, fps: float) -> Scroll:
    """How far the roll had scrolled in each frame.

    A video rendered from the notes moves the same distance every frame: ``speed`` times the
    frame number says it all, exactly (but for a recording that stutters, see ``_through_stutters``).
    A screen recording repeats and skips frames, so the distance moved is taken as measured, frame
    by frame. Its roll still runs evenly with the sound, which a running average of the measured
    positions recovers.
    """
    count = len(readings.moved)
    measured = readings.moved[readings.present & ~np.isnan(readings.moved)]
    line = speed * np.arange(count, dtype=np.float64)
    if len(measured) < 30:
        return Scroll(line, line, True, np.ones(count, dtype=bool))
    spread = float(np.median(np.abs(measured - np.median(measured)))) * 1.4826
    if spread < UNEVEN and abs(float(np.median(measured)) - speed) < 0.2 * speed:
        return _through_stutters(readings.moved, readings.present, speed, readings.jumped)
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
    return Scroll(shown, even, False, np.ones(count, dtype=bool))


def unroll_readings(readings: Readings, scroll: Scroll, layout: Layout) -> Roll:
    frames = np.flatnonzero(readings.present & scroll.placed)
    if len(frames) == 0:
        raise JobError("keyboard_not_found", "The keyboard is never on screen long enough to read notes.")
    return unroll(
        readings.solid[frames],
        readings.grey[frames],
        readings.cover[frames],
        readings.colour[frames],
        readings.spread[frames],
        frames,
        scroll.shown,
        scroll.even,
        float(layout.lead),
        across=readings.across[frames],
        spare=readings.spare,
    )


def _starts(mask: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Indices where a run of True begins."""
    return np.flatnonzero(mask & ~np.concatenate(([False], mask[:-1])))


def _runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


# More bars than two hands have fingers, all on neighbouring keys and starting and ending together,
# are one thing laid across the keyboard (a title banner, a card, a wipe), not notes.
OVERLAY_BARS = 11


def _in_order(values: list[float]) -> bool:
    """Whether ``values`` rise or fall along the list (rank correlation of at least 0.8; ties share a rank)."""
    v = np.asarray(values, dtype=np.float64)
    ranks = (v[None, :] < v[:, None]).sum(axis=1) + ((v[None, :] == v[:, None]).sum(axis=1) - 1) / 2.0
    if len(v) < 3 or float(ranks.std()) == 0.0:
        return False
    return abs(float(np.corrcoef(ranks, np.arange(len(v)))[0, 1])) >= 0.8


def drop_overlays(bars: list[Bar], keyboard: Keyboard, tolerance: float) -> tuple[list[Bar], int]:
    """(``bars`` without those of overlays, how many were dropped). ``tolerance``: roll rows within which
    bars count as starting, or ending, together.

    A tutorial may scroll a banner (the song's title, say) down over the whole keyboard before the
    notes begin. Read like the notes, it is a bar on every key it crosses, all struck at one moment
    and released at another: dozens of neighbouring keys at once, which no two hands can play.
    """
    order = sorted(range(len(bars)), key=lambda i: bars[i].start)
    dropped: set[int] = set()
    first = 0
    while first < len(order):
        last = first
        while last + 1 < len(order) and bars[order[last + 1]].start - bars[order[first]].start <= tolerance:
            last += 1
        group = order[first : last + 1]
        first = last + 1
        if len(group) < OVERLAY_BARS:
            continue
        by_end = sorted(group, key=lambda i: bars[i].end)
        while by_end:
            same = [i for i in by_end if bars[i].end - bars[by_end[0]].end <= tolerance]
            by_end = by_end[len(same) :]
            pitches = sorted((keyboard.keys[bars[i].key].pitch, i) for i in same)
            # Runs of neighbouring keys: no key left out between two of them.
            breaks = [k for k in range(1, len(pitches)) if pitches[k][0] - pitches[k - 1][0] > 2]
            for a, b in zip([0, *breaks], [*breaks, len(pitches)], strict=True):
                run = [i for _, i in pitches[a:b]]
                # A glissando also strikes many neighbouring keys in a moment, but one after the other.
                if len(run) >= OVERLAY_BARS and not _in_order([bars[i].start for i in run]):
                    dropped.update(run)
    return [bar for i, bar in enumerate(bars) if i not in dropped], len(dropped)


SLIVER = 0.06  # seconds: shorter than any note that is struck again at once (a 32nd note at 120 is 0.0625)


def merge_slivers(bars: list[Bar], shortest: float) -> tuple[list[Bar], int]:
    """(``bars`` with slivers joined back, how many were). ``shortest``: roll rows below which a piece is a sliver.

    Where a bar's note name or a blemish of the encoding reads as the seam between two touching
    bars, one note comes out as two, one of them a few rows long. No key is struck twice within
    a twentieth of a second: a piece that short, touching another on its key, is part of it.
    """
    by_key: dict[int, list[Bar]] = {}
    for bar in sorted(bars, key=lambda b: (b.key, b.start)):
        by_key.setdefault(bar.key, []).append(bar)
    out: list[Bar] = []
    merged = 0
    for members in by_key.values():
        kept = [members[0]]
        for bar in members[1:]:
            before = kept[-1]
            touching = bar.start - before.end <= 1.5
            if touching and min(bar.end - bar.start, before.end - before.start) < shortest:
                longer = before if before.end - before.start >= bar.end - bar.start else bar
                kept[-1] = Bar(
                    before.key, before.start, bar.end, longer.colour, before.joined, longer.doubtful, longer.shared
                )
                merged += 1
            else:
                kept.append(bar)
        out += kept
    out.sort(key=lambda b: (b.start, b.key))
    return out, merged


HEARD = 0.06  # seconds within which a seam and an attack in the audio are the same moment
FAINT_APART = 0.09  # seconds a faint seam must lie from either end of its bar (a sixteenth at 160 to the minute)


def split_at_faint_seams(
    bars: list[Bar],
    notches: list[tuple[int, float, float]],
    roll: Roll,
    fps: float,
    audio: tuple[NDArray[np.float32], int] | None,
    *,
    together: float,
) -> tuple[list[Bar], dict[str, int]]:
    """(``bars`` cut at the faint seams that something else confirms, how many by which sign).

    Two bars that touch are told apart by their rounded corners; in a small picture a seam can
    leave too faint a notch to cut on, and two notes are read as one. ``notches`` are those
    faint places (``faint_notches``: bar, row, depth). One is a seam when

    * it is deep, though short of what a seam is cut on outright (``DEEP_NOTCH``); or
    * another bar has one at the same moment (within ``together`` rows): a chord struck again; or
    * it divides its bar into pieces as long as a bar touching it on the same key: one more of a
      run of repeated notes; or
    * the audio has a strong attack there that no note read so far accounts for.
    """
    spare = np.zeros(0)
    offset = None
    if audio is not None:
        samples, rate = audio
        onsets = [float(roll.clock_of(bar.start)) / fps for bar in bars]
        offset = sound.audio_offset(onsets, samples, rate)
        if offset is not None and offset.reliable:
            struck = np.array(sorted(onsets)) + offset.seconds
            spare = np.array([t for t in sound.attack_times(samples, rate) if np.abs(struck - t).min() > HEARD])
    cuts: dict[int, list[float]] = {}
    signs = {"deep": 0, "together": 0, "run": 0, "heard": 0}
    touching: dict[int, list[float]] = {}  # per bar: the lengths of the bars it touches on its key
    by_key: dict[int, list[int]] = {}
    for number in sorted(range(len(bars)), key=lambda i: bars[i].start):
        by_key.setdefault(bars[number].key, []).append(number)
    for members in by_key.values():
        for one, next_one in zip(members, members[1:], strict=False):
            if bars[next_one].start - bars[one].end <= 1.5:
                touching.setdefault(one, []).append(bars[next_one].end - bars[next_one].start)
                touching.setdefault(next_one, []).append(bars[one].end - bars[one].start)
    for number, row, depth in notches:
        others = [
            r for n, r, _ in notches if n != number and bars[n].key != bars[number].key and abs(r - row) <= together
        ]
        pieces = (row - bars[number].start, bars[number].end - row)
        run = any(
            min(abs(piece - length) for piece in pieces) <= 0.2 * length and min(pieces) >= 0.75 * length
            for length in touching.get(number, [])
        )
        heard = False
        if offset is not None and len(spare):
            heard = bool(np.abs(spare - (float(roll.clock_of(row)) / fps + offset.seconds)).min() <= HEARD)
        sign = "deep" if depth >= DEEP_NOTCH else "together" if others else "run" if run else "heard" if heard else None
        if sign is not None:
            cuts.setdefault(number, []).append(row)
            signs[sign] += 1
    out: list[Bar] = []
    for number, bar in enumerate(bars):
        edges = [bar.start, *sorted(cuts.get(number, [])), bar.end]
        for piece, (a, b) in enumerate(zip(edges, edges[1:], strict=False)):
            out.append(Bar(bar.key, a, b, bar.colour, bar.joined if piece == 0 else True, bar.doubtful, bar.shared))
    return out, signs


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


GREY = 25.0  # largest difference between the channels of a colour that is grey
COLOURED = 0.98  # share of the bars whose keys light that must be clearly coloured for a grey piece to be none


def no_bars_colour(bars: list[Bar], check: KeyCheck) -> list[bool]:
    """Which of ``bars`` are no bars: grey where the bars are coloured, and their key never lit.

    A roll may print other things than notes down its side: the numbers of the measures, in grey,
    over the lowest keys. A stroke of a digit covers a key's zone for a few rows as a bar would.
    Two things tell it from one, and both must: the key under it does not light, and it has no
    colour in a video whose bars all have one.
    """
    if not check.keys_light or not bars:
        return [False] * len(bars)
    chroma = np.array([max(bar.colour) - min(bar.colour) for bar in bars])
    lit = check.confirmed
    if lit.sum() < 20 or float((chroma[lit] >= 2 * GREY).mean()) < COLOURED:
        return [False] * len(bars)
    return [bool(c < GREY and not ok) for c, ok in zip(chroma, lit, strict=True)]


TWO_HANDS = 0.3  # share of the frames in which skin shows on both sides of the divide for the hands to be read


def _colour_of(pile: stacked.Stack, lane: stacked.Lane, block: stacked.Block) -> tuple[float, float, float]:
    """The mean colour of a block: the middle half of its lane, clear of its ends."""
    rows = len(pile.colour)
    first, last = int(np.ceil(block.start)) + 1, int(np.floor(block.end)) - 1
    if last <= first:
        first, last = int(np.floor(block.start)), int(np.ceil(block.end)) + 1
    first, last = max(0, min(first, rows - 1)), max(1, min(last, rows))
    quarter = (lane.x1 - lane.x0) / 4.0
    a = max(0, int(np.floor(lane.x0 + quarter)))
    b = min(pile.colour.shape[1], max(a + 1, int(np.ceil(lane.x1 - quarter))))
    mean = pile.colour[first:last, a:b].reshape(-1, 3).astype(np.float64).mean(axis=0)
    return float(mean[0]), float(mean[1]), float(mean[2])


def _read_stacked(
    video: Path,
    settings: Settings,
    info: VideoInfo,
    samples: NDArray[np.uint8],
    view: KeyboardView,
    audio: tuple[NDArray[np.float32], int] | None,
    report: Progress,
) -> VisualReading:
    """Read a video with small keys or a filmed keyboard from one picture of its whole roll."""
    keyboard = view.keyboard
    light = stacked.block_light(samples, view)
    watched = stacked.watch(video, info, settings, samples, view, light, lambda f, m: report(0.05 + 0.2 * f, m))
    speed = stacked.measure_speed(watched, info, keyboard.top)
    pile = stacked.stack(video, info, settings, view, light, watched, speed, lambda f, m: report(0.25 + 0.55 * f, m))
    del samples
    report(0.82, "Measuring the notes")
    lanes = stacked.find_lanes(pile.picture, pile.level, view)
    presence = stacked.read_rows(pile.picture, pile.level, lanes)
    least_rows = max(
        3, int(round(0.02 * info.fps * pile.rows_per_frame))
    )  # nothing shorter than a fiftieth of a second
    blocks, faint, slivers = stacked.read_blocks(presence, lanes, least_rows=least_rows)
    if not blocks:
        raise JobError(
            "no_notes_found",
            "The keyboard was found, but no falling notes. The piano-video option reads blocks that fall onto "
            "the keys.",
        )

    def seconds_of(row: float) -> float:
        return float(pile.frame_of(row)) / info.fps

    octave: dict[str, object] = {"source": "full keyboard" if keyboard.octave_certain else "middle of the keyboard"}
    offset_fact: dict[str, object] = {}
    shift_seconds = 0.0
    transpose = 0
    heard: dict[str, object] = {}
    cut_at_faint = 0
    if audio is not None:
        samples_audio, rate = audio
        report(0.9, "Checking the notes against the audio")
        onsets = [seconds_of(block.start) for block in blocks]
        measured = sound.audio_offset(onsets, samples_audio, rate)
        if measured is not None:
            offset_fact = {"seconds": round(measured.seconds, 4), "sharpness": round(measured.sharpness, 1)}
            if measured.reliable:
                shift_seconds = measured.seconds
                offset_fact["applied"] = True
                attacks = np.array(sound.attack_times(samples_audio, rate))
                struck = np.array(sorted(onsets)) + shift_seconds
                # A faint pinch is where a note begins when the audio has an attack there that no note accounts for.
                spare = np.array([t for t in attacks if np.abs(struck - t).min() > HEARD])
                for number, row, _ in sorted(faint, key=lambda item: -item[1]):
                    if len(spare) and np.abs(spare - (seconds_of(row) + shift_seconds)).min() <= HEARD:
                        whole = blocks[number]
                        blocks[number] = stacked.Block(whole.lane, whole.start, float(row), whole.joined, False)
                        blocks.append(stacked.Block(whole.lane, float(row), whole.end, True, whole.cut_short))
                        cut_at_faint += 1
                struck = np.array(sorted(seconds_of(block.start) for block in blocks)) + shift_seconds
                near = np.array([np.abs(struck - t).min() for t in attacks]) if len(attacks) else np.zeros(0)
                heard = {
                    "attacks": len(attacks),
                    "attacks_with_a_note": int((near <= HEARD).sum()),
                    "attacks_without_a_note": int((near > HEARD).sum()),
                }
        if not keyboard.octave_certain:
            scores = sound.octave_scores(
                [
                    (seconds_of(b.start) + shift_seconds, seconds_of(b.end) + shift_seconds, lanes.lanes[b.lane].pitch)
                    for b in blocks
                ],
                samples_audio,
                rate,
            )
            octave["agreement"] = {str(k): round(v, 4) for k, v in scores.items()}
            best = sound.best_octave(scores)
            if best is not None:
                octave["source"] = "audio"
                low = min(lanes.lanes[b.lane].pitch for b in blocks)
                high = max(lanes.lanes[b.lane].pitch for b in blocks)
                if best != 0 and low + best >= PIANO_LOWEST and high + best <= PIANO_HIGHEST:
                    transpose = best
    octave["shift"] = transpose
    if audio is not None:
        rings = sound.rings_on(
            [
                (seconds_of(b.start) + shift_seconds, seconds_of(b.end) + shift_seconds, lanes.lanes[b.lane].pitch)
                for b in blocks
            ],
            audio[0],
            audio[1],
        )
        if rings is not None:
            heard["rings_on"] = round(rings, 3)

    blocks.sort(key=lambda block: (block.start, block.lane))
    pitches = [lanes.lanes[block.lane].pitch for block in blocks]
    black = [lanes.lanes[block.lane].black for block in blocks]
    hands = assign_hands([_colour_of(pile, lanes.lanes[block.lane], block) for block in blocks], pitches, black)
    hands_from = "colours"
    hands_seen = 0.0
    if all(hand == UNKNOWN for hand in hands):
        # One colour for both hands. A filmed keyboard shows the hands themselves.
        skin_rows = max(1, int(watched.skin.max()))
        divide = divide_hands(watched.skin, skin_rows)
        columns = np.arange(watched.skin.shape[1])[None, :]
        skin = watched.skin >= max(2, 0.1 * skin_rows)
        both = (skin & (columns < divide[:, None])).any(axis=1) & (skin & (columns >= divide[:, None])).any(axis=1)
        hands_seen = float(both[watched.present].mean()) if watched.present.any() else 0.0
        if hands_seen >= TWO_HANDS:
            filmed = {key.pitch: key.center for key in keyboard.keys}
            where = [filmed.get(p, lanes.lanes[b.lane].center) for p, b in zip(pitches, blocks, strict=True)]
            # The hands are filmed with the sound: a key goes down when its note is heard.
            when = [(seconds_of(block.start) + shift_seconds) * info.fps for block in blocks]
            hands = hands_by_divide(where, when, divide)
            hands_from = "hands"
        else:
            hands = split_by_pitch(pitches)
            hands_from = "pitch"

    notes = []
    for block, pitch, hand in zip(blocks, pitches, hands, strict=True):
        start = max(0.0, seconds_of(block.start) + shift_seconds)
        length = max(seconds_of(block.end) - seconds_of(block.start), 0.02)
        notes.append(
            VisualNote(start=round(start, 4), end=round(start + length, 4), pitch=pitch + transpose, hand=hand)
        )
    notes.sort(key=lambda note: (note.start, note.pitch))
    by_hand = {name: sum(note.hand == name for note in notes) for name in (RIGHT, LEFT, UNKNOWN)}
    present = watched.present
    shown = np.flatnonzero(present)
    lit = float((pile.picture >= stacked.THERE * pile.level).sum())
    facts: dict[str, object] = {
        "video": {
            "width": info.source_width,
            "height": info.source_height,
            "read_at": [info.width, info.height],
            "fps": info.fps,
        },
        "keyboard": {
            "keys": len(lanes.lanes),
            "lowest": pitch_name(lanes.lanes[0].pitch + transpose),
            "highest": pitch_name(lanes.lanes[-1].pitch + transpose),
            "white_key_width": round(keyboard.white_width, 2),
            "top_row": keyboard.top,
            "filmed": view.filmed,
        },
        "octave": octave,
        "roll": {
            "read_from": "one picture of the whole roll",
            "first_second": round(float(shown[0]) / info.fps, 3) if len(shown) else 0.0,
            "last_second": round(float(shown[-1]) / info.fps, 3) if len(shown) else 0.0,
            "rows_per_second": round(pile.speed * info.fps, 2),
            "steady": True,
            "seconds_on_screen": round(keyboard.top / (pile.speed * info.fps), 2),
            "film_behind": False,
            "bar_brightness": round(pile.level, 1),
            "keyboard_shown": round(float(present.mean()), 4),
            "block_channels": [("red", "green", "blue")[c] for c in light.channels] or ["strongest"],
            "bands_apart_before_alignment": pile.bands_late,
        },
        "lanes": {
            "white_step": round(lanes.white_step, 3),
            "white_block_width": round(lanes.white_width, 2),
            "black_block_width": round(lanes.black_width, 2),
            "white_lanes_measured": lanes.seen_white,
            "black_lanes_measured": lanes.seen_black,
            "furthest_from_filmed_key": lanes.off_keys,
        },
        "notes": {
            "total": len(notes),
            "right_hand": by_hand[RIGHT],
            "left_hand": by_hand[LEFT],
            "hand_unknown": by_hand[UNKNOWN],
            "hands_from": hands_from,
            "two_hands_seen": round(hands_seen, 3),
            "separated_by_corners": sum(block.joined for block in blocks),
            "between_two_sounding_keys": sum(block.cut_short for block in blocks),
            # A black key between two white keys that both sound is under their blocks: for this long, in all,
            # the picture could not have shown one.
            "seconds_a_black_key_was_out_of_sight": round(
                float(presence.hidden.sum()) / (pile.rows_per_frame * info.fps), 3
            ),
            "from_lit_keys_only": 0,
            "overlay_bars_dropped": 0,
            "slivers_joined": 0,
            "runs_too_short": slivers,
            "separated_at_faint_seams": {"heard": cut_at_faint, "left_whole": len(faint) - cut_at_faint},
            "let_ring": _let_ring(heard),
        },
        "picture_check": {
            "lit_columns": int(lit),
            "lit_without_a_key": int(presence.unexplained.sum()),
            "key_on_but_dark": int(presence.missing.sum()),
            "most_rows_in_a_row_unexplained": _longest_run((presence.unexplained > 2) | (presence.missing > 2)),
        },
        "key_check": {
            "keys_light": False,
            "bars_confirmed": 0,
            "bars": len(blocks),
            "lit_without_a_bar": 0,
            "timing_shift_frames": 0.0,
            "onsets_timed": 0,
        },
        "audio_offset": offset_fact,
        "audio_check": heard,
    }
    log.info(
        "visual.read", **{k: v for k, v in facts.items() if k in ("notes", "lanes", "picture_check", "octave", "roll")}
    )
    return VisualReading(notes, facts)


def _longest_run(mask: NDArray[np.bool_]) -> int:
    return max((b - a for a, b in _runs(mask)), default=0)


def _let_ring(heard: dict[str, object]) -> bool:
    """Whether the notes go on sounding after their keys are released: the pedal holds them (``audio.rings_on``)."""
    rings = heard.get("rings_on")
    return isinstance(rings, float) and rings >= sound.RINGS


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
    if view.filmed and keyboard.white_width > 1.3 * stacked.STACK_BELOW:
        # One picture of the whole roll has a row for every pixel the notes fall and a column for every
        # column of the frame: a large frame makes it larger than there is any use for. Read smaller.
        scale = 1.2 * stacked.STACK_BELOW / keyboard.white_width
        smaller = replace(info, width=_even(info.width * scale), height=_even(info.height * scale))
        samples_smaller = sample_frames(video, smaller, settings, count=settings.visual_sample_frames)
        try:
            view_smaller = find_keyboard(samples_smaller)
        except JobError:
            view_smaller = None
        if view_smaller is not None and view_smaller.filmed:
            info, samples, view, keyboard = smaller, samples_smaller, view_smaller, view_smaller.keyboard
    if view.filmed or keyboard.white_width < stacked.STACK_BELOW:
        return _read_stacked(video, settings, info, samples, view, audio, report)
    zones = build_zones(keyboard)
    floor = backdrop_floor(samples, view)
    # (In front of a film every colour is seen everywhere: the flare of such a video is allowed for as it is.)
    flare = flare_reach(samples, view, zones) if floor is None else 0
    layout = plan_layout(keyboard, info.fps, film=floor is not None, flare=flare)
    readings = read_all_frames(
        video, info, settings, samples, view, zones, layout, lambda f, m: report(0.05 + 0.75 * f, m), floor=floor
    )
    speed = measure_speed(readings, zones, layout, info, keyboard.top)
    seen_moving = readings.moved[readings.present & ~np.isnan(readings.moved)]
    fastest = max(speed, float(np.percentile(seen_moving, 99.5)) if len(seen_moving) else 0.0)
    if fastest > layout.band_rows - 1:
        # The notes fall further in a frame than the strip is tall: rows would slip between frames. Read again, taller.
        layout = plan_layout(
            keyboard,
            info.fps,
            band_rows=int(np.ceil(max(2 * speed, fastest + 2))) + 2,
            film=floor is not None,
            flare=flare,
        )
        readings = read_all_frames(
            video, info, settings, samples, view, zones, layout, lambda f, m: report(0.05 + 0.75 * f, m), floor=floor
        )
        speed = measure_speed(readings, zones, layout, info, keyboard.top)
    del samples
    report(0.82, "Measuring the notes")
    scroll = scroll_positions(readings, speed, info.fps)
    steady = scroll.steady
    roll = unroll_readings(readings, scroll, layout)
    least_rows = max(3, int(round(0.02 * info.fps * speed)))  # nothing shorter than a fiftieth of a second
    bars, slivers = merge_slivers(read_bars(roll, zones, keyboard, least_rows=least_rows), SLIVER * info.fps * speed)
    faint_pieces = without_a_bars_colour(bars, roll, zones, keyboard)
    bars = [bar for bar, faint in zip(bars, faint_pieces, strict=True) if not faint]
    bars, overlays = drop_overlays(bars, keyboard, tolerance=3.0 * speed)
    notches = faint_notches(
        roll,
        zones,
        keyboard,
        bars,
        # A bar's own rounded ends and the shading at its foot reach some rows in: a faint seam leaves
        # a piece on either side longer than the shortest note there is to separate.
        least=max(least_rows, int(np.ceil(FAINT_APART * info.fps * speed))),
        wide=max(4, int(np.ceil(HEARD * info.fps * speed))),
    )
    bars, faint = split_at_faint_seams(bars, notches, roll, info.fps, audio, together=max(2.0, 0.02 * info.fps * speed))
    if not bars:
        raise JobError(
            "no_notes_found",
            "The keyboard was found, but no falling notes. The piano-video option reads coloured bars that fall "
            "onto the keys.",
        )
    check = check_against_keys(bars, roll, readings)
    strays = no_bars_colour(bars, check)
    if any(strays):
        bars = [bar for bar, stray in zip(bars, strays, strict=True) if not stray]
        check.confirmed = check.confirmed[~np.array(strays)]

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
    heard: dict[str, object] = {}
    if audio is not None:
        rings = sound.rings_on(
            [(s + shift_seconds, e + shift_seconds, p + transpose) for (s, e), p in zip(seconds, pitches, strict=True)],
            audio[0],
            audio[1],
        )
        if rings is not None:
            heard["rings_on"] = round(rings, 3)

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
    # The bar lines the roll draws across itself land as the bars do, and are timed as they are.
    drawn = lines_across(roll, max(5, round(0.12 * keyboard.white_width)))
    lines = [
        round(float(roll.clock_of(row + early)) / info.fps + shift_seconds, 4) for row in drawn if row + early >= 0
    ]
    apart = np.diff(lines)
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
            "landing_light_rows": flare,
            "read_rows_above_keys": layout.lead,
            "stutters": scroll.stutters,
            "frames_still": scroll.frames_still,
            "rows_read_above_the_strip": roll.filled,
            "rows_never_shown": int((~roll.seen).sum()),
            "keyboard_shown": round(float(present.mean()), 4),
        },
        "notes": {
            "total": len(notes),
            "right_hand": by_hand[RIGHT],
            "left_hand": by_hand[LEFT],
            "hand_unknown": by_hand[UNKNOWN],
            "hands_from": hands_from,
            "separated_by_corners": sum(bar.joined and not bar.shared for bar in bars),
            "two_hands_on_one_key": sum(bar.shared for bar in bars),
            "between_two_sounding_keys": sum(bar.doubtful for bar in bars),
            "from_lit_keys_only": recovered,
            "overlay_bars_dropped": overlays,
            "slivers_joined": slivers,
            "faint_pieces_dropped": sum(faint_pieces),
            "grey_pieces_dropped": sum(strays),
            "separated_at_faint_seams": faint,
            "let_ring": _let_ring(heard),
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
        "audio_check": heard,
        "bar_lines": {"drawn": len(lines), "seconds_apart": round(float(np.median(apart)), 4) if len(apart) else None},
    }
    log.info("visual.read", **{k: v for k, v in facts.items() if k in ("notes", "key_check", "octave", "roll")})
    return VisualReading(notes, facts, lines)


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

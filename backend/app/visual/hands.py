"""Which hand plays each note, from the colour of its bar.

Tutorial videos draw the two hands in two colours. Within a hand, the bars of black
keys are usually a darker and slightly different shade, so colours are not compared
across black and white keys: the white-key bars are grouped by hue first, and each
black-key bar then joins the hand whose black-key shade it has.

Hue is measured on the two colour-difference axes (red against green, yellow against
blue), where white label text and dark outlines inside a bar count for nothing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

RIGHT, LEFT, UNKNOWN = "right", "left", "unknown"
MIN_HUE_GAP = 40.0  # degrees between two hands' colours
MIN_CHROMA = 18.0  # below this a bar is grey or white: it has no hue
MIN_SHARE = 0.04  # a colour used by fewer notes than this is not a hand


def split_by_pitch(pitches: Sequence[int], middle: int = 60) -> list[str]:
    """Hands for a video that draws both in one colour: from middle C up is the right hand."""
    return [RIGHT if pitch >= middle else LEFT for pitch in pitches]


def hue(colours: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(hue in degrees, chroma) of RGB rows."""
    red_green = colours[:, 0] - colours[:, 1]
    yellow_blue = (colours[:, 0] + colours[:, 1]) / 2.0 - colours[:, 2]
    return np.degrees(np.arctan2(yellow_blue, red_green)) % 360.0, np.hypot(red_green, yellow_blue)


def _groups(hues: NDArray[np.float64]) -> list[NDArray[np.int64]]:
    """Split angles into groups separated by gaps of at least ``MIN_HUE_GAP`` on the circle."""
    if len(hues) == 0:
        return []
    order = np.argsort(hues)
    ordered = hues[order]
    gaps = np.diff(np.concatenate((ordered, [ordered[0] + 360.0])))
    cuts = np.flatnonzero(gaps >= MIN_HUE_GAP)
    if len(cuts) == 0:
        return [order]
    rolled = np.roll(order, -(int(cuts[-1]) + 1))  # start just after the last gap
    sizes = np.diff(np.concatenate(([cuts[-1] - len(hues)], cuts)))
    groups, start = [], 0
    for size in sizes:
        groups.append(rolled[start : start + int(size)])
        start += int(size)
    return groups


def _mean_hue(hues: NDArray[np.float64]) -> float:
    angles = np.radians(hues)
    return float(np.degrees(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())) % 360.0)


def _distance(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def assign_hands(
    colours: Sequence[tuple[float, float, float]], pitches: Sequence[int], black: Sequence[bool]
) -> list[str]:
    """``right`` / ``left`` for each note, or ``unknown`` for all when the video uses one colour."""
    count = len(colours)
    if count == 0:
        return []
    values = np.asarray(colours, dtype=np.float64).reshape(count, 3)
    pitch = np.asarray(pitches, dtype=np.float64)
    is_black = np.asarray(black, dtype=bool)
    hues, chroma = hue(values)
    coloured = chroma >= MIN_CHROMA
    white = np.flatnonzero(coloured & ~is_black)
    if len(white) < 2:
        return [UNKNOWN] * count
    groups = [white[g] for g in _groups(hues[white])]
    groups = [g for g in groups if len(g) >= max(2, MIN_SHARE * len(white))]
    if len(groups) < 2:
        return [UNKNOWN] * count
    if len(groups) > 2:
        # More than two colours (fingers, voices): the two hands are the two pitch ranges.
        groups.sort(key=lambda g: float(pitch[g].mean()))
        means = [float(pitch[g].mean()) for g in groups]
        cut = int(np.argmax(np.diff(means))) + 1
        groups = [np.concatenate(groups[:cut]), np.concatenate(groups[cut:])]
    groups.sort(key=lambda g: float(pitch[g].mean()))
    low, high = groups
    hand = np.full(count, UNKNOWN, dtype=object)
    hand[low], hand[high] = LEFT, RIGHT
    white_hue = {LEFT: _mean_hue(hues[low]), RIGHT: _mean_hue(hues[high])}

    # Black keys: their shades form their own groups; pair each with the nearer hand.
    dark = np.flatnonzero(coloured & is_black)
    for group in [dark[g] for g in _groups(hues[dark])]:
        centre = _mean_hue(hues[group])
        hand[group] = min((LEFT, RIGHT), key=lambda name: _distance(centre, white_hue[name]))
    # Notes with no hue (or in a stray colour) take the hand of the nearest coloured note in pitch.
    known = np.flatnonzero(hand != UNKNOWN)
    for index in np.flatnonzero(hand == UNKNOWN):
        nearest = known[np.argmin(np.abs(pitch[known] - pitch[index]))]
        hand[index] = hand[nearest]
    return [str(h) for h in hand]


# --- the player's own hands, in a video filmed from above -----------------------------------------
#
# A filmed keyboard has the player's hands over it, and a video that draws both hands' notes in
# one colour still shows which hand plays what. Skin is redder than it is green; keys, shadows
# and the flashes of light where notes land are not. Counting the skin in each column of the
# picture, over the lower part of the keys and just below them, gives every frame a profile
# with one hump per hand, and the notes left of the gap between the humps are the left hand's.

SKIN = 15  # how much redder than green, beyond what the still picture is there, a pixel of skin is
SKIN_BRIGHT = 50  # ... and how bright its red must be: shadows have no colour to speak of


def skin_floor(samples: NDArray[np.uint8]) -> NDArray[np.int16]:
    """How much redder than green each pixel is with no hand over it: [rows, columns], from sampled frames.

    The lower fifth of what a pixel shows: a hand makes it redder, and rests on few pixels
    four fifths of the time.
    """
    excess = samples[..., 0].astype(np.int16) - samples[..., 1].astype(np.int16)
    floor: NDArray[np.int16] = np.percentile(excess, 20, axis=0).astype(np.int16)
    return floor


def skin_columns(frames: NDArray[np.uint8], floor: NDArray[np.int16]) -> NDArray[np.uint8]:
    """How many rows of each column of ``frames`` ([frames, rows, columns, 3]) show skin: [frames, columns]."""
    red = frames[..., 0].astype(np.int16)
    skin = (red - frames[..., 1].astype(np.int16) - floor[None] > SKIN) & (red > SKIN_BRIGHT)
    counted: NDArray[np.uint8] = np.minimum(skin.sum(axis=1), 255).astype(np.uint8)
    return counted


def divide_hands(skin: NDArray[np.uint8], rows: int, *, step: int = 2) -> NDArray[np.float32]:
    """The column between the two hands in every frame, from ``skin`` ([frames, columns], see ``skin_columns``).

    ``rows``: how many rows were counted, so that a column full of skin counts as one.

    The dividing column is followed through the video as the path that crosses the least skin,
    keeps skin on both sides of it where two hands show, and moves no more than it must
    (found over all frames at once, so that it also passes between two hands that touch for a
    moment, and stays where it was while one hand is out of the picture).
    """
    frames, width = skin.shape
    share = skin.astype(np.float32) / max(1, rows)
    kernel = np.ones(5, dtype=np.float32) / 5.0
    padded = np.pad(share, ((0, 0), (2, 2)), mode="edge")
    smooth = sum(padded[:, k : k + width] * kernel[k] for k in range(5))
    places = np.arange(0, width, step)
    total = np.cumsum(share, axis=1)
    left = total[:, places]
    right = total[:, -1:] - left
    both = np.minimum(left, right) / np.maximum(total[:, -1:], 1e-6)  # 0 beside the hands, up to 0.5 between them
    cost = 4.0 * smooth[:, places] - 2.0 * both  # [frames, places]
    reach = max(1, int(round(24 / step)))  # columns the divide may move from one frame to the next
    moves = np.arange(-reach, reach + 1)
    move_cost = 0.02 * step * np.abs(moves).astype(np.float32)
    count = len(places)
    best = cost[0].copy()
    came = np.zeros((frames, count), dtype=np.int16)
    index = np.arange(count)
    for frame in range(1, frames):
        candidates = np.full((len(moves), count), np.inf, dtype=np.float32)
        for k, move in enumerate(moves):
            source = index - move
            valid = (source >= 0) & (source < count)
            candidates[k, valid] = best[source[valid]] + move_cost[k]
        chosen = candidates.argmin(axis=0)
        came[frame] = moves[chosen]
        best = candidates[chosen, index] + cost[frame]
    path = np.zeros(frames, dtype=np.int64)
    path[-1] = int(np.argmin(best))
    for frame in range(frames - 1, 0, -1):
        path[frame - 1] = path[frame] - came[frame, path[frame]]
    divide: NDArray[np.float32] = (places[path] + step / 2.0).astype(np.float32)
    return divide


def hands_by_divide(columns: Sequence[float], frames: Sequence[float], divide: NDArray[np.float32]) -> list[str]:
    """``left`` or ``right`` for notes struck at ``columns`` of the keyboard in ``frames``: the side of the divide."""
    out = []
    last = len(divide) - 1
    for column, frame in zip(columns, frames, strict=True):
        first = int(np.clip(np.floor(frame) - 1, 0, last))
        around = divide[first : min(last, first + 4) + 1]  # as the key goes down, and the three frames after
        out.append(LEFT if column < float(np.median(around)) else RIGHT)
    return out

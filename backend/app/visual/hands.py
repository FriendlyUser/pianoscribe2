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


def assign_hands(colours: Sequence[tuple[float, float, float]], pitches: Sequence[int], black: Sequence[bool]) -> list[str]:
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

"""Pure helpers for ``notate_cover.py``: no model imports, so they can be unit-tested anywhere."""

from __future__ import annotations

import numpy as np

WARMUP_BARS = 2
# Bars of N beats must make up this share of a piece before bars of 2N beats are split to match.
COMMON_BAR_SHARE = 0.25


def _bar_beats(annotation: dict) -> list[list[float]]:
    """The beat times inside each bar."""
    beats = np.asarray(annotation.get("beats") or [], dtype=float)
    starts = [float(b["start"]) for b in annotation.get("bars") or []]
    out = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else np.inf
        out.append([float(b) for b in beats[(beats >= start - 1e-3) & (beats < end - 1e-3)]])
    return out


def regularize_bars(annotation: dict) -> dict:
    """Even out bar lengths where the beat tracker hesitates.

    1. The tracker sometimes marks every other bar line, mixing bars of N and 2N beats.
       Where bars of N beats are common, each bar of 2N beats becomes two.
    2. A first bar longer than the usual bar (by less than a full bar) holds an anacrusis
       followed by a complete bar; the bar line the tracker missed is put back.
    """
    bars = [dict(b) for b in annotation.get("bars") or []]
    if len(bars) < 4:
        return annotation
    inside = _bar_beats(annotation)

    def count(beats: int, unit: int) -> int:
        return sum(1 for b in bars if (b["beats"], b["unit"]) == (beats, unit))

    def split(index: int, at: int) -> None:
        bar, times = bars[index], inside[index]
        unit = bar["unit"]
        first = {**bar, "beats": at, "meter": f"{at}/{unit}"}
        second = {**bar, "start": times[at], "beats": bar["beats"] - at, "meter": f"{bar['beats'] - at}/{unit}"}
        bars[index : index + 1] = [first, second]
        inside[index : index + 1] = [times[:at], times[at:]]

    total = len(bars)
    index = 0
    while index < len(bars):
        beats, unit = bars[index]["beats"], bars[index]["unit"]
        half = beats // 2
        if (
            beats % 2 == 0
            and half >= 2
            and len(inside[index]) == beats
            and count(half, unit) >= COMMON_BAR_SHARE * total
        ):
            split(index, half)
            index += 1
        index += 1

    kinds = [(b["beats"], b["unit"]) for b in bars]
    usual, unit = max(set(kinds), key=kinds.count)
    first = bars[0]
    if first["unit"] == unit and usual < first["beats"] < 2 * usual and len(inside[0]) == first["beats"]:
        split(0, first["beats"] - usual)

    if len(bars) == len(annotation["bars"]):
        return annotation
    return {**annotation, "bars": bars, "downbeats": [b["start"] for b in bars]}


def plan_warmup(annotation: dict, first_note: float) -> tuple[float, float] | None:
    """(start, end) in seconds of the opening stretch to repeat as context, or None if too short."""
    downbeats = [float(d) for d in annotation.get("downbeats") or []]
    beats = [float(b) for b in annotation.get("beats") or []]
    if len(downbeats) < WARMUP_BARS + 2:
        return None
    if first_note >= downbeats[0] - 1e-3:
        start = max((d for d in downbeats if d <= first_note + 1e-3), default=downbeats[0])
    else:  # an anacrusis: start at the beat it begins on
        start = max((b for b in beats if b <= first_note + 1e-3), default=first_note)
    later = [d for d in downbeats if d > start + 1e-3]
    if len(later) < WARMUP_BARS + 1:
        return None
    return start, later[WARMUP_BARS - 1]


def warmed_annotation(annotation: dict, start: float, end: float) -> dict:
    """The annotation of a recording whose opening stretch [start, end) is played twice."""
    shift = end - start
    eps = 1e-3
    out = dict(annotation)
    out["duration"] = float(annotation.get("duration", 0.0)) + shift
    out["beats"] = [b for b in annotation["beats"] if b < end - eps] + [
        b + shift for b in annotation["beats"] if b >= start - eps
    ]
    opening = [d for d in annotation["downbeats"] if start - eps <= d < end - eps]
    boundary = [] if any(abs(d - start) < eps for d in annotation["downbeats"]) else [end]
    out["downbeats"] = sorted(opening + boundary + [d + shift for d in annotation["downbeats"] if d >= start - eps])
    bars = annotation.get("bars") or []
    out["bars"] = [b for b in bars if start - eps <= b["start"] < end - eps] + [
        {**b, "start": b["start"] + shift} for b in bars if b["start"] >= start - eps
    ]
    melody = annotation.get("melody") or []
    out["melody"] = [n for n in melody if n["start"] < end - eps] + [
        {**n, "start": n["start"] + shift, "end": n["end"] + shift} for n in melody if n["start"] >= start - eps
    ]
    return out


def mark_anacrusis(measures: list) -> bool:
    """Write a short first bar as a pickup in the piece's meter, not as a bar in a meter of its own.

    A one-beat anacrusis before 3/4 is conventionally printed under the 3/4 signature as an
    incomplete bar; left alone it would be engraved as a "1/4" bar followed by a change to 3/4.
    """
    if len(measures) < 2:
        return False
    first, second = measures[0], measures[1]
    if first.time_signature == second.time_signature or first.time_signature[1] != second.time_signature[1]:
        return False
    if not first.length < second.length:
        return False
    first.time_signature = second.time_signature
    return True


def warm_events(events: np.ndarray, start: int, end: int, guard: int, note_kind: int) -> np.ndarray:
    """Performance events (rows of frame, kind, value, duration) with the stretch [start, end) played twice.

    The copy holds notes only, cut off at ``end``. Notes that begin within ``guard`` frames
    of ``end`` are left out of it: they are struck slightly ahead of the next downbeat, and
    in the copy that downbeat is the first bar of the real score, where they would show up
    as an invented note. Everything from ``start`` on moves later by the length of the
    stretch; earlier events (a pedal pressed before the first note) stay where they are, so
    every pedal press keeps its release.
    """
    is_note = events[:, 1] == note_kind
    copy = events[is_note & (events[:, 0] >= start) & (events[:, 0] < end - guard)].copy()
    copy[:, 3] = np.minimum(copy[:, 3], end - copy[:, 0])
    shifted = events.copy()
    shifted[shifted[:, 0] >= start, 0] += end - start
    return np.concatenate([copy, shifted])


def drop_warmup(measures: list, count: int) -> list:
    """Remove the measures written for the warm-up copy and repair what pointed into them."""
    if count <= 0 or count >= len(measures):
        return measures
    dropped, kept = measures[:count], measures[count:]
    first = kept[0]
    for prefix in ("metronome_", "dyn_"):
        if not any(pos == 0 and name.startswith(prefix) for pos, name in first.directions):
            carried = [name for m in dropped for _, name in m.directions if name.startswith(prefix)]
            if carried:
                first.directions.insert(0, (type(first.length)(0), carried[-1]))
    if not first.tempos:
        carried_tempos = [t for m in dropped for t in m.tempos]
        if carried_tempos:
            first.tempos.append((type(first.length)(0), carried_tempos[-1][1]))
    # Slurs, pedal lines and hairpins that began in a dropped measure have no start any more.
    open_slurs = 0
    open_lines = {"pedal": False, "wedge": False}
    for measure in kept:
        for group in sorted(measure.groups, key=lambda g: (g.onset, g.staff, g.voice)):
            group.slur_stop = min(group.slur_stop, open_slurs)
            open_slurs += group.slur_start - group.slur_stop
        cleaned = []
        for pos, name in sorted(measure.directions, key=lambda d: d[0]):
            kind = "pedal" if name.startswith("pedal_") else "wedge" if name.startswith("wedge_") else None
            if kind is not None:
                if name.endswith("_stop"):
                    if not open_lines[kind]:
                        continue  # its start was in a dropped measure
                    open_lines[kind] = False
                else:
                    open_lines[kind] = True
            cleaned.append((pos, name))
        measure.directions[:] = cleaned
    return kept


def complete_meters(meters: list, measure_starts, beat_seconds: float) -> list:
    """Fill ``None`` entries: neighbour's beat unit, own length counted in beats."""
    known = [i for i, m in enumerate(meters) if m is not None]
    if not known or len(known) == len(meters):
        return list(meters)
    starts = np.asarray(measure_starts, dtype=np.float64)
    out = list(meters)
    for i, meter in enumerate(meters):
        if meter is not None:
            continue
        neighbour = min(known, key=lambda k: abs(k - i))
        beats, unit = meters[neighbour]
        if i + 1 < len(starts):
            beats = max(1, int(round((starts[i + 1] - starts[i]) / beat_seconds)))
        out[i] = (beats, unit)
    return out

"""Score a piano arrangement against a human reference of the same piece.

Two arrangements of one song never agree note for note, so the comparison
rests on what any faithful arrangement must share with the reference:

* the **melody** (top line): which pitches, in which order;
* its **rhythm**: the spacing between consecutive melody notes;
* **bar placement**: each melody note sits at the same place in its bar,
  which is what correct meter and bar lines look like from the inside;
* the **bass**: the pitch class under each bar;
* **structure**: time signature, number of bars, key signatures.

The reference melody and the candidate's top line are matched with a banded
global alignment (order-preserving, so chance pitch coincidences elsewhere in
the piece cannot count).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from eval.scoreio import Score

_EXACT, _OCTAVE, _MISMATCH = 2.0, 1.2, -1.0
_SKIP_CANDIDATE, _SKIP_REFERENCE = -0.25, -1.0
_BAND = 0.06  # alignment may drift this far (share of the piece) from proportional time


@dataclass(frozen=True)
class RefNote:
    bar: int  # 1-based
    pos: float  # quarter lengths from the bar line
    duration: float
    pitch: int
    time: float  # quarter lengths from the start
    weight: float


@dataclass
class Reference:
    title: str
    meter: tuple[int, int]
    bars: int
    bar_ql: float
    key_regions: list[tuple[int, int]]
    melody: list[RefNote]
    bass: list[tuple[int, float, int]]  # (bar, pos ql, pitch class)
    felt_bpm_range: tuple[float, float] | None
    excerpt: bool = False  # the reference covers only the opening of the piece

    @classmethod
    def load(cls, path: str | Path) -> Reference:
        raw = json.loads(Path(path).read_text())
        numerator, denominator = raw["meter"]
        bar_ql = 4.0 * numerator / denominator
        unit = 0.5  # reference positions are stored in eighth notes
        unsure = set(raw.get("unsure_bars", []))
        melody = [
            RefNote(
                int(bar),
                s * unit,
                d * unit,
                int(p),
                (int(bar) - 1) * bar_ql + s * unit,
                0.5 if int(bar) in unsure else 1.0,
            )
            for bar, notes in raw["melody"].items()
            for s, d, p in notes
        ]
        melody.sort(key=lambda n: n.time)
        bass = [(int(bar), pos * unit, int(pc)) for bar, entries in raw.get("bass", {}).items() for pos, pc in entries]
        bpm = tuple(raw["felt_bpm_range"]) if raw.get("felt_bpm_range") else None
        return cls(
            raw["title"], (numerator, denominator), int(raw["bars"]), bar_ql,
            [(r["bar"], r["fifths"]) for r in raw.get("key_regions", [])], melody, sorted(bass), bpm,
            bool(raw.get("excerpt_only", False)),
        )  # fmt: skip


@dataclass
class ArrangementScores:
    melody_recall: float  # reference melody notes found, in order (any octave)
    melody_precision: float  # candidate top-line notes that belong to the reference melody
    melody_f1: float
    melody_octave_exact: float  # of the found notes, share in the reference octave
    rhythm_agreement: float  # consecutive found notes with the reference spacing
    bar_placement: float  # found notes at the reference position inside their bar
    bass_accuracy: float
    meter_correct: bool
    bar_count_ratio: float
    key_accuracy: float
    notes_per_bar: float
    overall: float

    def as_dict(self) -> dict[str, float | bool]:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def top_line(score: Score) -> list[tuple[float, int, float, int]]:
    """(start, pitch, position-in-bar share, bar) of the highest note at each onset of the upper staff."""
    upper = [n for n in score.notes if n.staff == 1] or score.notes
    best: dict[Fraction, object] = {}
    for note in upper:
        if note.start not in best or note.pitch > best[note.start].pitch:  # type: ignore[attr-defined]
            best[note.start] = note
    line = []
    for start in sorted(best):
        note = best[start]
        bar = score.bars[note.bar]  # type: ignore[attr-defined]
        nominal = Fraction(4 * bar.meter[0], bar.meter[1])
        # A pickup bar is the end of a bar: its notes sit against the bar line that follows.
        lead = nominal - bar.length if bar.index == 0 and 0 < bar.length < nominal else 0
        line.append((float(start), note.pitch, float((note.pos + lead) / nominal), note.bar))  # type: ignore[attr-defined]
    return line


def _align(
    reference: list[RefNote],
    line: list[tuple[float, int, float, int]],
    ref_total: float,
    cand_total: float,
    cand_offset: float = 0.0,
):
    """Banded Needleman-Wunsch; returns matched (reference index, candidate index) pairs.

    ``cand_offset`` (quarter lengths) is the candidate time that corresponds to reference time 0.
    """
    n, m = len(reference), len(line)
    ref_time = np.array([r.time for r in reference]) / ref_total
    cand_time = (np.array([c[0] for c in line]) - cand_offset) / cand_total
    score = np.full((n + 1, m + 1), -np.inf)
    move = np.zeros((n + 1, m + 1), dtype=np.int8)
    score[0, :] = np.arange(m + 1) * _SKIP_CANDIDATE
    move[0, 1:] = 2
    for i in range(1, n + 1):
        lo = int(np.searchsorted(cand_time, ref_time[i - 1] - _BAND)) + 1
        hi = int(np.searchsorted(cand_time, ref_time[i - 1] + _BAND)) + 1
        lo, hi = max(1, lo), min(m, hi)
        if lo - 1 >= 0 and np.isfinite(score[i - 1, lo - 1]):
            score[i, lo - 1] = score[i - 1, lo - 1] + _SKIP_REFERENCE
            move[i, lo - 1] = 1
        for j in range(lo, hi + 1):
            ref_pitch, cand_pitch = reference[i - 1].pitch, line[j - 1][1]
            if ref_pitch == cand_pitch:
                gain = _EXACT
            elif (ref_pitch - cand_pitch) % 12 == 0:
                gain = _OCTAVE
            else:
                gain = _MISMATCH
            options = (score[i - 1, j - 1] + gain, score[i - 1, j] + _SKIP_REFERENCE, score[i, j - 1] + _SKIP_CANDIDATE)
            k = int(np.argmax(options))
            score[i, j], move[i, j] = options[k], k
    j = int(np.argmax(score[n]))
    i = n
    pairs: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        step = move[i, j]
        if step == 0:
            if (reference[i - 1].pitch - line[j - 1][1]) % 12 == 0:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def _best_scale(reference: list[RefNote], line, ref_total: float, cand_total: float) -> float:
    """Candidates may be written in note values twice as long or short; that is not an error."""
    ratio = cand_total / ref_total
    return min((0.5, 1.0, 2.0), key=lambda s: abs(np.log(ratio / s)))


def evaluate(reference: Reference, score: Score) -> ArrangementScores:
    line = top_line(score)
    ref_total = reference.bars * reference.bar_ql
    cand_total = float(score.length) or 1.0
    if reference.excerpt:
        # Only the opening is annotated: compare it with the opening of the candidate,
        # trying each plausible note-value scale and keeping the best reading.
        # The start is anchored on the first notes rather than on the first bar line: how a
        # score pads an anacrusis is judged by bar_placement, not by losing the melody.
        best: tuple[float, float, list, list] | None = None
        first_ref = reference.melody[0].time if reference.melody else 0.0
        anchors = sorted({entry[0] for entry in line})[:4]
        for candidate_scale in (0.5, 1.0, 2.0):
            span = ref_total * candidate_scale
            for anchor in [None, *anchors]:
                offset = 0.0 if anchor is None else anchor - first_ref * candidate_scale
                opening = [entry for entry in line if entry[0] - offset < span * 1.3 + 2.0]
                if not opening:
                    continue
                matched = _align(reference.melody, opening, ref_total, span, offset)
                quality = len(matched) - 0.01 * abs(np.log2(candidate_scale)) - (0.0 if anchor is None else 0.001)
                if best is None or quality > best[0]:
                    best = (quality, candidate_scale, matched, opening)
        if best is None:
            scale, pairs = 1.0, []
        else:
            _, scale, pairs, line = best
        cand_total = ref_total * scale
    else:
        scale = _best_scale(reference.melody, line, ref_total, cand_total)
        pairs = _align(reference.melody, line, ref_total, cand_total)

    weights = np.array([r.weight for r in reference.melody])
    found = np.zeros(len(reference.melody), dtype=bool)
    for i, _ in pairs:
        found[i] = True
    recall = float((weights * found).sum() / weights.sum())

    # Precision over the stretch of the candidate that the reference covers.
    if pairs:
        first, last = pairs[0][1], pairs[-1][1]
        span = last - first + 1
        precision = len({j for _, j in pairs}) / span
    else:
        precision = 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    exact = float(np.mean([reference.melody[i].pitch == line[j][1] for i, j in pairs])) if pairs else 0.0

    spacing_ok, spacing_n = 0, 0
    for (i0, j0), (i1, j1) in zip(pairs, pairs[1:], strict=False):
        if i1 != i0 + 1:
            continue  # only directly consecutive reference notes
        want = reference.melody[i1].time - reference.melody[i0].time
        got = (line[j1][0] - line[j0][0]) / scale
        if want <= 0:
            continue
        spacing_n += 1
        spacing_ok += abs(got - want) <= max(0.13, 0.2 * want)
    rhythm = spacing_ok / spacing_n if spacing_n else 0.0

    placed = [abs(reference.melody[i].pos / reference.bar_ql - line[j][2]) <= 0.045 for i, j in pairs]
    placement = float(np.mean(placed)) if placed else 0.0

    # Bass: map each reference bar to candidate time through the nearest matched melody notes.
    bass_ok, bass_n = 0, 0
    if pairs:
        ref_times = np.array([reference.melody[i].time for i, _ in pairs])
        cand_times = np.array([line[j][0] for _, j in pairs])
        lower = [n for n in score.notes if n.staff != 1] or score.notes
        lower_starts = np.array([float(n.start) for n in lower])
        for bar, pos, pitch_class in reference.bass:
            target = (bar - 1) * reference.bar_ql + pos
            if target < ref_times[0] - reference.bar_ql or target > ref_times[-1] + reference.bar_ql:
                continue
            centre = float(np.interp(target, ref_times, cand_times))
            window = 0.26 * reference.bar_ql * scale
            nearby = [lower[k] for k in np.flatnonzero(np.abs(lower_starts - centre) <= window)]
            bass_n += 1
            if nearby and min(nearby, key=lambda n: n.pitch).pitch % 12 == pitch_class:
                bass_ok += 1
    bass = bass_ok / bass_n if bass_n else 0.0

    meters = [b.meter for b in score.bars]
    common = max(set(meters), key=meters.count) if meters else (0, 0)
    meter_ok = common == reference.meter
    bar_ratio = 1.0 if reference.excerpt else (cand_total / scale / reference.bar_ql) / reference.bars
    key_hits = []
    for index, (bar, fifths) in enumerate(reference.key_regions):
        end = reference.key_regions[index + 1][0] if index + 1 < len(reference.key_regions) else reference.bars + 1
        middle = ((bar + end) / 2 - 1) * reference.bar_ql * scale * (cand_total / scale / ref_total)
        at = next((b for b in reversed(score.bars) if float(b.start) <= middle), score.bars[0] if score.bars else None)
        key_hits.append((end - bar, at is not None and at.fifths == fifths))
    key_accuracy = sum(length for length, ok in key_hits if ok) / max(1, sum(length for length, _ in key_hits))

    overall = float(0.35 * f1 + 0.2 * rhythm + 0.2 * placement + 0.15 * bass + 0.05 * meter_ok + 0.05 * key_accuracy)
    return ArrangementScores(
        melody_recall=recall,
        melody_precision=precision,
        melody_f1=f1,
        melody_octave_exact=exact,
        rhythm_agreement=rhythm,
        bar_placement=placement,
        bass_accuracy=bass,
        meter_correct=bool(meter_ok),
        bar_count_ratio=float(bar_ratio),
        key_accuracy=float(key_accuracy),
        notes_per_bar=len(score.notes) / max(1, len(score.bars)),
        overall=overall,
    )

"""Write notes read from a piano-tutorial video as MusicXML, by rule. Runs inside the tsumugi-piano-cover environment.

    python notate_exact.py --notes visual.json --annotation sheetsage.json --keys keys.json --out score.musicxml

The notes' pitches, onsets, lengths and hands are known exactly (``app.visual``), so the
score model is not asked to write them: ``exact_score.plan_score`` places them on the
beat grid and this driver engraves the plan with the model repository's own MusicXML
writer (rests, beams, tuplet brackets and accidentals are its rules). No network is run.

From the audio models come only what the picture cannot show: the beats and bar lines
(``--annotation``, SheetSage2-Refined) and the key (``--keys``, its key tracker).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from exact_score import plan_score
from score_fixes import bar_signatures, regularize_bars

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)


def signature_from_notes(pitches: list[int]) -> int:
    """Key signature (sharps +, flats -) whose scale holds most of the notes; fewest accidentals on a tie."""
    classes = Counter(pitch % 12 for pitch in pitches)

    def inside(fifths: int) -> int:
        tonic = (7 * fifths) % 12
        return sum(count for pitch_class, count in classes.items() if (pitch_class - tonic) % 12 in MAJOR_SCALE)

    return max(range(-6, 7), key=lambda fifths: (inside(fifths), -abs(fifths)))


def outside(pitches: list[int], fifths: int) -> int:
    """How many of ``pitches`` are outside the major scale of the key signature ``fifths``."""
    tonic = (7 * fifths) % 12
    return sum((pitch - tonic) % 12 not in MAJOR_SCALE for pitch in pitches)


KEY_CHANGE = 12.0  # notes outside the scale that a change of key signature must save to be worth writing
HEARD = 0.5  # what a bar costs in a signature other than the one the key tracker heard there
PLAIN = 0.05  # ... for every sharp or flat of its signature: of two that fit alike, unheard, the plainer one
LEAST_SIGNATURE_BARS = 4  # bars a key signature lasts at the least


def signatures_by_the_notes(pitches: list[list[int]], heard: list[int]) -> list[int]:
    """The key signature of each bar: the ones that take the fewest accidentals, a change of signature counting too.

    ``pitches``: the notes of each bar. ``heard``: the signature the key tracker heard for it.

    The tracker hears where the harmony leans, a few bars late or early, and a passage of some
    bars in another key not at all. But a key signature is there to save accidentals, and here
    the notes are known exactly: so the signatures are chosen that leave the fewest notes
    outside their scales, a change of signature counting as ``KEY_CHANGE`` such notes (a bar or
    two of another harmony are written with accidentals, as a printed score has them). What was
    heard decides only between signatures the notes fit alike, and after that the plainer one. A
    signature lasts ``LEAST_SIGNATURE_BARS`` bars or it is not written.
    """
    if not pitches:
        return []
    states = list(range(-6, 7))

    def cost(bar: int, signature: int) -> float:
        return outside(pitches[bar], signature) + (0.0 if signature == heard[bar] else HEARD) + PLAIN * abs(signature)

    def cheapest_path(first: int, last: int, allowed: list[int]) -> list[int]:
        """Signatures of the bars ``first`` .. ``last`` (cheapest in all, each from ``allowed``)."""
        total = [[cost(first, signature) for signature in allowed]]
        back: list[list[int]] = []
        for bar in range(first + 1, last + 1):
            before = total[-1]
            least = min(range(len(allowed)), key=lambda i: before[i])
            row, origin = [], []
            for i, signature in enumerate(allowed):
                stay, move = before[i], before[least] + KEY_CHANGE
                origin.append(i if stay <= move else least)
                row.append(min(stay, move) + cost(bar, signature))
            total.append(row)
            back.append(origin)
        state = min(range(len(allowed)), key=lambda i: total[-1][i])
        path = [state]
        for origin in reversed(back):
            state = origin[state]
            path.append(state)
        return [allowed[i] for i in path[::-1]]

    out = cheapest_path(0, len(pitches) - 1, states)
    while True:
        # A signature for a bar or two is no signature: its bars go to whichever neighbour suits their notes better.
        runs: list[tuple[int, int]] = []
        for bar in range(len(out)):
            if runs and out[runs[-1][0]] == out[bar]:
                runs[-1] = (runs[-1][0], bar)
            else:
                runs.append((bar, bar))
        short = [run for run in runs if run[1] - run[0] + 1 < LEAST_SIGNATURE_BARS]
        if len(runs) < 2 or not short:
            return out
        first, last = min(short, key=lambda run: run[1] - run[0])
        around = [out[first - 1]] if first > 0 else []
        around += [out[last + 1]] if last + 1 < len(out) else []
        best = min(around, key=lambda signature: sum(cost(bar, signature) for bar in range(first, last + 1)))
        out[first : last + 1] = [best] * (last - first + 1)


def mark_tempos(measures: list, starts: list[float], end: float, steady: object | None) -> list[tuple[int, int]] | None:
    """Metronome marks from the bars' own lengths, where the notes keep a steady beat; the marks set (bar, tempo).

    A beat fitted to the notes gives each bar's tempo to a tenth of a beat a minute, so a tempo
    is marked as it is (106, not the 105 a metronome has) and each lasting change of it is
    marked where it happens. A bar or two slower than what went before, between marks, is a
    ritardando. Returns None, having done nothing, when the notes keep no steady beat.
    """
    stretches = getattr(steady, "stretches", None) or []
    drawn = getattr(steady, "drawn", None)  # the bars are the ones the roll draws: each as long as it is drawn
    edges = [*starts, end]
    if getattr(steady, "followed", None):
        lengths = zip(measures, edges, edges[1:], strict=False)
        return mark_played(measures, [60.0 * float(measure.length) / (b - a) for measure, a, b in lengths])
    if not stretches and not drawn:
        return None
    fine = bool(stretches) and float(np.median([s["scatter_ms"] for s in stretches])) <= 5.0
    apart, lasting = (0.012, 2) if fine else (0.03, 4)
    tempos = []
    sure = []  # per bar: its tempo is that of a beat fitted to the notes, not the length of a tracked bar
    for k, measure in enumerate(measures):
        middle = (edges[k] + edges[k + 1]) / 2.0
        inside = [s for s in stretches if s["start"] <= middle <= s["end"]]
        if inside:
            # The fitted beat is the one the bar is counted in: an eighth in 6/8, where a quarter lasts two.
            tempos.append(float(inside[0]["bpm"]) * 4.0 / measure.time_signature[1])
        else:
            tempos.append(60.0 * float(measure.length) / (edges[k + 1] - edges[k]))
        sure.append(bool(inside) or bool(drawn))
    runs: list[tuple[int, int, float]] = []  # first bar, bar after the last, tempo
    first = 0
    for k in range(1, len(tempos) + 1):
        if k == len(tempos) or abs(tempos[k] / float(np.median(tempos[first:k])) - 1.0) > apart:
            runs.append((first, k, float(np.median(tempos[first:k]))))
            first = k
    for measure in measures:
        measure.directions = [d for d in measure.directions if not d[1].startswith(("metronome_", "tempo_"))]
        measure.tempos = []
    long = [run for run in runs if run[1] - run[0] >= lasting]
    shown = round((long[0][2] if long else float(np.median(tempos))) / _counted(measures[0]))
    measures[0].directions.insert(0, (Fraction(0), _metronome(measures[0], shown)))
    marks = [(0, shown)]
    steady_tempo = float(shown)
    slowed = False
    closing = len(measures) - 2  # from here on a ritardando is the close of the piece: nothing returns to tempo
    for a, b, tempo in runs:
        measures[a].tempos = [(Fraction(0), tempo)]  # what is played, bar by bar
        if b - a >= lasting:
            # A tracked bar's length says its tempo to a few beats a minute, no better: that is no new tempo.
            # Nor is a tempo no further from the one marked than bars of one tempo are from each other.
            near = apart if all(sure[a:b]) else 0.04
            counted = tempo / _counted(measures[a])
            same = round(counted) == shown or abs(counted / shown - 1.0) <= near
            if not same and a > 0:
                shown = round(counted)
                measures[a].directions.insert(0, (Fraction(0), _metronome(measures[a], shown)))
                marks.append((a, shown))
            elif slowed and a < closing:
                measures[a].directions.insert(0, (Fraction(0), "tempo_a_tempo"))  # the tempo it slowed from
            steady_tempo = tempo
            slowed = False
        elif a > 0 and tempo < 0.96 * steady_tempo and b < len(measures):
            if not slowed:
                measures[a].directions.insert(0, (Fraction(0), "tempo_rit"))
            slowed = True
        elif slowed and tempo >= 0.96 * steady_tempo:
            if a < closing:
                measures[a].directions.insert(0, (Fraction(0), "tempo_a_tempo"))
            slowed = False
    return marks


def _counted(measure: object) -> float:
    """Quarter notes in the beat a metronome mark counts in ``measure``: a dotted quarter in 6/8, 9/8 and 12/8."""
    beats, unit = getattr(measure, "time_signature", (4, 4))
    return 1.5 if unit == 8 and beats % 3 == 0 else 1.0


def _metronome(measure: object, shown: int) -> str:
    return f"metronome_quarter{'.' if _counted(measure) > 1.0 else ''}_{shown}"


SLOWED = 0.94  # of the tempo of the bars before it: a bar played slower than this is marked as a ritardando


def mark_played(measures: list, tempos: list[float]) -> list[tuple[int, int]]:
    """Tempo marks for playing that keeps no click's time (``steady_beats.follow``); the marks set (bar, tempo).

    Each bar takes a little longer or shorter than the last, and none of that is a change of
    tempo: one metronome mark stands at the head, the tempo of the opening bars. A bar plainly
    slower than the eight before it (broadened before a new verse, or at the close) is marked
    *rit.*, and *a tempo* where the playing is back to what it was. Every bar's own tempo is kept
    for the sound, so that the score played back keeps the player's time.
    """
    for measure in measures:
        measure.directions = [d for d in measure.directions if not d[1].startswith(("metronome_", "tempo_"))]
        measure.tempos = []
    shown = round(float(np.median(tempos[:16])) / _counted(measures[0]))
    measures[0].directions.insert(0, (Fraction(0), _metronome(measures[0], shown)))
    slowed_from: float | None = None
    for k, tempo in enumerate(tempos):
        measures[k].tempos = [(Fraction(0), tempo)]
        if k == 0:
            continue
        if slowed_from is None:
            before = float(np.median(tempos[max(0, k - 8) : k]))
            if tempo < SLOWED * before and k + 1 < len(tempos):
                measures[k].directions.insert(0, (Fraction(0), "tempo_rit"))
                slowed_from = before
        elif tempo >= 0.96 * slowed_from:
            if k < len(tempos) - 2:  # after that a ritardando is the close of the piece
                measures[k].directions.insert(0, (Fraction(0), "tempo_a_tempo"))
            slowed_from = None
    return [(0, shown)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notes", required=True, help="notes read from the video (app.visual: visual.json)")
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--keys", default=None, help="key tracker output (sheetsage2_refined.keys)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    from piano_perf2score.beat_tempo import apply_bar_tempos
    from piano_perf2score.staves import readable_staves
    from piano_perf2score.tidy import chromatic_spelling, spell_pitch_class
    from piano_score.musicxml import write_musicxml
    from piano_score.score import Duration, Group, Measure, Note

    reading = json.loads(Path(args.notes).read_text(encoding="utf-8"))
    notes = reading["notes"]
    # Whether the notes ring on after their keys are released, as the reader heard it (app.visual.audio.rings_on).
    let_ring = bool(((reading.get("facts") or {}).get("notes") or {}).get("let_ring"))
    annotation = regularize_bars(json.loads(Path(args.annotation).read_text(encoding="utf-8")))
    plan = plan_score(notes, annotation, let_ring=let_ring, lines=[float(t) for t in reading.get("lines") or []])

    starts = [bar.seconds for bar in plan.bars]
    overall = signature_from_notes([int(n["pitch"]) for n in notes])
    heard: list[int | None] = [None] * len(plan.bars)
    if args.keys:
        tracked = json.loads(Path(args.keys).read_text(encoding="utf-8")).get("keys") or []
        if tracked:
            # As heard, bar by bar: how long a key must last to be written is decided with the notes, below.
            heard = bar_signatures(tracked, starts, min_bars=1)
    sounding: list[list[int]] = [[] for _ in plan.bars]
    for value in plan.written:
        sounding[value.bar] += value.pitches
    as_heard = [overall if signature is None else signature for signature in heard]
    signatures = signatures_by_the_notes(sounding, as_heard)

    def changes(per_bar: list[int]) -> set[int]:
        return {bar for bar in range(1, len(per_bar)) if per_bar[bar] != per_bar[bar - 1]}

    key_changes_not_written = len(changes(as_heard) - changes(signatures))
    key_changes_from_the_notes = len(changes(signatures) - changes(as_heard))

    measures = [
        Measure(
            time_signature=(bar.beats, bar.unit),
            key=signature,
            clefs=("G2", "F4"),
            length=Fraction(bar.ticks) * bar.quarter,
        )
        for bar, signature in zip(plan.bars, signatures, strict=True)
    ]
    for value in plan.written:
        measure = measures[value.bar]
        measure.groups.append(
            Group(
                onset=value.onset,
                staff=value.staff,
                voice=value.voice,
                duration=Duration(value.kind, value.dots, (3, 2) if value.triplet else None),
                notes=[
                    Note(pitch, spell_pitch_class(pitch % 12, measure.key), tie=value.tied) for pitch in value.pitches
                ],
                articulations=("arpeggiate",) if value.spread else (),
            )
        )
    chromatic_spelling(measures)
    readable_staves(measures)
    last = plan.bars[-1]
    end = (
        last.seconds + (starts[-1] - starts[-2]) * last.ticks / plan.bars[-2].ticks
        if len(starts) > 1
        else last.seconds + 2.0
    )
    marks = mark_tempos(measures, starts, end, plan.steady)
    if marks is None:
        apply_bar_tempos(measures, SimpleNamespace(bar_seconds=np.array([*starts, end])), 0)
    write_musicxml(measures, Path(args.out), title=args.title)
    facts = {
        "notes": plan.notes,
        "bars": len(plan.bars),
        "beats_moved_onto_notes": plan.moved_beats,
        "notes_off_the_grid": plan.off_grid,
        "notes_cut_short": plan.cut_short,
        "voices": plan.voices,
        "notes_on_the_other_staff": plan.shared,
        "let_ring": let_ring,
        "notes_let_ring": plan.rung,
        "octaves_rolled": plan.rolled,
        "chords_spread": plan.spread,
        "key_signatures": [{"bar": bar + 1, "fifths": signatures[bar]} for bar in sorted({0} | changes(signatures))],
        "key_changes_not_written": key_changes_not_written,
        "key_changes_from_the_notes": key_changes_from_the_notes,
        "beats_put_back": plan.beats_put_back,
        "tempo_marks": [
            {"bar": bar + 1, "dotted_quarter" if _counted(measures[bar]) > 1.0 else "quarter": tempo}
            for bar, tempo in marks or []
        ],
        "steady_beat": {
            "beats_from_the_notes": sum(plan.steady.fitted),
            "beats_from_the_tracker": len(plan.steady.fitted) - sum(plan.steady.fitted),
            "bars_counted": plan.steady.regular_bars,
            "bars_as_tracked": plan.steady.tracked_bars,
            "stretches_on_another_beat_than_tracked": plan.steady.rephased,
            "stretches": plan.steady.stretches,
            "followed": plan.steady.followed,
            "drawn": plan.steady.drawn,
        }
        if plan.steady is not None
        else None,
    }
    print(f"@@facts {json.dumps(facts)}", flush=True)
    print(f"{len(measures)} measures, {plan.notes} notes -> {args.out}")


if __name__ == "__main__":
    main()

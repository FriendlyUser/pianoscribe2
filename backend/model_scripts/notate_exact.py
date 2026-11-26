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


def mark_tempos(measures: list, starts: list[float], end: float, steady: object | None) -> list[tuple[int, int]] | None:
    """Metronome marks from the bars' own lengths, where the notes keep a steady beat; the marks set (bar, tempo).

    A beat fitted to the notes gives each bar's tempo to a tenth of a beat a minute, so a tempo
    is marked as it is (106, not the 105 a metronome has) and each lasting change of it is
    marked where it happens. A bar or two slower than what went before, between marks, is a
    ritardando. Returns None, having done nothing, when the notes keep no steady beat.
    """
    stretches = getattr(steady, "stretches", None) or []
    if not stretches:
        return None
    fine = float(np.median([s["scatter_ms"] for s in stretches])) <= 5.0
    apart, lasting = (0.012, 2) if fine else (0.03, 4)
    edges = [*starts, end]
    tempos = []
    sure = []  # per bar: its tempo is that of a beat fitted to the notes, not the length of a tracked bar
    for k, measure in enumerate(measures):
        middle = (edges[k] + edges[k + 1]) / 2.0
        inside = [s for s in stretches if s["start"] <= middle <= s["end"]]
        tempos.append(float(inside[0]["bpm"]) if inside else 60.0 * float(measure.length) / (edges[k + 1] - edges[k]))
        sure.append(bool(inside))
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
    shown = round(long[0][2] if long else float(np.median(tempos)))
    measures[0].directions.insert(0, (Fraction(0), f"metronome_quarter_{shown}"))
    marks = [(0, shown)]
    steady_tempo = float(shown)
    slowed = False
    for a, b, tempo in runs:
        measures[a].tempos = [(Fraction(0), tempo)]  # what is played, bar by bar
        if b - a >= lasting:
            # A tracked bar's length says its tempo to a few beats a minute, no better: that is no new tempo.
            same = round(tempo) == shown or (not all(sure[a:b]) and abs(tempo / shown - 1.0) <= 0.04)
            if not same and a > 0:
                shown = round(tempo)
                measures[a].directions.insert(0, (Fraction(0), f"metronome_quarter_{shown}"))
                marks.append((a, shown))
            elif slowed:
                measures[a].directions.insert(0, (Fraction(0), "tempo_a_tempo"))  # the tempo it slowed from
            steady_tempo = tempo
            slowed = False
        elif a > 0 and tempo < 0.96 * steady_tempo and b < len(measures):
            if not slowed:
                measures[a].directions.insert(0, (Fraction(0), "tempo_rit"))
            slowed = True
        elif slowed and tempo >= 0.96 * steady_tempo:
            measures[a].directions.insert(0, (Fraction(0), "tempo_a_tempo"))
            slowed = False
    return marks


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
    plan = plan_score(notes, annotation, let_ring=let_ring)

    starts = [bar.seconds for bar in plan.bars]
    overall = signature_from_notes([int(n["pitch"]) for n in notes])
    signatures: list[int | None] = [None] * len(plan.bars)
    if args.keys:
        tracked = json.loads(Path(args.keys).read_text(encoding="utf-8")).get("keys") or []
        if tracked:
            signatures = bar_signatures(tracked, starts)

    measures = [
        Measure(
            time_signature=(bar.beats, bar.unit),
            key=overall if signature is None else signature,
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
        "tempo_marks": [{"bar": bar + 1, "quarter": tempo} for bar, tempo in marks or []],
        "steady_beat": {
            "beats_from_the_notes": sum(plan.steady.fitted),
            "beats_from_the_tracker": len(plan.steady.fitted) - sum(plan.steady.fitted),
            "bars_counted": plan.steady.regular_bars,
            "bars_as_tracked": plan.steady.tracked_bars,
            "stretches_on_another_beat_than_tracked": plan.steady.rephased,
            "stretches": plan.steady.stretches,
        }
        if plan.steady is not None
        else None,
    }
    print(f"@@facts {json.dumps(facts)}", flush=True)
    print(f"{len(measures)} measures, {plan.notes} notes -> {args.out}")


if __name__ == "__main__":
    main()

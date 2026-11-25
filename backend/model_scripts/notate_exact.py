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

    notes = json.loads(Path(args.notes).read_text(encoding="utf-8"))["notes"]
    annotation = regularize_bars(json.loads(Path(args.annotation).read_text(encoding="utf-8")))
    plan = plan_score(notes, annotation)

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
    apply_bar_tempos(measures, SimpleNamespace(bar_seconds=np.array([*starts, end])), 0)
    write_musicxml(measures, Path(args.out), title=args.title)
    facts = {
        "notes": plan.notes,
        "bars": len(plan.bars),
        "beats_moved_onto_notes": plan.moved_beats,
        "notes_off_the_grid": plan.off_grid,
        "notes_cut_short": plan.cut_short,
        "voices": plan.voices,
    }
    print(f"@@facts {json.dumps(facts)}", flush=True)
    print(f"{len(measures)} measures, {plan.notes} notes -> {args.out}")


if __name__ == "__main__":
    main()

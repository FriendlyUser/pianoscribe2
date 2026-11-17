"""Write a generated piano cover as MusicXML. Runs inside the tsumugi-piano-cover environment.

    python notate_cover.py --checkpoint REPO --cover cover.mid --annotation sheetsage.json \
        --chords source.mid --out score.musicxml

This is ``piano_perf2score.generate.score_from_cover`` with three additions.

1. **Anticipated first chord.** A generated performance often strikes its first
   notes a few hundredths of a second before the first downbeat. Read literally,
   that is an anacrusis bar holding nothing but those notes, and the score model
   then stumbles through the opening bars. Notes that precede the first downbeat by
   less than half a beat are moved onto it.
2. **Meter of bars the beat tracker did not label.** The score model takes each
   bar's time signature from the tracker's bars, but only when *every* bar has one.
   A real anacrusis bar lies before the tracker's first bar and has none, so the
   model would fall back to guessing the beat unit and meter for the whole piece
   (a 6/8 song comes out in 3/4). Such bars inherit the beat unit of their
   neighbours and get their own length in beats.
3. **Warm-up bars.** The score model writes bar by bar from left to right, and with
   nothing before it the first full bar often comes out at the wrong note values
   (eighths written as sixteenths, the bar padded with invented notes); from the
   third bar on it is reliable. So the opening of the performance is played to it
   twice: a copy of the first bars is put in front as context, and the measures
   written for that copy are discarded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from score_fixes import (
    complete_meters,
    drop_warmup,
    mark_anacrusis,
    plan_warmup,
    regularize_bars,
    warm_events,
    warmed_annotation,
)


def settle_anticipation(cover: Path, annotation: dict, out: Path) -> Path:
    """Move notes that anticipate the first downbeat by under half a beat onto it."""
    from symusic import Score

    downbeats, beats = annotation.get("downbeats") or [], annotation.get("beats") or []
    if not downbeats or len(beats) < 2:
        return cover
    first = float(downbeats[0])
    half_beat = 0.5 * float(np.median(np.diff(beats)))
    score = Score(str(cover)).to("second")
    starts = [note.time for track in score.tracks for note in track.notes]
    if not starts or not (first - half_beat <= min(starts) < first):
        return cover
    if any(t < first - half_beat for t in starts):
        return cover  # a genuine anacrusis: leave it alone
    for track in score.tracks:
        for note in track.notes:
            if note.time < first:
                note.duration = max(0.05, note.duration - (first - note.time))
                note.time = first
        for pedal in getattr(track, "pedals", []):
            if pedal.time < first:
                pedal.duration = max(0.05, pedal.duration - (first - pedal.time))
                pedal.time = first
    score.to("tick").dump_midi(str(out))
    return out


def install_warmup(start: float, end: float, guard: float) -> None:
    """Make the performance reader return the opening stretch twice (notes only in the copy).

    Notes that begin within ``guard`` seconds of the end of the stretch are left out of the
    copy: they are played slightly ahead of the next downbeat, and in the copy that downbeat
    is the first bar of the real score, where they would appear as an invented note.
    """
    from piano_ar.tokenizer import KIND_NOTE, PianoTokenizer, sort_events

    original = PianoTokenizer.midi_to_events

    def midi_to_events(self, path, trim: bool = True):  # noqa: ANN001, ANN202
        events, _ = original(self, path, trim=False)
        rate = self.config.frame_rate
        merged = sort_events(
            warm_events(events, int(round(start * rate)), int(round(end * rate)), int(round(guard * rate)), KIND_NOTE)
        )
        end_frame = int((merged[:, 0] + merged[:, 3]).max()) if len(merged) else 0
        if trim and len(merged):
            merged[:, 0] -= merged[:, 0].min()
        return merged, end_frame

    PianoTokenizer.midi_to_events = midi_to_events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cover", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--chords", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    from piano_perf2score.beat_input import BEAT_SECONDS
    from piano_perf2score.beat_tempo import apply_bar_tempos
    from piano_perf2score.chords import place_chords, read_chord_markers
    from piano_perf2score.generate import beat_performance, transcribe_on_beats
    from piano_perf2score.hub import load_perf2score
    from piano_perf2score.staves import readable_staves
    from piano_perf2score.tidy import tidy_score
    from piano_score.musicxml import write_musicxml

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, banned = load_perf2score(args.checkpoint, device)
    annotation = json.loads(Path(args.annotation).read_text(encoding="utf-8"))
    annotation = regularize_bars(annotation)
    chords = read_chord_markers(args.chords or args.cover)
    cover = settle_anticipation(Path(args.cover), annotation, Path(args.out).with_suffix(".settled.mid"))

    from symusic import Score

    first_note = min(
        (n.time for track in Score(str(cover)).to("second").tracks if not track.is_drum for n in track.notes),
        default=0.0,
    )
    warmup = None if args.no_warmup else plan_warmup(annotation, float(first_note))
    if warmup is not None:
        start, end = warmup
        beat = float(np.median(np.diff(annotation["beats"]))) if len(annotation.get("beats") or []) > 1 else 0.4
        install_warmup(start, end, guard=0.25 * beat)
        annotation = warmed_annotation(annotation, start, end)
        chords = [(t + (end - start), text) for t, text in chords]

    performance = beat_performance(cover, annotation, chords)
    performance.meters = complete_meters(list(performance.meters), performance.measure_starts, BEAT_SECONDS)
    warmup_measures = 0
    if warmup is not None:
        warmup_measures = int(np.sum(np.asarray(performance.bar_seconds[:-1]) < warmup[1] - 1e-3))

    def on_measure(done: int, total: int) -> None:
        print(f"@@progress {json.dumps({'stage': 'notate', 'done': int(done), 'total': int(total)})}", flush=True)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        patches = transcribe_on_beats(
            model, tokenizer, performance, temperature=0.0, banned=banned, on_measure=on_measure
        )
    measures = tokenizer.decode(patches)[0]
    tidied = tidy_score(measures)
    readable_staves(measures)
    apply_bar_tempos(measures, performance, tidied.rests)
    if performance.chords:
        place_chords(measures, performance, performance.chords, tidied.rests)
    measures = drop_warmup(measures, warmup_measures)
    mark_anacrusis(measures)
    write_musicxml(measures, Path(args.out), title=args.title)
    print(f"{len(measures)} measures -> {args.out} (warm-up measures dropped: {warmup_measures})")


if __name__ == "__main__":
    main()

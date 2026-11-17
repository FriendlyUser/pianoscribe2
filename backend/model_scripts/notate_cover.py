"""Write a generated piano cover as MusicXML. Runs inside the tsumugi-piano-cover environment.

    python notate_cover.py --checkpoint REPO --cover cover.mid --annotation sheetsage.json \
        --chords source.mid --out score.musicxml

This is ``piano_perf2score.generate.score_from_cover`` with additions, each a small
function in ``score_fixes.py`` that grounded tests showed to be needed.

1. **Lead-in before the first downbeat.** A generated performance often strikes a bass
   note or chord up to a beat ahead of the first downbeat. Read literally that is a
   pickup bar, which the score model writes at double speed, swallowing the start of
   bar 1. When the melody itself starts on the downbeat, the lead-in is moved onto it.
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
4. **Key signatures from the key tracker** (``--keys``). The score model infers keys
   from the notes it is given and misses or misplaces key changes; SheetSage2-Refined's
   key tracker hears them in the recording.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from score_fixes import (
    bar_signatures,
    complete_meters,
    correct_regions,
    drop_warmup,
    follow_tracker,
    lead_in_to_settle,
    mark_anacrusis,
    plan_warmup,
    regularize_bars,
    warm_events,
    warmed_annotation,
)


def settle_lead_in(cover: Path, annotation: dict, out: Path) -> Path:
    """Move an accompaniment lead-in before the first downbeat onto it (see ``lead_in_to_settle``)."""
    from symusic import Score

    score = Score(str(cover)).to("second")
    starts = [note.time for track in score.tracks for note in track.notes]
    first = lead_in_to_settle(starts, annotation)
    if first is None:
        return cover
    for track in score.tracks:
        on_downbeat = {note.pitch for note in track.notes if first - 1e-3 <= note.time <= first + 0.06}
        early = [note for note in track.notes if note.time < first - 1e-3]
        for note in early:
            if note.pitch in on_downbeat or note.time + note.duration <= first + 0.05:
                note.duration = 0.0  # struck again on the downbeat, or over before it: drop
            else:
                note.duration = max(0.05, note.duration - (first - note.time))
                note.time = first
                on_downbeat.add(note.pitch)
        kept = [note for note in track.notes if note.duration > 0.0]
        if len(kept) != len(track.notes):
            track.notes.clear()
            track.notes.extend(kept)
        for pedal in getattr(track, "pedals", []):
            if pedal.time < first:
                pedal.duration = max(0.05, pedal.duration - (first - pedal.time))
                pedal.time = first
    score.to("tick").dump_midi(str(out))
    return out


def apply_signatures(measures: list, signatures: list[int]) -> int:
    """Give each measure its key signature and spell the notes of changed measures for it."""
    from piano_perf2score.tidy import chromatic_spelling, spell_pitch_class

    changed = 0
    for measure, signature in zip(measures, signatures, strict=True):
        if measure.key == signature:
            continue
        changed += 1
        measure.key = signature
        for group in measure.groups:
            for note in group.notes:
                note.alter = spell_pitch_class(note.pitch % 12, signature)
    if changed:
        chromatic_spelling(measures)
    return changed


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
    parser.add_argument("--keys", default=None, help="key tracker output (sheetsage2_refined.keys)")
    parser.add_argument(
        "--key-policy",
        choices=("follow", "correct"),
        default="follow",
        help="follow: key signatures come from the tracker (songs); correct: the score model's regions, "
        "each corrected where the tracker hears a single signature (solo piano)",
    )
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
    cover = settle_lead_in(Path(args.cover), annotation, Path(args.out).with_suffix(".settled.mid"))

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
    # tidy_score may have removed empty measures at the very start; they count toward the warm-up.
    first_bar = max(warmup_measures, tidied.rests)
    measures = drop_warmup(measures, first_bar - tidied.rests)
    key_changes = 0
    if args.keys:
        tracked = json.loads(Path(args.keys).read_text(encoding="utf-8")).get("keys") or []
        shift = (warmup[1] - warmup[0]) if warmup is not None else 0.0
        bar_starts = [float(t) - shift for t in performance.bar_seconds[first_bar : first_bar + len(measures)]]
        if tracked and len(bar_starts) == len(measures):
            heard = bar_signatures(tracked, bar_starts)
            model_keys = [m.key for m in measures]
            policy = follow_tracker if args.key_policy == "follow" else correct_regions
            key_changes = apply_signatures(measures, policy(model_keys, heard))
    mark_anacrusis(measures)
    write_musicxml(measures, Path(args.out), title=args.title)
    print(
        f"{len(measures)} measures -> {args.out} "
        f"(warm-up measures dropped: {warmup_measures}, measures given the tracked key: {key_changes})"
    )


if __name__ == "__main__":
    main()

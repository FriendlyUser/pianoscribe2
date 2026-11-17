"""A synthesized piano piece whose every note is known: the strictest ground truth available."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import mido
import pytest

from eval.arrangement import Reference, RefNote, evaluate
from eval.scoreio import read_musicxml
from eval.transcription import note_scores
from tests import synth

from .conftest import Produced

pytestmark = pytest.mark.grounded
LEAD_IN = 0.25  # seconds of silence synth.render puts before the first note
MELODY_VELOCITY = 92


@pytest.fixture(scope="module")
def ode(produce: Callable[..., Produced], tmp_path_factory: pytest.TempPathFactory) -> Produced:
    wav = synth.ode_to_joy_wav(tmp_path_factory.mktemp("synth") / "ode_to_joy.wav")
    return produce(wav, solo=True, title="Ode to Joy")


def truth_midi(path: Path) -> Path:
    seconds_per_beat = 60.0 / synth.ODE_BPM
    events = []
    for note in synth.ode_to_joy():
        start = LEAD_IN + note.start_beat * seconds_per_beat
        events += [(start, "note_on", note), (start + note.duration_beats * seconds_per_beat * 0.95, "note_off", note)]
    track, now = mido.MidiTrack(), 0.0
    track.append(mido.MetaMessage("set_tempo", tempo=500_000))
    for when, kind, note in sorted(events, key=lambda e: (e[0], e[1] == "note_on")):
        track.append(mido.Message(kind, note=note.pitch, velocity=note.velocity, time=round((when - now) * 960)))
        now += round((when - now) * 960) / 960
    midi = mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(track)
    midi.save(path)
    return path


def reference() -> Reference:
    notes = synth.ode_to_joy()
    melody = [n for n in notes if n.velocity == MELODY_VELOCITY]
    bass: dict[tuple[int, float], int] = {}
    for n in notes:
        if n.velocity != MELODY_VELOCITY and n.start_beat % 2 == 0:
            key = (int(n.start_beat // 4) + 1, n.start_beat % 4)
            bass[key] = min(bass.get(key, 127), n.pitch)
    return Reference(
        title="Ode to Joy (synthesized)",
        meter=synth.ODE_METER,
        bars=8,
        bar_ql=4.0,
        key_regions=[(1, synth.ODE_KEY_FIFTHS)],
        melody=[
            RefNote(int(n.start_beat // 4) + 1, n.start_beat % 4, n.duration_beats, n.pitch, n.start_beat, 1.0)
            for n in sorted(melody, key=lambda n: n.start_beat)
        ],
        bass=sorted((bar, pos, pitch % 12) for (bar, pos), pitch in bass.items()),
        felt_bpm_range=None,
    )


def test_every_played_note_is_transcribed(ode: Produced, tmp_path: Path) -> None:
    scores = note_scores(truth_midi(tmp_path / "truth.mid"), ode.directory / "performance.mid")
    assert scores["onset_f1"] >= 0.97, scores
    assert scores["offset_f1"] >= 0.80, scores


def test_score_facts_match_the_piece(ode: Produced) -> None:
    result = ode.result
    assert result.mode == "transcription"
    assert result.key == "G major"
    assert result.time_signature == "4/4"
    assert result.measures == 8
    assert result.tempo_unit == "quarter" and abs(result.tempo_bpm - synth.ODE_BPM) <= 4
    assert result.notes == len(synth.ode_to_joy())


def test_the_written_score_is_the_piece(ode: Produced) -> None:
    scores = evaluate(reference(), read_musicxml(ode.score))
    assert scores.melody_recall == 1.0, scores
    assert scores.melody_precision >= 0.97, scores
    assert scores.melody_octave_exact == 1.0, scores
    assert scores.rhythm_agreement >= 0.97, scores
    assert scores.bar_placement >= 0.97, scores
    assert scores.bass_accuracy >= 0.9, scores
    assert scores.meter_correct and scores.key_accuracy == 1.0, scores


def test_hands_are_split_at_the_right_place(ode: Produced) -> None:
    score = read_musicxml(ode.score)
    upper = [n.pitch for n in score.notes if n.staff == 1]
    lower = [n.pitch for n in score.notes if n.staff == 2]
    assert upper and lower
    assert min(upper) >= 67 and max(lower) <= 57  # melody G4..D5 above, accompaniment G2..A3 below

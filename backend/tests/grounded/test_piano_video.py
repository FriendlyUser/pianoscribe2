"""A piano-tutorial video rendered from a piece whose every note is known, through the whole pipeline.

The notes are read from the picture (``app.visual``) and written down by rule; only the beats,
the bar lines and the key come from the audio models. So unlike a transcription by ear, the
score is held to being the piece exactly: every note, on the staff of the hand that plays it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.jobs.models import Job
from eval.arrangement import evaluate
from eval.scoreio import read_musicxml
from tests import piano_video, synth

from .test_synthetic_piano import reference

pytestmark = pytest.mark.grounded


@pytest.fixture(scope="module")
def ode(model_settings: Settings, tmp_path_factory: pytest.TempPathFactory) -> tuple[Job, Path]:
    from app.pipeline.local import run_local

    folder = tmp_path_factory.mktemp("piano-video")
    notes = piano_video.from_beats(synth.ode_to_joy(), synth.ODE_BPM)
    falling = [piano_video.VideoNote(n.pitch, n.start + 3.0, n.end + 3.0, n.hand) for n in notes]
    video = piano_video.render(falling, folder / "ode_to_joy.mp4")
    job = run_local(video, folder / "out", model_settings, video=True, title="Ode to Joy")
    return job, folder / "out" / "score.musicxml"


def test_every_note_shown_is_read_and_written(ode: tuple[Job, Path]) -> None:
    result = ode[0].result
    assert result is not None and result.mode == "video" and result.video is not None
    assert result.video.notes == len(synth.ode_to_joy()) == result.notes
    assert result.video.confirmed_by_keys == result.video.notes
    assert result.video.score_kept == 1.0 and result.video.score_backed == 1.0
    assert result.video.hands_from == "colours" and result.video.octave_from == "audio"


def test_score_facts_match_the_piece(ode: tuple[Job, Path]) -> None:
    result = ode[0].result
    assert result is not None
    assert result.key == "G major" and result.time_signature == "4/4" and result.measures == 8
    assert result.tempo_unit == "quarter" and abs(result.tempo_bpm - synth.ODE_BPM) <= 4


def test_the_written_score_is_the_piece(ode: tuple[Job, Path]) -> None:
    scores = evaluate(reference(), read_musicxml(ode[1]))
    assert scores.melody_recall == 1.0 and scores.melody_precision == 1.0, scores
    assert scores.melody_octave_exact == 1.0 and scores.rhythm_agreement == 1.0, scores
    assert scores.bar_placement == 1.0 and scores.bass_accuracy == 1.0, scores
    assert scores.meter_correct and scores.key_accuracy == 1.0, scores


def test_each_hand_is_on_its_own_staff(ode: tuple[Job, Path]) -> None:
    score = read_musicxml(ode[1])
    melody = sorted(n.pitch for n in synth.ode_to_joy() if n.velocity == 92)
    accompaniment = sorted(n.pitch for n in synth.ode_to_joy() if n.velocity != 92)
    assert sorted(n.pitch for n in score.notes if n.staff == 1) == melody
    assert sorted(n.pitch for n in score.notes if n.staff == 2) == accompaniment

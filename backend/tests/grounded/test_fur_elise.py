"""A real recording of a public-domain piece, against its printed score (bars 1-9).

Recording: samples/fur_elise_opening.mp3 (see samples/README.md for its licence).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from eval.arrangement import ArrangementScores, Reference, evaluate
from eval.scoreio import read_musicxml

from .conftest import REFERENCES, SAMPLES, Produced

pytestmark = pytest.mark.grounded


@pytest.fixture(scope="module")
def scores(produce: Callable[..., Produced]) -> ArrangementScores:
    recording = SAMPLES / "fur_elise_opening.mp3"
    if not recording.is_file():
        pytest.skip(f"{recording} is missing")
    produced = produce(recording, solo=True, title="Für Elise")
    return evaluate(Reference.load(REFERENCES / "fur_elise_opening.json"), read_musicxml(produced.score))


def test_the_melody_is_complete_and_in_the_right_octave(scores: ArrangementScores) -> None:
    # 33 of 35: every note is in the score, but two middle Cs that the edition gives to the
    # right hand are written on the lower staff, and the melody is read from the upper one.
    assert scores.melody_recall >= 0.93, scores
    assert scores.melody_precision >= 0.95, scores
    assert scores.melody_octave_exact >= 0.97, scores


def test_the_rhythm_matches_the_printed_score(scores: ArrangementScores) -> None:
    assert scores.rhythm_agreement >= 0.95, scores


def test_the_bass_is_right(scores: ArrangementScores) -> None:
    assert scores.bass_accuracy >= 0.8, scores


def test_notes_sit_at_the_right_place_in_their_bar(scores: ArrangementScores) -> None:
    assert scores.bar_placement >= 0.9, scores  # includes the pickup, written as an incomplete first bar


def test_the_key_signature_is_a_minor(scores: ArrangementScores) -> None:
    # The score model alone writes one sharp for this short clip; the key tracker corrects it.
    assert scores.key_accuracy == 1.0, scores


@pytest.mark.xfail(
    strict=True,
    reason="Known limit: the beat model counts this piece in quarters, so 3/8 is written as 3/4 "
    "with every note value doubled. The notes and bar lines are right; the unit is not.",
)
def test_the_time_signature_is_three_eight(scores: ArrangementScores) -> None:
    assert scores.meter_correct, scores

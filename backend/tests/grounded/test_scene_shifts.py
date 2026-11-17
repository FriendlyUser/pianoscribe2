"""A band recording arranged for piano, against a published human piano arrangement.

Needs material that is not in the repository (see eval/README.md):
  eval/data/scene_shifts_there.wav      the recording
  eval/private/scene_shifts_there.json  the reference, read from the published arrangement
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from eval.arrangement import ArrangementScores, Reference, evaluate
from eval.scoreio import read_musicxml

from .conftest import DATA, PRIVATE, Produced

pytestmark = pytest.mark.grounded
REFERENCE = PRIVATE / "scene_shifts_there.json"


def _recording():
    return next(
        (p for p in sorted(DATA.glob("scene_shifts_there.*")) if p.suffix in {".wav", ".mp3", ".mp4", ".mov"}), None
    )


@pytest.fixture(scope="module")
def produced(produce: Callable[..., Produced]) -> Produced:
    recording = _recording()
    if recording is None or not REFERENCE.is_file():
        pytest.skip("the Scene Shifts There recording or reference is not present")
    return produce(recording, solo=False, title="Scene Shifts There")


@pytest.fixture(scope="module")
def scores(produced: Produced) -> ArrangementScores:
    return evaluate(Reference.load(REFERENCE), read_musicxml(produced.score))


def test_overall_agreement_with_the_human_arrangement(scores: ArrangementScores) -> None:
    # The first two generations of this project scored 0.40 and 0.73 on the same reference.
    # Arrangements are sampled: seven of this recording scored between 0.82 and 0.86.
    assert scores.overall >= 0.80, scores


def test_the_melody_is_carried_throughout(scores: ArrangementScores) -> None:
    assert scores.melody_recall >= 0.88, scores
    assert scores.melody_f1 >= 0.78, scores


def test_rhythm_and_bar_lines(scores: ArrangementScores) -> None:
    assert scores.rhythm_agreement >= 0.82, scores
    assert scores.bar_placement >= 0.80, scores
    assert scores.meter_correct, scores  # 6/8, not 3/4
    assert abs(scores.bar_count_ratio - 1.0) <= 0.03, scores


def test_bass_line(scores: ArrangementScores) -> None:
    assert scores.bass_accuracy >= 0.62, scores


def test_the_opening_key_is_right(produced: Produced) -> None:
    assert read_musicxml(produced.score).bars[0].fifths == -1  # F major


@pytest.mark.xfail(
    strict=False,
    reason="Known limit: the piece changes key three times (F, D, C, F) and the score model does not "
    "follow every change in every arrangement. Across seven arrangements of this recording the share "
    "of bars under the right key signature ranged from 0.42 to 0.93.",
)
def test_every_key_change_is_followed(scores: ArrangementScores) -> None:
    assert scores.key_accuracy >= 0.9, scores


def test_the_arrangement_is_playable_density(scores: ArrangementScores, produced: Produced) -> None:
    assert 6 <= scores.notes_per_bar <= 16, scores
    assert produced.result.mode == "arrangement"
    assert produced.result.melody_fidelity is not None and produced.result.melody_fidelity >= 0.7

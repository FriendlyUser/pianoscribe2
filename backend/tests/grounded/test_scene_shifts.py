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
    # Arrangements are sampled: ten of this recording scored between 0.84 and 0.88.
    assert scores.overall >= 0.82, scores


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


def test_key_changes_are_followed(scores: ArrangementScores, produced: Produced) -> None:
    # F, (D for eight bars), C, F. The key tracker hears the changes in the recording; the
    # eight bars of D major are too short for a signature of their own under our rule.
    assert scores.key_accuracy >= 0.9, scores
    signatures = [bar.fifths for bar in read_musicxml(produced.score).bars]
    assert signatures[0] == -1 and signatures[40] == 0 and signatures[-1] == -1


def test_the_first_bar_states_the_opening_phrase(produced: Produced) -> None:
    # Bar 1 of the published arrangement: A D E F G in eighth notes from the downbeat.
    score = read_musicxml(produced.score)
    first_bar = [(float(n.pos), n.pitch % 12) for n in score.notes if n.bar == 0]
    wanted = [(0.0, 9), (0.5, 2), (1.0, 4), (1.5, 5), (2.0, 7)]
    found = [any(abs(pos - at) <= 0.13 and pc == pitch_class for pos, pc in first_bar) for at, pitch_class in wanted]
    assert score.bars[0].length == 3  # a full 6/8 bar, no pickup
    assert sum(found) >= 4, first_bar


def test_the_arrangement_is_playable_density(scores: ArrangementScores, produced: Produced) -> None:
    assert 6 <= scores.notes_per_bar <= 16, scores
    assert produced.result.mode == "arrangement"
    assert produced.result.melody_fidelity is not None and produced.result.melody_fidelity >= 0.7

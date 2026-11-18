"""A long solo-piano recording: the case in which the score model, given the whole piece, writes notes twice.

Found in a field test on a 203-bar waltz (4.5 minutes): written in one pass, the score had 39 %
more notes than were played. Long solo pieces are therefore written in sections.

Needs a solo-piano recording of more than about three minutes that is not in the repository:
  eval/data/long_solo.wav   (any; the field test used Joe Hisaishi's "Merry-Go-Round", piano solo version)
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from eval.fidelity import NoteFidelity, note_fidelity
from eval.scoreio import read_musicxml

from .conftest import DATA, Produced

pytestmark = pytest.mark.grounded
RECORDING = DATA / "long_solo.wav"


@pytest.fixture(scope="module")
def produced(produce: Callable[..., Produced]) -> Produced:
    if not RECORDING.is_file():
        pytest.skip(f"{RECORDING} is not present")
    return produce(RECORDING, solo=True, title="Long solo")


@pytest.fixture(scope="module")
def fidelity(produced: Produced) -> NoteFidelity:
    return note_fidelity(produced.directory / "performance.mid", read_musicxml(produced.score))


def test_the_piece_is_long_enough_to_be_written_in_sections(produced: Produced) -> None:
    assert produced.result.measures > 64


def test_no_notes_are_invented(fidelity: NoteFidelity) -> None:
    assert fidelity.backed >= 0.95, fidelity
    assert abs(fidelity.extra) <= 0.05, fidelity


def test_no_notes_are_lost(fidelity: NoteFidelity) -> None:
    assert fidelity.kept >= 0.97, fidelity

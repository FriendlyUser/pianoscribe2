"""Note accuracy of the piano transcription stage on MAESTRO v3 test recordings.

The reference is the Disklavier's own MIDI: exact notes, measured at the instrument.
Needs eval/data/maestro/<name>.wav and <name>.midi (python -m eval.fetch_maestro).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from eval.transcription import note_scores

from .conftest import DATA

pytestmark = pytest.mark.grounded
PIECES = sorted(p.stem for p in (DATA / "maestro").glob("*.midi") if p.with_suffix(".wav").is_file())


@pytest.fixture(scope="module")
def results(model_settings: Settings, tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, float]]:
    from app.models import song

    if not PIECES:
        pytest.skip("no MAESTRO recordings in eval/data/maestro")
    out = tmp_path_factory.mktemp("maestro")
    scores = {}
    for name in PIECES:
        midi = song.transcribe_piano(DATA / "maestro" / f"{name}.wav", Path(out) / f"{name}.mid", model_settings)
        scores[name] = note_scores(DATA / "maestro" / f"{name}.midi", midi)
    return scores


def test_mean_onset_f1(results: dict[str, dict[str, float]]) -> None:
    mean = sum(r["onset_f1"] for r in results.values()) / len(results)
    assert mean >= 0.96, {name: round(r["onset_f1"], 3) for name, r in results.items()}


def test_no_piece_falls_far_behind(results: dict[str, dict[str, float]]) -> None:
    worst = min(results.items(), key=lambda item: item[1]["onset_f1"])
    assert worst[1]["onset_f1"] >= 0.93, worst


def test_note_lengths(results: dict[str, dict[str, float]]) -> None:
    mean = sum(r["offset_f1"] for r in results.values()) / len(results)
    assert mean >= 0.80, {name: round(r["offset_f1"], 3) for name, r in results.items()}

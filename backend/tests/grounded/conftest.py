"""Accuracy tests against ground truth, with the real models.

    pytest -m grounded

Each recording goes through the same pipeline the worker runs, once per session.
Tests whose reference material is not present are skipped (see eval/README.md).
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.config import Settings
from app.jobs.models import JobResult

DATA = Path(os.environ.get("PS_GROUNDED_DATA", Path(__file__).resolve().parents[2] / "eval" / "data"))
SAMPLES = Path(os.environ.get("PS_SAMPLES_DIR", "/samples"))
if not SAMPLES.is_dir():
    SAMPLES = Path(__file__).resolve().parents[3] / "samples"
REFERENCES = Path(__file__).resolve().parents[2] / "eval" / "references"
PRIVATE = Path(__file__).resolve().parents[2] / "eval" / "private"
# Set PS_GROUNDED_KEEP to a directory to keep the produced scores for inspection.
KEEP = os.environ.get("PS_GROUNDED_KEEP")


@dataclass(frozen=True)
class Produced:
    directory: Path
    result: JobResult
    timings: dict[str, float]

    @property
    def score(self) -> Path:
        return self.directory / "score.musicxml"


@pytest.fixture(scope="session")
def model_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    settings = Settings(data_dir=tmp_path_factory.mktemp("data"), log_format="console")
    if not settings.tsumugi_python.is_file() or not settings.cover_python.is_file():
        pytest.skip("the model environments are not installed here (run inside the worker / tests image)")
    return settings


@pytest.fixture(scope="session")
def cover_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """For tests that run only tsumugi-piano-cover's environment (beats, keys, the score writers),
    such as a piano video's: CI installs just that one (.github/workflows/piano-video.yml)."""
    settings = Settings(data_dir=tmp_path_factory.mktemp("data"), log_format="console")
    if not settings.cover_python.is_file():
        pytest.skip("the tsumugi-piano-cover environment is not installed here")
    return settings


@pytest.fixture(scope="session")
def produce(model_settings: Settings, tmp_path_factory: pytest.TempPathFactory) -> Callable[..., Produced]:
    """produce(recording, solo=...) -> the pipeline's output for it (cached for the session)."""
    from app.pipeline.local import run_local

    done: dict[tuple[Path, bool], Produced] = {}

    def run(recording: Path, *, solo: bool, title: str | None = None) -> Produced:
        key = (recording, solo)
        if key not in done:
            out = tmp_path_factory.mktemp(recording.stem)
            job = run_local(recording, out, model_settings, solo=solo, title=title)
            assert job.result is not None
            if KEEP:
                shutil.copytree(out, Path(KEEP) / f"{recording.stem}{'-solo' if solo else ''}", dirs_exist_ok=True)
            done[key] = Produced(out, job.result, job.timings)
        return done[key]

    return run

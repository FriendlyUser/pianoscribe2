from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import fakeredis
import pytest

from app.config import Settings
from app.jobs.store import JobStore
from app.storage.artifacts import ArtifactStore


class RecordingQueue:
    """In-memory stand-in for the Dramatiq queue."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, int]] = []

    def enqueue(self, job_id: str, *, delay_ms: int = 0) -> None:
        self.sent.append((job_id, delay_ms))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    models = Path(os.environ.get("PS_MODELS_DIR", tmp_path / "models"))
    cfg = Settings(
        data_dir=tmp_path / "data",
        models_dir=models,
        log_format="console",
        job_lock_ttl_seconds=5,
        youtube_backoff_base_seconds=0.01,
        youtube_backoff_max_seconds=0.02,
    )
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def redis_client() -> Iterator[fakeredis.FakeRedis]:
    client = fakeredis.FakeRedis()
    yield client
    client.flushall()


@pytest.fixture
def store(redis_client: fakeredis.FakeRedis) -> JobStore:
    return JobStore(redis_client, record_ttl_seconds=3600)


@pytest.fixture
def queue() -> RecordingQueue:
    return RecordingQueue()


@pytest.fixture
def artifacts(settings: Settings) -> ArtifactStore:
    return ArtifactStore(settings.artifacts_dir)

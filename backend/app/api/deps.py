"""Dependency providers (overridable in tests)."""

from __future__ import annotations

from functools import lru_cache

import redis.asyncio as aioredis

from app.config import Settings, get_settings
from app.jobs.service import JobQueue
from app.jobs.store import JobStore
from app.storage.artifacts import ArtifactStore


def settings_dep() -> Settings:
    return get_settings()


def store_dep() -> JobStore:
    from app.jobs.queue import get_store

    return get_store()


def queue_dep() -> JobQueue:
    from app.jobs.queue import DramatiqQueue

    return DramatiqQueue()


def job_models_dep() -> dict[str, str | int | float] | None:
    """Per-job model settings: none on the server; the desktop app supplies the user's choices."""
    return None


def artifacts_dep() -> ArtifactStore:
    from app.jobs.queue import get_artifacts

    return get_artifacts()


@lru_cache
def _async_redis(url: str) -> aioredis.Redis:
    return aioredis.Redis.from_url(url)


def async_redis_dep() -> aioredis.Redis:
    return _async_redis(get_settings().redis_url)

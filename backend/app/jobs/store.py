"""Redis-backed job state store with change notifications."""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

import redis

from app.jobs.models import Job

_PREFIX = "ps"
_RELEASE_LOCK = """
if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) end
return 0
"""
_RENEW_LOCK = """
if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('pexpire', KEYS[1], ARGV[2]) end
return 0
"""


def job_key(job_id: str) -> str:
    return f"{_PREFIX}:job:{job_id}"


def events_channel(job_id: str) -> str:
    return f"{_PREFIX}:events:{job_id}"


class JobStore:
    """All reads and writes of job records go through this class."""

    def __init__(self, client: redis.Redis, *, record_ttl_seconds: int) -> None:
        self.client = client
        self.record_ttl = record_ttl_seconds

    def get(self, job_id: str) -> Job | None:
        raw = self.client.get(job_key(job_id))
        return Job.model_validate_json(raw) if raw else None

    def create_if_absent(self, job: Job) -> bool:
        """Atomically create ``job``; False when a record with its id already exists."""
        created = self.client.set(job_key(job.id), job.model_dump_json(), nx=True, ex=self.record_ttl)
        if created:
            self.client.zadd(f"{_PREFIX}:jobs", {job.id: job.created_at})
        return bool(created)

    def mutate(self, job_id: str, change: Callable[[Job], Job | None]) -> Job | None:
        """Apply ``change`` under optimistic locking and publish the new state.

        ``change`` returns the updated job, or None to leave it untouched.
        """
        key = job_key(job_id)
        with self.client.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(key)  # type: ignore[no-untyped-call]
                    raw = pipe.get(key)
                    if not raw:
                        pipe.unwatch()
                        return None
                    updated = change(Job.model_validate_json(raw))
                    if updated is None:
                        pipe.unwatch()
                        return None
                    updated.updated_at = time.time()
                    payload = updated.model_dump_json()
                    pipe.multi()
                    pipe.set(key, payload, ex=self.record_ttl)
                    pipe.publish(events_channel(job_id), payload)
                    pipe.execute()
                    return updated
                except redis.WatchError:
                    continue

    def update(self, job_id: str, **fields: Any) -> Job | None:
        return self.mutate(job_id, lambda job: job.model_copy(update=fields))

    def delete(self, job_id: str) -> None:
        self.client.delete(job_key(job_id))
        self.client.zrem(f"{_PREFIX}:jobs", job_id)

    def lock(self, job_id: str, ttl_seconds: int) -> JobLock:
        return JobLock(self.client, f"{_PREFIX}:lock:{job_id}", ttl_seconds)

    def is_locked(self, job_id: str) -> bool:
        return bool(self.client.exists(f"{_PREFIX}:lock:{job_id}"))


class JobLock:
    """Per-job mutual exclusion with a heartbeat.

    The lock expires on its own if the holder dies, so a redelivered message
    can take over; while the holder lives a background thread keeps renewing it.
    """

    def __init__(self, client: redis.Redis, key: str, ttl_seconds: int) -> None:
        self.client = client
        self.key = key
        self.ttl_ms = ttl_seconds * 1000
        self.token = uuid.uuid4().hex
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def acquire(self) -> bool:
        if not self.client.set(self.key, self.token, nx=True, px=self.ttl_ms):
            return False
        self._thread = threading.Thread(target=self._heartbeat, name="job-lock-heartbeat", daemon=True)
        self._thread.start()
        return True

    def _heartbeat(self) -> None:
        interval = self.ttl_ms / 3000
        while not self._stop.wait(interval):
            try:
                self.client.eval(_RENEW_LOCK, 1, self.key, self.token, self.ttl_ms)
            except redis.RedisError:
                continue

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        with contextlib.suppress(redis.RedisError):
            self.client.eval(_RELEASE_LOCK, 1, self.key, self.token)

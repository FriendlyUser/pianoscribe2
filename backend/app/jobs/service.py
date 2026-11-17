"""Job submission: deduplication, cache hits and idempotent enqueueing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings
from app.jobs.models import Job, JobOptions, JobSource, JobStatus, Stage
from app.jobs.store import JobStore
from app.logging import get_logger
from app.storage.artifacts import ArtifactStore

log = get_logger(__name__)


class JobQueue(Protocol):
    def enqueue(self, job_id: str, *, delay_ms: int = 0) -> None: ...


@dataclass(frozen=True)
class Submission:
    job: Job
    cached: bool  # finished result served without recomputation
    deduplicated: bool  # attached to an identical job already in flight


def submit_job(
    *,
    job_id: str,
    source: JobSource,
    options: JobOptions,
    title: str,
    store: JobStore,
    queue: JobQueue,
    artifacts: ArtifactStore,
    settings: Settings,
) -> Submission:
    """Create or reuse the job identified by ``job_id``.

    * finished and artifacts still present -> returned as a cache hit;
    * queued / running -> the caller is attached to the in-flight job;
    * failed, or finished but purged -> reset and re-queued;
    * unknown -> created and queued.
    """
    fresh = Job(id=job_id, source=source, options=options, title=title)
    if store.create_if_absent(fresh):
        queue.enqueue(job_id)
        log.info("job.submitted", job_id=job_id, source=source.kind.value, outcome="created")
        return Submission(job=fresh, cached=False, deduplicated=False)

    existing = store.get(job_id)
    if existing is None:  # expired between the two calls; retry once as new
        store.create_if_absent(fresh)
        queue.enqueue(job_id)
        return Submission(job=fresh, cached=False, deduplicated=False)

    if existing.status == JobStatus.SUCCEEDED and artifacts.complete(job_id):
        log.info("job.submitted", job_id=job_id, source=source.kind.value, outcome="cache_hit")
        return Submission(job=existing, cached=True, deduplicated=False)

    if existing.status in (JobStatus.QUEUED, JobStatus.RUNNING):
        # A stale "running" record whose worker died is recovered by the
        # broker's redelivery; re-sending here is harmless (the runner is
        # idempotent) and unblocks records orphaned by a lost queue.
        stalled = time.time() - existing.updated_at > settings.job_lock_ttl_seconds * 3
        if stalled and not store.is_locked(job_id):
            queue.enqueue(job_id)
        log.info("job.submitted", job_id=job_id, source=source.kind.value, outcome="in_flight")
        return Submission(job=existing, cached=False, deduplicated=True)

    def reset(job: Job) -> Job | None:
        if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            return None  # someone else restarted it first
        return job.model_copy(
            update={
                "status": JobStatus.QUEUED,
                "stage": Stage.QUEUED,
                "progress": 0.0,
                "message": "Waiting for a worker",
                "error": None,
                "result": None,
                "dead_lettered": False,
                "attempts": 0,
                "finished_at": None,
                "source": source,  # a re-upload refreshes the stored raw file reference
                "title": title or job.title,
            }
        )

    restarted = store.mutate(job_id, reset)
    if restarted is not None:
        queue.enqueue(job_id)
        log.info("job.submitted", job_id=job_id, source=source.kind.value, outcome="requeued")
        return Submission(job=restarted, cached=False, deduplicated=False)
    current = store.get(job_id) or existing
    return Submission(job=current, cached=False, deduplicated=True)

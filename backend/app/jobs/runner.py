"""Idempotent job execution, independent of the queueing library."""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable

from app.config import Settings
from app.errors import JobError
from app.jobs.models import Job, JobErrorInfo, JobStatus, Stage
from app.jobs.service import JobQueue
from app.jobs.store import JobStore
from app.logging import bind_job, get_logger
from app.storage.artifacts import ArtifactStore

log = get_logger(__name__)

Execute = Callable[[Job], Job]
_INTERNAL_ERROR = JobErrorInfo(
    code="internal_error",
    message="Something went wrong while processing this audio. Please try again.",
    retryable=True,
)


class TransientJobFailure(Exception):
    """Raised to the queue so the message is retried with backoff."""


def _default_execute(store: JobStore, artifacts: ArtifactStore, settings: Settings) -> Execute:
    def execute(job: Job) -> Job:
        from app.pipeline.runner import Pipeline

        effective = settings.model_copy(update=job.options.models) if job.options.models else settings
        return Pipeline(job, store=store, artifacts=artifacts, settings=effective).run()

    return execute


def fail_job(
    store: JobStore, settings: Settings, job_id: str, error: JobErrorInfo, *, dead_lettered: bool = False
) -> None:
    """Mark a job as finally failed and discard its intermediate files."""
    store.update(
        job_id,
        status=JobStatus.FAILED,
        stage=Stage.FAILED,
        message=error.message,
        error=error.model_copy(update={"retryable": False}) if dead_lettered else error,
        dead_lettered=dead_lettered,
        finished_at=time.time(),
    )
    shutil.rmtree(settings.work_dir / job_id, ignore_errors=True)


def run_job(
    job_id: str,
    *,
    store: JobStore,
    queue: JobQueue,
    artifacts: ArtifactStore,
    settings: Settings,
    execute: Execute | None = None,
) -> None:
    """Process one delivery of ``job_id``.

    Safe under at-least-once delivery: a per-job lock prevents concurrent
    runs, finished jobs are no-ops, and work resumes from stage checkpoints.
    User-facing errors end the job; anything else propagates so the queue
    retries it and eventually dead-letters the message.
    """
    execute = execute or _default_execute(store, artifacts, settings)
    with bind_job(job_id):
        lock = store.lock(job_id, settings.job_lock_ttl_seconds)
        if not lock.acquire():
            # Another worker holds this job. Check back after the lock could
            # have expired, in case that worker died mid-run.
            log.info("job.duplicate_delivery")
            queue.enqueue(job_id, delay_ms=settings.job_lock_ttl_seconds * 1000)
            return
        try:
            job = store.get(job_id)
            if job is None:
                log.warning("job.missing")
                return
            if job.terminal:
                log.info("job.already_finished", status=job.status.value)
                return
            job = store.update(job_id, status=JobStatus.RUNNING, attempts=job.attempts + 1, message="Starting") or job
            log.info("job.start", attempt=job.attempts, source=job.source.kind.value)
            started = time.perf_counter()
            try:
                execute(job)
            except JobError as exc:
                info = JobErrorInfo(code=exc.code, message=exc.message, retryable=exc.retryable)
                if exc.retryable:
                    log.warning("job.transient_failure", code=exc.code, error=exc.message)
                    store.update(job_id, error=info, message="Temporary problem, retrying shortly")
                    raise TransientJobFailure(exc.code) from exc
                log.info("job.failed", code=exc.code, error=exc.message)
                fail_job(store, settings, job_id, info)
                return
            except Exception as exc:
                log.exception("job.crashed", error=str(exc))
                store.update(job_id, error=_INTERNAL_ERROR, message="Temporary problem, retrying shortly")
                raise
            log.info("job.succeeded", seconds=round(time.perf_counter() - started, 2))
        finally:
            lock.release()


def dead_letter(job_id: str, *, store: JobStore, settings: Settings) -> None:
    """Record that the queue gave up on ``job_id`` (retries exhausted)."""
    with bind_job(job_id):
        job = store.get(job_id)
        if job is None or job.status == JobStatus.SUCCEEDED:
            return
        log.error("job.dead_lettered", attempts=job.attempts)
        fail_job(store, settings, job_id, job.error or _INTERNAL_ERROR, dead_lettered=True)

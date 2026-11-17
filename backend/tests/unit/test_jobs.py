"""Submission dedupe/caching and idempotent, failure-tolerant job execution."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.errors import JobError
from app.jobs.keys import upload_job_id
from app.jobs.models import Job, JobOptions, JobSource, JobStatus, SourceKind, Stage
from app.jobs.runner import TransientJobFailure, dead_letter, run_job
from app.jobs.service import submit_job
from app.jobs.store import JobStore
from app.storage.artifacts import REQUIRED, ArtifactStore
from tests.conftest import RecordingQueue

SHA = "ab" * 32
SOURCE = JobSource(kind=SourceKind.UPLOAD, filename="a.wav", extension="wav", content_sha256=SHA)


def submit(store: JobStore, queue: RecordingQueue, artifacts: ArtifactStore, settings: Settings, **options: bool):
    opts = JobOptions(**options)
    return submit_job(
        job_id=upload_job_id(SHA, opts), source=SOURCE, options=opts, title="A",
        store=store, queue=queue, artifacts=artifacts, settings=settings,
    )  # fmt: skip


def succeed(store: JobStore, artifacts: ArtifactStore):
    def execute(job: Job) -> Job:
        artifacts.put_all(job.id, {name: b"data" for name in REQUIRED})
        return store.update(job.id, status=JobStatus.SUCCEEDED, stage=Stage.READY, progress=1.0)  # type: ignore[return-value]

    return execute


def test_first_submission_creates_and_enqueues(store, queue, artifacts, settings) -> None:
    result = submit(store, queue, artifacts, settings)
    assert not result.cached and not result.deduplicated
    assert result.job.status == JobStatus.QUEUED
    assert queue.sent == [(result.job.id, 0)]


def test_identical_submission_while_in_flight_is_deduplicated(store, queue, artifacts, settings) -> None:
    first = submit(store, queue, artifacts, settings)
    second = submit(store, queue, artifacts, settings)
    assert second.deduplicated and second.job.id == first.job.id
    assert len(queue.sent) == 1  # no second enqueue, no duplicate compute


def test_finished_job_is_served_from_cache(store, queue, artifacts, settings) -> None:
    first = submit(store, queue, artifacts, settings)
    run_job(
        first.job.id,
        store=store,
        queue=queue,
        artifacts=artifacts,
        settings=settings,
        execute=succeed(store, artifacts),
    )
    again = submit(store, queue, artifacts, settings)
    assert again.cached and again.job.status == JobStatus.SUCCEEDED
    assert len(queue.sent) == 1


def test_different_options_are_a_different_job(store, queue, artifacts, settings) -> None:
    a = submit(store, queue, artifacts, settings, solo_piano=False)
    b = submit(store, queue, artifacts, settings, solo_piano=True)
    assert a.job.id != b.job.id and len(queue.sent) == 2


def test_purged_results_are_recomputed(store, queue, artifacts, settings) -> None:
    first = submit(store, queue, artifacts, settings)
    run_job(
        first.job.id,
        store=store,
        queue=queue,
        artifacts=artifacts,
        settings=settings,
        execute=succeed(store, artifacts),
    )
    artifacts.delete(first.job.id)
    again = submit(store, queue, artifacts, settings)
    assert not again.cached and again.job.status == JobStatus.QUEUED and len(queue.sent) == 2


def test_failed_job_can_be_resubmitted(store, queue, artifacts, settings) -> None:
    first = submit(store, queue, artifacts, settings)

    def reject(job: Job) -> Job:
        raise JobError("too_long", "Too long.")

    run_job(first.job.id, store=store, queue=queue, artifacts=artifacts, settings=settings, execute=reject)
    failed = store.get(first.job.id)
    assert failed is not None and failed.status == JobStatus.FAILED and failed.error and failed.error.code == "too_long"
    retry = submit(store, queue, artifacts, settings)
    assert retry.job.status == JobStatus.QUEUED and retry.job.error is None and len(queue.sent) == 2


def test_redelivery_of_a_finished_job_is_a_no_op(store, queue, artifacts, settings) -> None:
    job = submit(store, queue, artifacts, settings).job
    calls = []

    def execute(j: Job) -> Job:
        calls.append(j.id)
        return succeed(store, artifacts)(j)

    for _ in range(3):
        run_job(job.id, store=store, queue=queue, artifacts=artifacts, settings=settings, execute=execute)
    assert calls == [job.id]


def test_concurrent_delivery_is_deferred_not_run_twice(store, queue, artifacts, settings) -> None:
    job = submit(store, queue, artifacts, settings).job
    held = store.lock(job.id, 30)
    assert held.acquire()
    try:
        ran = []
        run_job(
            job.id,
            store=store,
            queue=queue,
            artifacts=artifacts,
            settings=settings,
            execute=lambda j: ran.append(j) or j,
        )
        assert ran == []
        assert queue.sent[-1] == (job.id, settings.job_lock_ttl_seconds * 1000)  # re-check later
    finally:
        held.release()
    assert not store.is_locked(job.id)


def test_transient_errors_propagate_for_retry_and_keep_job_alive(store, queue, artifacts, settings) -> None:
    job = submit(store, queue, artifacts, settings).job

    def flaky(j: Job) -> Job:
        raise JobError("model_unavailable", "Model download failed.", retryable=True)

    with pytest.raises(TransientJobFailure):
        run_job(job.id, store=store, queue=queue, artifacts=artifacts, settings=settings, execute=flaky)
    current = store.get(job.id)
    assert current is not None and current.status == JobStatus.RUNNING and current.attempts == 1
    assert not store.is_locked(job.id)  # lock released for the retry

    def crash(j: Job) -> Job:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        run_job(job.id, store=store, queue=queue, artifacts=artifacts, settings=settings, execute=crash)
    assert store.get(job.id).attempts == 2  # type: ignore[union-attr]


def test_dead_lettering_marks_the_job_failed(store, queue, artifacts, settings) -> None:
    job = submit(store, queue, artifacts, settings).job
    with pytest.raises(RuntimeError):
        run_job(
            job.id,
            store=store,
            queue=queue,
            artifacts=artifacts,
            settings=settings,
            execute=lambda j: (_ for _ in ()).throw(RuntimeError("x")),
        )
    (settings.work_dir / job.id).mkdir(parents=True, exist_ok=True)
    dead_letter(job.id, store=store, settings=settings)
    final = store.get(job.id)
    assert final is not None and final.status == JobStatus.FAILED and final.dead_lettered
    assert final.error is not None and final.error.code == "internal_error"
    assert not (settings.work_dir / job.id).exists()


def test_dead_letter_never_overwrites_success(store, queue, artifacts, settings) -> None:
    job = submit(store, queue, artifacts, settings).job
    run_job(job.id, store=store, queue=queue, artifacts=artifacts, settings=settings, execute=succeed(store, artifacts))
    dead_letter(job.id, store=store, settings=settings)
    assert store.get(job.id).status == JobStatus.SUCCEEDED  # type: ignore[union-attr]


def test_store_publishes_updates(store, redis_client) -> None:
    from app.jobs.store import events_channel

    job = Job(id="c" * 32, source=SOURCE)
    assert store.create_if_absent(job) and not store.create_if_absent(job)
    pubsub = redis_client.pubsub()
    pubsub.subscribe(events_channel(job.id))
    pubsub.get_message(timeout=1)
    store.update(job.id, stage=Stage.TRANSCRIBING, progress=0.5)
    message = pubsub.get_message(timeout=1)
    assert message is not None and Job.model_validate_json(message["data"]).stage == Stage.TRANSCRIBING
    assert store.update("f" * 32, progress=1.0) is None

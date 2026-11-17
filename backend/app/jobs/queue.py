"""Dramatiq wiring: broker, the job actor and dead-letter handling.

Dramatiq's Redis broker provides at-least-once delivery (messages are acked
only after the actor returns; messages held by a crashed worker are
re-queued), retries with exponential backoff, and a dead-letter queue for
messages that exhaust their retries.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, cast

import dramatiq
import redis
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware import Middleware

from app.config import Settings, get_settings
from app.jobs.runner import dead_letter, run_job
from app.jobs.store import JobStore
from app.logging import configure_logging, get_logger
from app.storage.artifacts import ArtifactStore

log = get_logger(__name__)
ACTOR_NAME = "process_job"


@lru_cache
def get_redis() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url)


@lru_cache
def get_store() -> JobStore:
    return JobStore(get_redis(), record_ttl_seconds=get_settings().result_retention_seconds)


@lru_cache
def get_artifacts() -> ArtifactStore:
    return ArtifactStore(get_settings().artifacts_dir)


class DeadLetterMiddleware(Middleware):
    """Reflect broker-level dead-lettering in the job record."""

    def after_nack(self, broker: dramatiq.Broker, message: Any) -> None:
        if message.actor_name == ACTOR_NAME and message.args:
            dead_letter(str(message.args[0]), store=get_store(), settings=get_settings())

    def after_skip_message(self, broker: dramatiq.Broker, message: Any) -> None:
        self.after_nack(broker, message)


class WorkerBootMiddleware(Middleware):
    def after_process_boot(self, broker: dramatiq.Broker) -> None:
        settings = get_settings()
        configure_logging(settings.log_level, settings.log_format)
        settings.ensure_dirs()


def _build_broker(settings: Settings) -> RedisBroker:
    broker = RedisBroker(  # type: ignore[no-untyped-call]
        url=settings.redis_url,
        namespace="ps-queue",
        heartbeat_timeout=settings.queue_heartbeat_timeout_seconds * 1000,
        maintenance_chance=settings.queue_maintenance_chance,
    )
    broker.add_middleware(DeadLetterMiddleware())
    broker.add_middleware(WorkerBootMiddleware())
    return broker


_settings = get_settings()
broker = _build_broker(_settings)
dramatiq.set_broker(broker)


class DramatiqQueue:
    """:class:`~app.jobs.service.JobQueue` backed by the Dramatiq actor."""

    def enqueue(self, job_id: str, *, delay_ms: int = 0) -> None:
        process_job.send_with_options(args=(job_id,), delay=delay_ms or None)


@dramatiq.actor(
    actor_name=ACTOR_NAME,
    queue_name=_settings.queue_name,
    max_retries=_settings.job_max_retries,
    min_backoff=_settings.job_retry_min_backoff_ms,
    max_backoff=_settings.job_retry_max_backoff_ms,
    time_limit=_settings.job_time_limit_seconds * 1000,
    max_age=24 * 3600 * 1000,
)
def process_job(job_id: str) -> None:
    run_job(job_id, store=get_store(), queue=DramatiqQueue(), artifacts=get_artifacts(), settings=get_settings())


def dead_letters() -> list[dict[str, Any]]:
    """List messages currently parked in the dead-letter queue."""
    key = f"{broker.namespace}:{_settings.queue_name}.XQ"
    out: list[dict[str, Any]] = []
    client = get_redis()
    for message_id in cast(list[bytes], client.zrange(key, 0, -1)):
        raw = cast(bytes | None, client.hget(f"{key}.msgs", message_id.decode()))
        if raw:
            message = dramatiq.Message.decode(raw)
            out.append(
                {
                    "message_id": message.message_id,
                    "job_id": message.args[0] if message.args else None,
                    "retries": message.options.get("retries", 0),
                }
            )
    return out

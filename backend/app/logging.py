"""Structured logging with job-ID correlation.

``bind_job`` attaches a job id to every log line emitted from the current
context (thread or task), so a single job can be traced across stages.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

import structlog

_configured = False


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    global _configured
    if _configured:
        return
    shared: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if fmt == "json" else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(level=level.upper(), stream=sys.stderr, format="%(message)s")
    # Third-party request logging is noise at INFO.
    for noisy in ("httpx", "httpcore", "huggingface_hub", "matplotlib", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]


@contextmanager
def bind_job(job_id: str) -> Iterator[None]:
    with structlog.contextvars.bound_contextvars(job_id=job_id):
        yield


@contextmanager
def timed(log: structlog.stdlib.BoundLogger, event: str, **fields: object) -> Iterator[dict[str, float]]:
    """Log ``event`` with its wall-clock duration; yields a dict receiving it."""
    out: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield out
    except Exception:
        out["seconds"] = round(time.perf_counter() - start, 3)
        log.warning(event, outcome="error", **out, **fields)
        raise
    out["seconds"] = round(time.perf_counter() - start, 3)
    log.info(event, outcome="ok", **out, **fields)

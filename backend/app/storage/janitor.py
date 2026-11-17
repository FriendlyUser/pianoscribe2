"""Retention enforcement: purge raw media, stale work files and expired results.

Run as a long-lived process (``python -m app.storage.janitor``) or once
(``--once``). Raw uploads and downloads are normally removed by the pipeline
the moment they are decoded; the janitor is the backstop for abandoned
uploads, crashed jobs and the result TTL.
"""

from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, get_settings
from app.jobs.store import JobStore
from app.logging import configure_logging, get_logger
from app.storage.artifacts import ArtifactStore

log = get_logger(__name__)


@dataclass
class PurgeReport:
    raw_files: int = 0
    work_dirs: int = 0
    results: int = 0


def _newest_mtime(path: Path) -> float:
    newest = path.stat().st_mtime
    if path.is_dir():
        for child in path.rglob("*"):
            try:
                newest = max(newest, child.stat().st_mtime)
            except FileNotFoundError:
                continue
    return newest


def purge(settings: Settings, store: JobStore, artifacts: ArtifactStore, *, now: float | None = None) -> PurgeReport:
    """Delete everything past its retention window."""
    now = now if now is not None else time.time()
    report = PurgeReport()

    if settings.uploads_dir.is_dir():
        for path in settings.uploads_dir.iterdir():
            if path.is_file() and now - path.stat().st_mtime > settings.raw_retention_seconds:
                path.unlink(missing_ok=True)
                report.raw_files += 1

    if settings.work_dir.is_dir():
        for path in settings.work_dir.iterdir():
            if not path.is_dir() or store.is_locked(path.name):
                continue  # never pull files out from under a running job
            if now - _newest_mtime(path) > settings.raw_retention_seconds:
                shutil.rmtree(path, ignore_errors=True)
                report.work_dirs += 1

    if settings.artifacts_dir.is_dir():
        for path in settings.artifacts_dir.iterdir():
            if not path.is_dir():
                continue
            if now - path.stat().st_mtime > settings.result_retention_seconds:
                if path.name.startswith(".staging-"):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    artifacts.delete(path.name)
                    store.delete(path.name)
                report.results += 1

    log.info("janitor.purge", raw_files=report.raw_files, work_dirs=report.work_dirs, results=report.results)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Purge expired media and results.")
    parser.add_argument("--once", action="store_true", help="run a single pass and exit")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    settings.ensure_dirs()
    import redis

    store = JobStore(redis.Redis.from_url(settings.redis_url), record_ttl_seconds=settings.result_retention_seconds)
    artifacts = ArtifactStore(settings.artifacts_dir)
    while True:
        try:
            purge(settings, store, artifacts)
        except Exception as exc:  # noqa: BLE001 - keep the loop alive
            log.exception("janitor.error", error=str(exc))
        if args.once:
            return
        time.sleep(settings.janitor_interval_seconds)


if __name__ == "__main__":
    main()

"""Run the pipeline on a local file or a YouTube link with no queue and no Redis (CLI, CI, tests, evaluation)."""

from __future__ import annotations

import hashlib
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from app.config import Settings
from app.ingest.signatures import sniff
from app.ingest.youtube import parse_video_id, placeholder_title
from app.jobs.keys import upload_job_id, youtube_job_id
from app.jobs.models import Job, JobOptions, JobSource, SourceKind
from app.jobs.store import JobStore
from app.pipeline.runner import Pipeline
from app.storage.artifacts import ArtifactStore


class LocalStore:
    """Just enough of the job store for one pipeline run: keeps the job in memory."""

    def __init__(self, job: Job, on_update: Callable[[Job], None] | None = None) -> None:
        self.job = job
        self._on_update = on_update

    def update(self, job_id: str, **fields: Any) -> Job:
        self.job = self.job.model_copy(update=fields)
        if self._on_update:
            self._on_update(self.job)
        return self.job


def print_progress() -> Callable[[Job], None]:
    shown = ""

    def show(job: Job) -> None:
        nonlocal shown
        line = f"[{job.progress * 100:5.1f}%] {job.message}"
        if line != shown:
            shown = line
            print(line, file=sys.stderr, flush=True)

    return show


def run_local(
    source: Path | str,
    out: Path,
    settings: Settings,
    *,
    solo: bool = False,
    video: bool = False,
    title: str | None = None,
    on_update: Callable[[Job], None] | None = None,
) -> Job:
    """Turn ``source``, a local file or a YouTube link, into a score; the artifacts are copied into ``out``.

    Work files live under ``settings.data_dir``. If a run fails, a second run with the
    same data directory resumes after the last finished model stage.
    """
    options = JobOptions(solo_piano=solo, piano_video=video)
    for path in (settings.uploads_dir, settings.work_dir, settings.artifacts_dir):
        path.mkdir(parents=True, exist_ok=True)
    job = (
        _youtube_job(source, options, title) if isinstance(source, str) else _file_job(source, options, title, settings)
    )
    artifacts = ArtifactStore(settings.artifacts_dir)
    store = LocalStore(job, on_update)
    finished = Pipeline(job, store=cast(JobStore, store), artifacts=artifacts, settings=settings).run()
    out.mkdir(parents=True, exist_ok=True)
    for name in finished.result.artifacts if finished.result else {}:
        stored = artifacts.path(job.id, name)
        if stored is not None:
            shutil.copyfile(stored, out / name)
    return finished


def _youtube_job(url: str, options: JobOptions, title: str | None) -> Job:
    """The job for a YouTube link; the video is fetched by the pipeline, as for the web server."""
    video_id = parse_video_id(url)
    return Job(
        id=youtube_job_id(video_id, options),
        title=title or placeholder_title(video_id),
        source=JobSource(kind=SourceKind.YOUTUBE, video_id=video_id),
        options=options,
    )


def _file_job(source: Path, options: JobOptions, title: str | None, settings: Settings) -> Job:
    """The job for a local file, stored as an upload is."""
    digest = hashlib.sha256()
    with source.open("rb") as fh:
        head = fh.read(4096)
        digest.update(head)
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    sha = digest.hexdigest()
    kind = sniff(head)
    extension = source.suffix.lower().lstrip(".")
    shutil.copyfile(source, settings.uploads_dir / f"{sha}.{extension}")
    return Job(
        id=upload_job_id(sha, options),
        title=title or source.stem.replace("_", " "),
        source=JobSource(
            kind=SourceKind.UPLOAD,
            filename=source.name,
            extension=extension,
            media_kind=kind.value if kind else None,
            content_sha256=sha,
            size_bytes=source.stat().st_size,
        ),
        options=options,
    )

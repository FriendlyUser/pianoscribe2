"""REST + Server-Sent Events API."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from typing import Annotated

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.api import deps
from app.api.schemas import ClientConfig, SubmissionResponse, YouTubeRequest
from app.config import Settings
from app.errors import ValidationError
from app.ingest.signatures import ALLOWED_EXTENSIONS, MediaKind, extension_of
from app.ingest.upload import receive_upload, title_from_filename
from app.ingest.youtube import parse_video_id, placeholder_title
from app.jobs.keys import upload_job_id, youtube_job_id
from app.jobs.models import Job, JobOptions, JobSource, SourceKind
from app.jobs.service import JobQueue, submit_job
from app.jobs.store import JobStore, events_channel
from app.logging import get_logger
from app.storage.artifacts import ARTIFACTS, ArtifactStore

log = get_logger(__name__)
router = APIRouter(prefix="/api")

SettingsDep = Annotated[Settings, Depends(deps.settings_dep)]
StoreDep = Annotated[JobStore, Depends(deps.store_dep)]
QueueDep = Annotated[JobQueue, Depends(deps.queue_dep)]
ArtifactsDep = Annotated[ArtifactStore, Depends(deps.artifacts_dep)]
AsyncRedisDep = Annotated[aioredis.Redis, Depends(deps.async_redis_dep)]
ModelsDep = Annotated[dict[str, str | int | float] | None, Depends(deps.job_models_dep)]

_JOB_ID = re.compile(r"^[0-9a-f]{32}$")
_HEARTBEAT_SECONDS = 15.0
# The exported score is self-contained; forbid everything it does not need,
# including any network access from its inline script.
_SCORE_CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
    "font-src data:; img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'self'"
)


def _checked_job_id(job_id: str) -> str:
    if not _JOB_ID.match(job_id):
        raise ValidationError("job_not_found", "No such job.", status=404)
    return job_id


def _load_job(store: JobStore, job_id: str) -> Job:
    job = store.get(_checked_job_id(job_id))
    if job is None:
        raise ValidationError("job_not_found", "No such job. It may have expired.", status=404)
    return job


@router.get("/health")
def health(store: StoreDep) -> dict[str, str]:
    store.client.ping()
    return {"status": "ok"}


@router.get("/config")
def client_config(settings: SettingsDep) -> ClientConfig:
    return ClientConfig(
        max_upload_bytes=settings.max_upload_bytes,
        max_duration_seconds=settings.max_duration_seconds,
        allowed_extensions=sorted(ALLOWED_EXTENSIONS),
        youtube_enabled=settings.youtube_enabled,
        result_retention_seconds=settings.result_retention_seconds,
        desktop=settings.desktop,
        server_pdf=settings.pdf_renderer != "none",
    )


@router.post("/jobs/upload", status_code=202)
async def create_upload_job(
    request: Request,
    settings: SettingsDep,
    store: StoreDep,
    queue: QueueDep,
    artifacts: ArtifactsDep,
    filename: Annotated[str, Query(min_length=1, max_length=255)],
    models: ModelsDep,
    solo_piano: bool = False,
    piano_video: bool = False,
) -> SubmissionResponse:
    """Upload a media file as the raw request body (streamed straight to disk)."""
    if piano_video and ALLOWED_EXTENSIONS.get(extension_of(filename)) not in (None, MediaKind.ISO_BMFF):
        raise ValidationError(
            "video_required", "The piano-video option reads the picture: upload an MP4 or MOV video.", status=415
        )
    declared = request.headers.get("content-length")
    upload = await receive_upload(
        request.stream(),
        filename=filename,
        declared_length=int(declared) if declared and declared.isdigit() else None,
        settings=settings,
    )
    options = JobOptions(solo_piano=solo_piano, piano_video=piano_video, models=models)
    source = JobSource(
        kind=SourceKind.UPLOAD,
        filename=upload.display_name,
        extension=upload.path.suffix.lstrip("."),
        media_kind=upload.kind.value,
        content_sha256=upload.sha256,
        size_bytes=upload.size,
    )
    submission = await run_in_threadpool(
        submit_job,
        job_id=upload_job_id(upload.sha256, options),
        source=source,
        options=options,
        title=title_from_filename(filename),
        store=store,
        queue=queue,
        artifacts=artifacts,
        settings=settings,
    )
    if submission.cached:
        # Nothing will consume this copy; do not keep raw media around.
        upload.path.unlink(missing_ok=True)
    return SubmissionResponse(job=submission.job, cached=submission.cached, deduplicated=submission.deduplicated)


@router.post("/jobs/youtube", status_code=202)
def create_youtube_job(
    body: YouTubeRequest,
    settings: SettingsDep,
    store: StoreDep,
    queue: QueueDep,
    artifacts: ArtifactsDep,
    models: ModelsDep,
) -> SubmissionResponse:
    if not settings.youtube_enabled:
        raise ValidationError(
            "youtube_disabled", "YouTube links are disabled on this server. Upload a file instead.", status=403
        )
    video_id = parse_video_id(body.url)
    options = JobOptions(solo_piano=body.solo_piano, piano_video=body.piano_video, models=models)
    submission = submit_job(
        job_id=youtube_job_id(video_id, options),
        source=JobSource(kind=SourceKind.YOUTUBE, video_id=video_id),
        options=options,
        title=placeholder_title(video_id),
        store=store,
        queue=queue,
        artifacts=artifacts,
        settings=settings,
    )
    return SubmissionResponse(job=submission.job, cached=submission.cached, deduplicated=submission.deduplicated)


@router.get("/jobs/{job_id}")
def get_job(job_id: str, store: StoreDep) -> Job:
    return _load_job(store, job_id)


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request, store: StoreDep, redis_client: AsyncRedisDep) -> StreamingResponse:
    """Live job updates as Server-Sent Events; closes once the job is terminal."""
    await run_in_threadpool(_load_job, store, job_id)

    async def stream() -> AsyncIterator[bytes]:
        pubsub = redis_client.pubsub()
        # Subscribe before reading the snapshot so no transition is missed.
        await pubsub.subscribe(events_channel(job_id))
        try:
            job = await run_in_threadpool(store.get, job_id)
            if job is None:
                return
            yield f"data: {job.model_dump_json()}\n\n".encode()
            if job.terminal:
                return
            idle = 0.0
            while not await request.is_disconnected():
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message is None:
                    idle += 1.0
                    if idle >= _HEARTBEAT_SECONDS:
                        idle = 0.0
                        yield b": keep-alive\n\n"
                    continue
                idle = 0.0
                payload = message["data"]
                text = payload.decode() if isinstance(payload, bytes) else str(payload)
                yield f"data: {text}\n\n".encode()
                if Job.model_validate_json(text).terminal:
                    return
        except asyncio.CancelledError:
            raise
        finally:
            await pubsub.unsubscribe()
            await pubsub.aclose()  # type: ignore[no-untyped-call]

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


def _download_name(title: str, extension: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-").lower()[:60] or "score"
    return f"{slug}.{extension}"


@router.get("/jobs/{job_id}/artifacts/{name}")
def get_artifact(
    job_id: str, name: str, store: StoreDep, artifacts: ArtifactsDep, download: bool = False
) -> FileResponse:
    job = _load_job(store, job_id)
    path = artifacts.path(job.id, name) if name in ARTIFACTS else None
    if path is None:
        raise ValidationError("artifact_not_found", "That file is not available for this job.", status=404)
    media_type, extension = ARTIFACTS[name]
    headers = {"X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=300"}
    if name.endswith(".html"):
        headers["Content-Security-Policy"] = _SCORE_CSP
    return FileResponse(
        path,
        media_type=media_type,
        filename=_download_name(job.title, extension) if download else None,
        headers=headers,
    )
